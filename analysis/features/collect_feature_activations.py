"""
Collect transcoder feature activations for visualization.

Runs the model on validation data and collects top-activating examples,
logit lens, and activation statistics for each transcoder feature.
Outputs per-feature JSONs compatible with the circuit-tracer frontend.

Each --val_data entry is either a plain path or a 'domain:path' pair.
When a domain label is given, the top-K max-activating examples for each
domain are tracked separately and surfaced as their own quantile in the
output JSON (for example: "Top activations (chat)", "Top activations (fineweb)").

Usage:
    # Single source (no domain label)
    python -m analysis.features.collect_feature_activations \
        --model_path nathu0/transcoder-adapters-R1-Distill-Qwen-7B-l1w0.001-l0-1.4 \
        --val_data hf://nathu0/transcoder-adapters-openthoughts3-stratified-55k/data/val.jsonl

    # Random subset (reproducible): --shuffle (default seed 60) or --shuffle_seed
    python -m analysis.features.collect_feature_activations \
        --model_path ... --val_data org/dataset --max_samples 1000 --shuffle
    python -m analysis.features.collect_feature_activations \
        --model_path ... --val_data org/dataset --max_samples 1000 --shuffle_seed 42

    # Two sources with domain labels (chat + fineweb)
    python -m analysis.features.collect_feature_activations \
        --model_path siddharthmb/2026.TA.gemma2_2b_... \
        --val_data chat:siddharthmb/2026.transcoder-adapters.lmsys-chat-1m-splits \
                   fineweb:science-of-finetuning/fineweb-1m-sample \
        --domain_top_k 10

Output:
    {output_dir}/
    ├── features/               # Per-feature JSON files (circuit-tracer format)
    │   ├── {cantor_id}.json   # cantor_pair(layer, feature) -> unique int
    │   └── ...
    └── feature_metadata.json  # Activation frequencies, domain/region breakdowns

    Browse results locally:
        python -m analysis.features.visualize.feature_dashboard --data_dir {output_dir}
"""

from pathlib import Path

from helpers.log import logger, setup_logging

import argparse
import json
import random
import heapq
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any
from concurrent.futures import ThreadPoolExecutor

import torch
from tqdm import tqdm
from models.auto import AutoModelForCausalLMWithTranscoder, load_tokenizer
from models.tokens import detect_special_tokens, find_token_positions, precompute_regions
from analysis.features.load_val_data import load_val_data


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ActivatingExample:
    """A single activating example for a feature."""
    activation: float
    token_id: int
    position: int
    context_tokens: list[int]
    context_activations: list[float]
    position_in_context: int
    # Metadata
    domain: str
    region: str  # bos, user_marker, question, assistant_marker, think_start, thinking, think_end, answer
    thinking_position: float | None  # 0-1 if in thinking region, else None
    sequence_idx: int

    def __lt__(self, other):
        """For heap comparison (min-heap on activation)."""
        return self.activation < other.activation


@dataclass
class FeatureStats:
    """Stats for a single feature."""
    layer_idx: int
    feature_idx: int

    # Examples (min-heap for top-k, list for reservoir sampling)
    top_k_examples: list = field(default_factory=list)
    random_examples: list = field(default_factory=list)
    random_seen_count: int = 0  # for reservoir sampling

    # Per-domain top-k min-heaps: domain -> list (heap)
    domain_top_k_examples: dict = field(default_factory=lambda: defaultdict(list))

    # Activation count
    activation_count: int = 0

    # Per-domain activation counts
    domain_counts: dict = field(default_factory=lambda: defaultdict(int))

    # Per-region activation counts
    region_counts: dict = field(default_factory=lambda: defaultdict(int))

    # Thinking position histogram (10 bins)
    thinking_position_counts: list = field(default_factory=lambda: [0] * 10)


# =============================================================================
# Feature Collector
# =============================================================================


def _run_backbone_for_hooks(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> None:
    """Run the transformer stack only (no ``lm_head``).

    Feature hooks are registered on ``model.model.layers[*].mlp``.  A full
    ``CausalLM`` forward also builds logits of shape ``[batch, seq, vocab]``,
    which dominates memory and can OOM inside ``final_logit_softcapping`` even
    when logits are unused for this analysis.
    """
    inner = getattr(model, "model", None)
    if inner is None:
        raise RuntimeError("Expected a HuggingFace CausalLM with a `.model` backbone.")
    for layer in inner.layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "_attention_mask"):
            mlp._attention_mask = attention_mask
    inner(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)


class FeatureCollector:
    """Collects feature activation statistics across sequences."""

    def __init__(
        self,
        n_layers: int,
        n_features: int,
        top_k: int = 20,
        n_random: int = 10,
        domain_top_k: int = 10,
        context_before: int = 50,
        context_after: int = 20,
    ):
        self.n_layers = n_layers
        self.n_features = n_features
        self.top_k = top_k
        self.n_random = n_random
        self.domain_top_k = domain_top_k
        self.context_before = context_before
        self.context_after = context_after

        # Per-feature stats
        self.stats = [
            [FeatureStats(layer_idx=l, feature_idx=f) for f in range(n_features)]
            for l in range(n_layers)
        ]

        # Global counts for normalization
        self.total_tokens = 0
        self.tokens_per_domain: dict[str, int] = defaultdict(int)
        self.tokens_per_region: dict[str, int] = defaultdict(int)
        self.tokens_per_thinking_bin: list[int] = [0] * 10

        # Hook storage
        self._hooks: list = []
        self._layer_activations: dict[int, torch.Tensor] = {}

    def _make_hook(self, layer_idx: int):
        """Create a forward hook that captures transcoder activations.

        Hooks ``mlp.transcoder_enc`` (the Linear), not the full MLP.  The MLP
        forward already runs ``transcoder_enc`` once; a hook on ``mlp`` would
        re-run the encoder and double matmul cost per layer.
        """
        def hook(module, input, output):
            # module is transcoder_enc; output is pre-ReLU [batch, seq, n_features]
            with torch.no_grad():
                features = torch.relu(output)
            self._layer_activations[layer_idx] = features

        return hook

    def register_hooks(self, model):
        """Register forward hooks on all transcoder encoder linears."""
        self._hooks = []
        for layer_idx, layer in enumerate(model.model.layers):
            enc = layer.mlp.transcoder_enc
            hook = enc.register_forward_hook(self._make_hook(layer_idx))
            self._hooks.append(hook)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def _get_context(self, tokens: list[int], position: int) -> tuple[list[int], int]:
        """Extract context window around a position."""
        start = max(0, position - self.context_before)
        end = min(len(tokens), position + self.context_after + 1)
        return tokens[start:end], position - start

    def _maybe_add_example(
        self,
        stats: FeatureStats,
        activation: float,
        tokens: list[int],
        position: int,
        features_gpu: torch.Tensor,  # full tensor on GPU
        feature_idx: int,
        domain: str,
        region: str,
        thinking_position: float | None,
        sequence_idx: int,
    ):
        """Add example to top-k heap, domain top-k heap, and/or random reservoir."""
        # Check if this could make it into global top-k
        dominated_by_heap = (
            len(stats.top_k_examples) >= self.top_k
            and activation <= stats.top_k_examples[0].activation
        )

        # Check if this could make it into per-domain top-k
        domain_heap = stats.domain_top_k_examples[domain]
        dominated_by_domain_heap = (
            len(domain_heap) >= self.domain_top_k
            and activation <= domain_heap[0].activation
        )

        # Reservoir sampling decision for random
        stats.random_seen_count += 1
        add_to_random = False
        random_replace_idx = None
        if len(stats.random_examples) < self.n_random:
            add_to_random = True
        else:
            j = random.randint(0, stats.random_seen_count - 1)
            if j < self.n_random:
                add_to_random = True
                random_replace_idx = j

        # Skip if not going into any buffer
        if dominated_by_heap and dominated_by_domain_heap and not add_to_random:
            return

        # Create example (only fetch context from GPU when actually keeping)
        context_tokens, pos_in_ctx = self._get_context(tokens, position)
        ctx_start = max(0, position - self.context_before)
        ctx_end = min(len(tokens), position + self.context_after + 1)
        # Fetch context window from GPU (only for kept examples)
        context_activations = features_gpu[ctx_start:ctx_end, feature_idx].float().cpu().tolist()

        example = ActivatingExample(
            activation=activation,
            token_id=tokens[position],
            position=position,
            context_tokens=context_tokens,
            context_activations=context_activations,
            position_in_context=pos_in_ctx,
            domain=domain,
            region=region,
            thinking_position=thinking_position,
            sequence_idx=sequence_idx,
        )

        # Add to global top-k heap
        if not dominated_by_heap:
            if len(stats.top_k_examples) < self.top_k:
                heapq.heappush(stats.top_k_examples, example)
            else:
                heapq.heapreplace(stats.top_k_examples, example)

        # Add to per-domain top-k heap
        if not dominated_by_domain_heap:
            if len(domain_heap) < self.domain_top_k:
                heapq.heappush(domain_heap, example)
            else:
                heapq.heapreplace(domain_heap, example)

        # Add to random reservoir
        if add_to_random:
            if random_replace_idx is None:
                stats.random_examples.append(example)
            else:
                stats.random_examples[random_replace_idx] = example

    def process_batch(
        self,
        model,
        batch_tokens: list[list[int]],
        batch_domains: list[str],
        batch_markers: list[dict],
        batch_seq_idxs: list[int],
        pad_token_id: int = 0,
    ):
        """Process a batch of sequences in one forward pass."""
        B = len(batch_tokens)
        seq_lens = [len(t) for t in batch_tokens]
        max_len = max(seq_lens)
        self._layer_activations = {}

        # Right-pad to max length in batch
        padded = torch.full((B, max_len), pad_token_id, dtype=torch.long, device=model.device)
        attention_mask = torch.zeros(B, max_len, dtype=torch.long, device=model.device)
        for b, tokens in enumerate(batch_tokens):
            L = len(tokens)
            padded[b, :L] = torch.tensor(tokens, dtype=torch.long)
            attention_mask[b, :L] = 1

        # Single backbone forward (hooks capture activations; skip lm_head logits)
        with torch.inference_mode():
            _run_backbone_for_hooks(model, padded, attention_mask)

        # Process each item in the batch
        for b in range(B):
            tokens = batch_tokens[b]
            seq_len = seq_lens[b]
            domain = batch_domains[b]
            markers = batch_markers[b]
            sequence_idx = batch_seq_idxs[b]

            regions, thinking_positions = precompute_regions(tokens, markers)

            self.total_tokens += seq_len
            self.tokens_per_domain[domain] += seq_len
            for pos in range(seq_len):
                self.tokens_per_region[regions[pos]] += 1
                think_pos_val = thinking_positions[pos]
                if think_pos_val is not None:
                    bin_idx = min(9, int(think_pos_val * 10))
                    self.tokens_per_thinking_bin[bin_idx] += 1

            for layer_idx in range(self.n_layers):
                # Slice out this item's real tokens (right-padded, so [:seq_len] is correct)
                features_gpu = self._layer_activations[layer_idx][b, :seq_len]  # [seq_len, n_features]

                nonzero = torch.nonzero(features_gpu > 0)  # [N, 2]
                if len(nonzero) == 0:
                    continue

                active_positions = nonzero[:, 0].cpu().numpy()
                active_features = nonzero[:, 1].cpu().numpy()
                active_values = features_gpu[nonzero[:, 0], nonzero[:, 1]].float().cpu().numpy()

                for i in range(len(active_positions)):
                    pos = int(active_positions[i])
                    feature_idx = int(active_features[i])
                    act = float(active_values[i])

                    stats = self.stats[layer_idx][feature_idx]
                    region = regions[pos]
                    think_pos = thinking_positions[pos]

                    stats.activation_count += 1
                    stats.domain_counts[domain] += 1
                    stats.region_counts[region] += 1
                    if think_pos is not None:
                        bin_idx = min(9, int(think_pos * 10))
                        stats.thinking_position_counts[bin_idx] += 1

                    self._maybe_add_example(
                        stats, act, tokens, pos, features_gpu, feature_idx,
                        domain, region, think_pos, sequence_idx
                    )

        self._layer_activations = {}


# =============================================================================
# Logit Lens
# =============================================================================

def compute_logit_lens(model, tokenizer, top_k: int = 10) -> list[dict]:
    """
    Compute top/bottom unembed tokens for each feature.

    For each feature, computes decoder @ unembed.T to find which tokens
    the feature most strongly promotes/suppresses.
    """
    logger.info("Computing logit lens...")

    # Get unembedding matrix
    unembed = model.lm_head.weight.data  # [vocab_size, d_model]

    logit_lens_data = []

    for layer_idx, layer in enumerate(tqdm(model.model.layers, desc="Logit lens")):
        mlp = layer.mlp
        # Decoder: [d_model, n_features] - each column is a feature's output direction
        decoder = mlp.transcoder_dec.weight.data  # [d_model, n_features]

        # Compute logits for each feature: unembed @ decoder = [vocab_size, n_features]
        with torch.no_grad():
            logits = unembed @ decoder  # [vocab_size, n_features]

        # Get top and bottom tokens for each feature
        top_vals, top_ids = logits.topk(top_k, dim=0)  # [top_k, n_features]
        bot_vals, bot_ids = logits.topk(top_k, dim=0, largest=False)

        layer_data = {
            'top_ids': top_ids.T.cpu().tolist(),  # [n_features, top_k]
            'top_vals': top_vals.T.cpu().tolist(),
            'bot_ids': bot_ids.T.cpu().tolist(),
            'bot_vals': bot_vals.T.cpu().tolist(),
        }
        logit_lens_data.append(layer_data)

    return logit_lens_data


# =============================================================================
# Export Functions
# =============================================================================

def cantor_pair(x: int, y: int) -> int:
    """Cantor pairing function - maps (layer, feature) to unique integer."""
    return (x + y) * (x + y + 1) // 2 + y


def format_example_for_circuit_tracer(ex: ActivatingExample, tokenizer) -> dict:
    """Format an ActivatingExample for circuit tracer JSON."""
    tokens = [tokenizer.decode([tok_id]) for tok_id in ex.context_tokens]
    return {
        "tokens": tokens,
        "tokens_acts_list": ex.context_activations,
        "train_token_ind": ex.position_in_context,
        "peak_activation": ex.activation,
        "is_repeated_datapoint": False,
    }


def _write_feature_json(args: tuple) -> None:
    """Write a single feature JSON file (for parallel execution)."""
    filepath, feature_json = args
    with open(filepath, 'w') as f:
        json.dump(feature_json, f)


def export_circuit_tracer_json(
    collector: FeatureCollector,
    logit_lens_data: list[dict],
    tokenizer,
    output_dir: Path,
    n_workers: int = 16,
):
    """Export feature data to circuit tracer JSON format."""
    features_dir = output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting features to {features_dir}...")

    # First pass: build all JSON objects
    write_tasks = []  # list of (filepath, json_dict)
    skipped = 0

    for layer_idx in tqdm(range(collector.n_layers), desc="Building JSON"):
        layer_logit_lens = logit_lens_data[layer_idx]

        for feature_idx in range(collector.n_features):
            stats = collector.stats[layer_idx][feature_idx]

            # Skip empty features
            if stats.activation_count == 0:
                skipped += 1
                continue

            # Get examples (sorted by activation for top-k)
            top_examples = sorted(stats.top_k_examples, key=lambda x: -x.activation)
            random_examples = stats.random_examples

            # Format examples
            top_formatted = [format_example_for_circuit_tracer(ex, tokenizer) for ex in top_examples]
            random_formatted = [format_example_for_circuit_tracer(ex, tokenizer) for ex in random_examples]

            # Compute activation range from examples
            all_acts = []
            for ex in top_examples + random_examples:
                all_acts.extend(ex.context_activations)
            act_min = min(all_acts) if all_acts else 0.0
            act_max = max(all_acts) if all_acts else 1.0

            # Get logit lens tokens
            top_logits = [tokenizer.decode([tok_id]) for tok_id in layer_logit_lens['top_ids'][feature_idx]]
            bottom_logits = [tokenizer.decode([tok_id]) for tok_id in layer_logit_lens['bot_ids'][feature_idx]]

            # Per-domain top examples (sorted by descending activation)
            domain_quantiles = []
            for domain_name, domain_heap in sorted(stats.domain_top_k_examples.items()):
                domain_sorted = sorted(domain_heap, key=lambda x: -x.activation)
                domain_formatted = [
                    format_example_for_circuit_tracer(ex, tokenizer) for ex in domain_sorted
                ]
                domain_quantiles.append({
                    "quantile_name": f"Top activations ({domain_name})",
                    "examples": domain_formatted,
                })

            # Build JSON
            feature_json = {
                "top_logits": top_logits,
                "bottom_logits": bottom_logits,
                "act_min": act_min,
                "act_max": act_max,
                "examples_quantiles": [
                    {"quantile_name": "Top activations", "examples": top_formatted},
                    *domain_quantiles,
                    {"quantile_name": "Random samples", "examples": random_formatted},
                ],
                "activation_frequency": stats.activation_count / max(1, collector.total_tokens),
                "layer": layer_idx,
                "feature": feature_idx,
            }

            cantor_id = cantor_pair(layer_idx, feature_idx)
            filepath = features_dir / f"{cantor_id}.json"
            write_tasks.append((filepath, feature_json))

    # Second pass: write files in parallel
    logger.info(f"Writing {len(write_tasks)} files with {n_workers} workers...")
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        list(tqdm(executor.map(_write_feature_json, write_tasks), total=len(write_tasks), desc="Writing files"))

    logger.info(f"Generated {len(write_tasks)} feature files, skipped {skipped} empty features")


def export_metadata(collector: FeatureCollector, output_dir: Path):
    """Export rich metadata to JSON for analysis."""
    logger.info("Exporting metadata...")

    metadata: dict[str, Any] = {
        # Global counts
        "total_tokens": collector.total_tokens,
        "tokens_per_domain": dict(collector.tokens_per_domain),
        "tokens_per_region": dict(collector.tokens_per_region),
        "tokens_per_thinking_bin": collector.tokens_per_thinking_bin,

        # Per-feature stats
        "features": [],
    }

    for layer_idx in range(collector.n_layers):
        for feature_idx in range(collector.n_features):
            stats = collector.stats[layer_idx][feature_idx]

            if stats.activation_count == 0:
                continue

            total_acts = stats.activation_count

            # Domain distributions
            domain_density = {}
            domain_fraction = {}
            for domain, count in stats.domain_counts.items():
                domain_tokens = collector.tokens_per_domain.get(domain, 0)
                domain_density[domain] = count / domain_tokens if domain_tokens > 0 else 0
                domain_fraction[domain] = count / total_acts

            # Region distributions
            region_density = {}
            region_fraction = {}
            for region, count in stats.region_counts.items():
                region_tokens = collector.tokens_per_region.get(region, 0)
                region_density[region] = count / region_tokens if region_tokens > 0 else 0
                region_fraction[region] = count / total_acts

            # Thinking position distributions
            thinking_density = []
            thinking_fraction = []
            thinking_total = sum(stats.thinking_position_counts)
            for bin_idx in range(10):
                bin_acts = stats.thinking_position_counts[bin_idx]
                bin_tokens = collector.tokens_per_thinking_bin[bin_idx]
                thinking_density.append(bin_acts / bin_tokens if bin_tokens > 0 else 0)
                thinking_fraction.append(bin_acts / thinking_total if thinking_total > 0 else 0)

            feature_meta = {
                "layer": layer_idx,
                "feature": feature_idx,
                "cantor_id": cantor_pair(layer_idx, feature_idx),
                "activation_count": total_acts,
                "activation_freq": total_acts / collector.total_tokens,
                "domain_density": domain_density,
                "domain_fraction": domain_fraction,
                "region_density": region_density,
                "region_fraction": region_fraction,
                "thinking_position_density": thinking_density,
                "thinking_position_fraction": thinking_fraction,
            }
            metadata["features"].append(feature_meta)

    with open(output_dir / "feature_metadata.json", 'w') as f:
        json.dump(metadata, f)

    logger.info(f"Saved metadata for {len(metadata['features'])} features")


# =============================================================================
# Main
# =============================================================================

# Default seed when --shuffle is passed without --shuffle_seed (reproducible subset).
DEFAULT_SHUFFLE_SEED = 60

# Default seed for reordering formed batches (execution order only; independent of row shuffle).
DEFAULT_BATCH_SHUFFLE_SEED = 72


def _dataset_indices_to_process(
    n_dataset: int,
    max_samples: int | None,
    shuffle_seed: int | None,
    source_idx: int,
) -> list[int]:
    """Row indices to load from a val source: sequential prefix, or seeded shuffle.

    When ``shuffle_seed`` is set, indices are ``torch.randperm(n_dataset)[:n_take]``
    with a ``torch.Generator`` seeded per-source so different ``--val_data`` sources
    do not share the same permutation when split sizes coincide.
    """
    n_take = min(max_samples, n_dataset) if max_samples is not None else n_dataset
    if shuffle_seed is None:
        return list(range(n_take))
    per_source_seed = int(shuffle_seed) + source_idx * 100_003
    g = torch.Generator()
    g.manual_seed(per_source_seed)
    perm = torch.randperm(n_dataset, generator=g)
    return perm[:n_take].tolist()


def _parse_val_data_entry(entry: str) -> tuple[str | None, str]:
    """Parse a val_data entry of the form 'domain:path' or just 'path'.

    Returns (domain, path). Domain is None if not specified.
    Handles paths starting with hf://, http://, https:// without stripping the scheme.
    """
    # Don't split on ':' in URL schemes or Windows drive letters (single char before ':')
    colon_idx = entry.find(':')
    if colon_idx > 1:
        prefix = entry[:colon_idx]
        rest = entry[colon_idx + 1:]
        # Only treat as domain:path if the prefix looks like a short domain label
        # (no slashes, not a URL scheme like 'hf' which is length 2 but that's fine
        # since hf:// has two slashes after)
        if '/' not in prefix and not rest.startswith('//'):
            return prefix, rest
    return None, entry


def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Collect transcoder feature activations for visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="HF repo ID or local path to transcoder checkpoint")
    parser.add_argument("--val_data", type=str, nargs='+', required=True,
                        help=(
                            "Validation data source(s). Each entry is either 'path' or 'domain:path'. "
                            "Examples: "
                            "chat:siddharthmb/lmsys-splits "
                            "fineweb:hf://org/dataset/data/val.jsonl"
                        ))
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: PRODUCTS_DIR/feature_data/<model>_<timestamp>)")

    # Optional args
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to process per data source (default: all)")
    parser.add_argument("--shuffle", action="store_true",
                        help=(
                            "Shuffle which rows are used (torch.Generator + torch.randperm). "
                            f"Uses seed {DEFAULT_SHUFFLE_SEED} unless --shuffle_seed is set."
                        ))
    parser.add_argument("--shuffle_seed", type=int, default=None,
                        help=(
                            "If set, sample rows with this seed instead of sequential order "
                            "(same mechanism as --shuffle). Overrides the default seed from "
                            f"--shuffle ({DEFAULT_SHUFFLE_SEED}). "
                            "Each --val_data source uses an independent seed offset."
                        ))
    parser.add_argument("--top_k", type=int, default=20,
                        help="Number of global top activating examples per feature")
    parser.add_argument("--domain_top_k", type=int, default=10,
                        help="Number of top activating examples per feature per domain")
    parser.add_argument("--n_random", type=int, default=10,
                        help="Number of random samples per feature")
    parser.add_argument("--context_before", type=int, default=50,
                        help="Context tokens before activating token")
    parser.add_argument("--context_after", type=int, default=20,
                        help="Context tokens after activating token")
    parser.add_argument("--batch_size", type=int, default=40,
                        help=(
                            "Max sequences per forward pass. GPU memory budget is auto-computed; "
                            "this caps CPU-side work (per-token bookkeeping) per batch. "
                            "Lower if the CPU bottleneck stalls the GPU on short sequences."
                        ))
    parser.add_argument(
        "--shuffle_batches",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After length-sorting and packing into batches (padding-efficient), randomly reorder "
            "**which batch runs first** using torch.randperm with --shuffle_batches_seed. "
            "Total compute is identical; only iteration order changes, so per-step progress/time "
            "estimates (e.g. tqdm) are less skewed by many cheap short batches at the start. "
            "Default: on. Use --no-shuffle_batches to run batches in strict length order (shortest "
            "batches first)."
        ),
    )
    parser.add_argument(
        "--shuffle_batches_seed",
        type=int,
        default=DEFAULT_BATCH_SHUFFLE_SEED,
        help=(
            "Seed for --shuffle_batches (ignored with --no-shuffle_batches). "
            f"Default: {DEFAULT_BATCH_SHUFFLE_SEED}."
        ),
    )
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Explicit tokenizer path (default: resolved from model_type)")
    parser.add_argument("--max_length", type=int, default=10000,
                        help="Max sequence length (longer sequences truncated)")

    args = parser.parse_args()

    if args.shuffle_seed is not None:
        shuffle_seed: int | None = args.shuffle_seed
    elif args.shuffle:
        shuffle_seed = DEFAULT_SHUFFLE_SEED
    else:
        shuffle_seed = None

    if args.output_dir is None:
        from helpers.paths import PRODUCTS_DIR, SLURM_JOB_ID
        from datetime import datetime
        # Truncate model path: take last component
        model_slug = args.model_path.rstrip("/").split("/")[-1]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(PRODUCTS_DIR / "feature_data" / f"{model_slug}_{timestamp}_{SLURM_JOB_ID}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    tokenizer = load_tokenizer(args.model_path, tokenizer_path=args.tokenizer)

    # Load model
    logger.info(f"Loading model: {args.model_path}")
    model = AutoModelForCausalLMWithTranscoder.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    n_layers = len(model.model.layers)
    first_mlp = next(model._transcoder_mlps())  # type: ignore[operator]
    n_features = first_mlp.n_features
    model_type = model.config.model_type
    logger.info(f"Model: {n_layers} layers, {n_features} features per layer, arch={model_type}")

    # Detect special tokens for this architecture/tokenizer
    special_tokens = detect_special_tokens(tokenizer, model_type=model_type)
    has_thinking = special_tokens.think_start is not None and special_tokens.think_end is not None
    logger.info(f"Detected special tokens: {special_tokens}")
    if not has_thinking:
        logger.info("Note: No <think> tags detected — thinking region analysis will be skipped")

    # Parse and load all val_data sources
    val_data_sources: list[tuple[str | None, str]] = [
        _parse_val_data_entry(entry) for entry in args.val_data
    ]
    logger.info(f"Loading {len(val_data_sources)} data source(s):")
    loaded_sources: list[tuple[Any, list[dict] | None]] = []
    for domain_label, path in val_data_sources:
        logger.info(f"  {path!r} (domain={domain_label!r})")
        dataset, examples_meta = load_val_data(path, tokenizer, args.max_length, domain=domain_label)
        loaded_sources.append((dataset, examples_meta))

    total_samples = sum(
        min(args.max_samples, len(ds)) if args.max_samples else len(ds)
        for ds, _ in loaded_sources
    )

    if args.max_samples is not None:
        logger.info(f"Subset: --max_samples={args.max_samples} (per --val_data source):")
        for source_idx, (dataset, _) in enumerate(loaded_sources):
            path = val_data_sources[source_idx][1]
            n_dataset = len(dataset)
            n_take = min(args.max_samples, n_dataset)
            if n_take < n_dataset:
                logger.info(
                    f"  [{source_idx + 1}] {path!r}: using {n_take} of {n_dataset} rows"
                )
            else:
                logger.info(
                    f"  [{source_idx + 1}] {path!r}: using all {n_dataset} rows "
                    f"(split size ≤ --max_samples)"
                )

    if shuffle_seed is not None:
        seed_src = "--shuffle_seed" if args.shuffle_seed is not None else "--shuffle (default seed)"
        logger.info(
            "Shuffling: row indices are a seeded random permutation (torch.randperm), "
            "not sequential dataset order. "
            f"seed={shuffle_seed} ({seed_src}); each source adds an independent offset to this seed."
        )

    logger.info(f"Processing {total_samples} samples total across {len(loaded_sources)} source(s)")

    # Create collector
    collector = FeatureCollector(
        n_layers=n_layers,
        n_features=n_features,
        top_k=args.top_k,
        n_random=args.n_random,
        domain_top_k=args.domain_top_k,
        context_before=args.context_before,
        context_after=args.context_after,
    )

    # Pre-extract all items across sources, filtering malformed ones
    items: list[tuple[list[int], str, dict]] = []
    skipped = 0
    for source_idx, (dataset, examples_meta) in enumerate(loaded_sources):
        domain_label = val_data_sources[source_idx][0]
        n_dataset = len(dataset)
        indices = _dataset_indices_to_process(
            n_dataset, args.max_samples, shuffle_seed, source_idx
        )
        source_desc = f"source {source_idx + 1}/{len(loaded_sources)}"
        if domain_label:
            source_desc += f" ({domain_label})"

        for idx in tqdm(indices, desc=f"Preparing {source_desc}"):
            item = dataset[idx]
            meta = examples_meta[idx] if examples_meta is not None else {}

            tokens = item['input_ids']
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.tolist()
            domain = meta.get('domain', domain_label or 'unknown')
            markers = find_token_positions(tokens, special_tokens)

            if has_thinking and (markers['think_start'] is None or markers['think_end'] is None):
                if skipped < 5:
                    logger.warning(f"Skipping sample {idx} (source {source_idx}) - missing <think> tags")
                skipped += 1
                continue

            items.append((tokens, domain, markers))

    if skipped > 0:
        logger.warning(f"Skipped {skipped} samples due to missing <think> tags")

    if not items:
        logger.error(
            "No sequences left to process after filtering "
            f"(has_thinking={has_thinking}, skipped={skipped}). Exiting."
        )
        return

    # Sort by length so similarly-sized sequences are batched together (less padding waste)
    items.sort(key=lambda x: len(x[0]))
    logger.info(f"Sorted {len(items)} sequences by length "
                f"(shortest={len(items[0][0])}, longest={len(items[-1][0])})")

    # Token budget from GPU memory.  Collection uses the transformer backbone only
    # (no lm_head logits), so the dominant term is hook tensors: each layer keeps a
    # bf16 [batch, seq, n_features] activation until the forward finishes.
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    bytes_per_element = 2  # bf16
    hook_bytes_per_token = n_layers * n_features * bytes_per_element
    hidden = int(getattr(model.config, "hidden_size", 0) or 0)
    # Small slack for attention / scratch (far smaller than full vocab logits).
    scratch_bytes_per_token = max(4096, hidden * 16)
    slack = 1.35
    total_bytes_per_token = int(hook_bytes_per_token * slack) + scratch_bytes_per_token
    free_bytes = torch.cuda.mem_get_info(model.device)[0]
    safety_margin = 0.55
    max_batch_tokens = int(free_bytes * safety_margin / total_bytes_per_token)
    logger.info(f"GPU free memory: {free_bytes / 1e9:.1f} GB, "
                f"per-token budget: {total_bytes_per_token / 1e6:.1f} MB "
                f"(hooks {hook_bytes_per_token / 1e6:.1f} × {slack:.2f} + scratch {scratch_bytes_per_token / 1e6:.1f}), "
                f"token budget: {max_batch_tokens:,}")
    batches: list[list[tuple[list[int], str, dict]]] = []
    current_batch: list[tuple[list[int], str, dict]] = []
    current_max_len = 0
    for item in items:
        item_len = len(item[0])
        new_max_len = max(current_max_len, item_len)
        padded_tokens = (len(current_batch) + 1) * new_max_len
        if current_batch and (len(current_batch) >= args.batch_size or padded_tokens > max_batch_tokens):
            batches.append(current_batch)
            current_batch = [item]
            current_max_len = item_len
        else:
            current_batch.append(item)
            current_max_len = new_max_len
    if current_batch:
        batches.append(current_batch)

    batch_sizes = [len(b) for b in batches]
    logger.info(f"Formed {len(batches)} batches (sizes {min(batch_sizes)}-{max(batch_sizes)}, "
                f"token budget={max_batch_tokens:,})")

    if args.shuffle_batches:
        g = torch.Generator()
        g.manual_seed(int(args.shuffle_batches_seed))
        order = torch.randperm(len(batches), generator=g).tolist()
        batches = [batches[i] for i in order]
        logger.info(
            "Shuffling batch execution order: %s batches permuted with torch.randperm "
            "(seed=%s). Batch membership is unchanged; only run order differs so step times are "
            "mixed. Disable with --no-shuffle_batches for shortest-batches-first order.",
            len(batches),
            args.shuffle_batches_seed,
        )
    else:
        logger.info(
            "Batch execution order: sequential after length-aware packing (shortest batches "
            "first; progress may look fast early then slow). Enable default --shuffle_batches to "
            "interleave cheap and expensive steps."
        )

    # Process batches
    collector.register_hooks(model)
    sequence_idx = 0
    for batch in tqdm(batches, desc="Processing batches"):
        batch_tokens = [t for t, _, _ in batch]
        batch_domains = [d for _, d, _ in batch]
        batch_markers = [m for _, _, m in batch]
        batch_seq_idxs = list(range(sequence_idx, sequence_idx + len(batch)))

        collector.process_batch(model, batch_tokens, batch_domains, batch_markers,
                                batch_seq_idxs, pad_token_id)
        sequence_idx += len(batch)

    collector.remove_hooks()

    # Summary stats
    logger.info("Collection summary:")
    logger.info(f"  Total tokens: {collector.total_tokens:,}")
    logger.info(f"  Domains: {dict(collector.tokens_per_domain)}")
    logger.info(f"  Regions: {dict(collector.tokens_per_region)}")

    # Compute logit lens
    logit_lens_data = compute_logit_lens(model, tokenizer)

    # Export
    export_circuit_tracer_json(collector, logit_lens_data, tokenizer, output_dir)
    export_metadata(collector, output_dir)

    logger.info(f"Done! Output written to {output_dir}")
    logger.info("  features/: Circuit tracer JSON files")
    logger.info("  feature_metadata.json: Rich metadata for analysis")


if __name__ == "__main__":
    main()
