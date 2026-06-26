"""
Collect transcoder feature activations for visualization.

Runs the model on validation data and collects top-activating examples,
logit lens, and activation statistics for each transcoder feature.
Outputs per-feature JSONs compatible with the circuit-tracer frontend.

Each --val_data entry is either a plain path or a 'domain:path' pair.
When a domain label is given, the top-K max-activating examples for each
domain are tracked separately and surfaced as their own quantile in the
output JSON (for example: "Top activations (chat)", "Top activations (fineweb)").
All positive activations are also aggregated into fixed activation-magnitude
histograms, globally and per domain. Dense per-feature histogram arrays are
stored in activation_histograms.npz; feature_metadata.json stores scalar
summaries for sorting and dashboard display. Dashboard histogram views show raw
counts, token-normalized density by domain, and conditional magnitude
distributions by domain. Metadata also includes feature_frequency_summary with
per-feature firing-frequency histograms and global nonzero token-feature density
by domain, so run-level chat-vs-web sparsity differences are visible. Configured
activation bands are reservoir-sampled into feature JSON example tabs such as
Activation range 2.5-3.0 (chat), so medium non-top activations can be inspected
directly.

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
    ├── activation_histograms.npz  # Exact all-nonzero activation histograms
    ├── feature_metadata.json  # Activation frequencies, domain/region breakdowns
    ├── collect_feature_activations_args.json  # Full parsed CLI settings
    ├── collect_feature_activations_command.sh  # Pasteable replay command
    └── circuit_tracer_features/  # Packed cache for circuit-tracer local features
        ├── index.json.gz
        ├── layer_0.bin
        └── ...

    Browse results locally:
        python -m analysis.features.visualize.feature_dashboard --data_dir {output_dir}
"""

from pathlib import Path

from helpers.log import logger, setup_logging

import argparse
import gzip
import json
import random
import heapq
import shlex
import struct
import textwrap
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait

import numpy as np
import torch
from tqdm import tqdm
from helpers.paths.output_path import generate_output_path
from analysis.features.batching import compute_max_batch_tokens, form_length_packed_batches
from analysis.features.collection_wandb import CollectionWandbLogger, add_collection_wandb_args
from models.auto import AutoModelForCausalLMWithTranscoder, load_tokenizer
from models.tokens import detect_special_tokens, find_token_positions, precompute_regions
from analysis.features.activation_histograms import (
    DEFAULT_ACTIVATION_EXAMPLE_RANGES,
    DEFAULT_HISTOGRAM_BIN_LOWER_BOUNDS,
    compute_relative_domain_scores,
    format_activation_range_label,
    histogram_bin_indices,
    parse_activation_example_ranges,
    save_activation_histograms_npz,
)
from analysis.features.annotate.annotate_assistant_response_features import (
    AssistantResponseFeatureAnnotator,
)
from analysis.features.annotate.annotation_framework import run_annotation
from analysis.features.load_val_data import (
    FeatureDataSourceSettings,
    load_val_data_from_settings,
)


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
    source_metadata: dict[str, Any] = field(default_factory=dict)

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

    # Activation-range examples for inspecting non-top medium activations.
    # Key format: "{domain}|{lo}-{hi}".
    activation_range_examples: dict = field(default_factory=lambda: defaultdict(list))
    activation_range_seen_counts: dict = field(default_factory=lambda: defaultdict(int))

    # Bounded top-token counts: token_id -> count of activations on that token, kept to a
    # fixed cap via Space-Saving. Powers the graph-viz "% of this feature's activations that
    # land on this token" (token-conditional specificity) stat.
    token_counts: dict = field(default_factory=dict)


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
        domain_names: list[str] | None = None,
        hist_bin_lower_bounds: np.ndarray | None = None,
        activation_example_ranges: list[tuple[float, float]] | None = None,
        activation_range_examples_per_domain: int = 1,
        track_token_counts: bool = True,
        token_count_cap: int = 32,
    ):
        self.n_layers = n_layers
        self.n_features = n_features
        self.top_k = top_k
        self.n_random = n_random
        self.domain_top_k = domain_top_k
        self.context_before = context_before
        self.context_after = context_after
        # Token-conditional specificity tracking (graph-viz "% of this feature's activations
        # on this token"). Bounded per feature to token_count_cap entries via Space-Saving.
        self.track_token_counts = track_token_counts
        self.token_count_cap = max(0, token_count_cap)

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
        # Exact activation histogram storage.
        self.domain_names = list(domain_names or [])
        self.domain_to_index = {
            domain: i
            for i, domain in enumerate(self.domain_names)
        }
        self.hist_bin_lower_bounds = np.asarray(
            hist_bin_lower_bounds
            if hist_bin_lower_bounds is not None
            else DEFAULT_HISTOGRAM_BIN_LOWER_BOUNDS,
            dtype=np.float32,
        )
        n_bins = len(self.hist_bin_lower_bounds)
        n_domains = len(self.domain_names)
        self.run_hist_total = np.zeros(n_bins, dtype=np.uint64)
        self.run_hist_by_domain = np.zeros((n_domains, n_bins), dtype=np.uint64)
        self.feature_hist_total_by_layer = [
            np.zeros((n_features, n_bins), dtype=np.uint32)
            for _ in range(n_layers)
        ]
        self.feature_hist_by_domain_by_layer = [
            np.zeros((n_domains, n_features, n_bins), dtype=np.uint32)
            for _ in range(n_layers)
        ]
        self.activation_example_ranges = list(activation_example_ranges or [])
        self.activation_range_examples_per_domain = max(
            0,
            activation_range_examples_per_domain,
        )

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

    def merge_from(self, other: "FeatureCollector") -> None:
        """Merge another collector's state into this one (for sharded collection).

        Each shard runs an independent collection over a disjoint subset of sequences; this
        recombines them into the exact same result a single run would produce, EXCEPT for the
        reservoir-sampled fields (``random_examples`` and ``activation_range_examples``), which
        cannot be merged without bias. Sharding therefore requires those disabled
        (``--n_random 0`` and no activation ranges) -- the base/GemmaScope collector's config --
        and we assert it here. Top-k / domain-top-k example heaps merge by union-then-keep-k;
        histograms and all counts add.
        """
        assert (self.n_layers, self.n_features, self.top_k, self.domain_top_k) == (
            other.n_layers, other.n_features, other.top_k, other.domain_top_k
        ), "merge_from: shard collectors have mismatched shape/top-k settings"
        assert self.n_random == 0 and other.n_random == 0, "sharded merge requires --n_random 0"
        assert not self.activation_example_ranges and not other.activation_example_ranges, (
            "sharded merge requires activation ranges disabled"
        )

        self.total_tokens += other.total_tokens
        for d, c in other.tokens_per_domain.items():
            self.tokens_per_domain[d] += c
        for r, c in other.tokens_per_region.items():
            self.tokens_per_region[r] += c

        self.run_hist_total += other.run_hist_total
        self.run_hist_by_domain += other.run_hist_by_domain
        for layer in range(self.n_layers):
            self.feature_hist_total_by_layer[layer] += other.feature_hist_total_by_layer[layer]
            self.feature_hist_by_domain_by_layer[layer] += other.feature_hist_by_domain_by_layer[layer]

        def _merge_heap(a: list, b: list, k: int) -> list:
            merged = a + b
            if len(merged) > k:
                merged = heapq.nlargest(k, merged)  # ActivatingExample.__lt__ ranks by activation
            heapq.heapify(merged)  # restore min-heap invariant
            return merged

        for layer in range(self.n_layers):
            s_layer, o_layer = self.stats[layer], other.stats[layer]
            for f in range(self.n_features):
                o = o_layer[f]
                if o.activation_count == 0:
                    continue
                s = s_layer[f]
                s.activation_count += o.activation_count
                s.top_k_examples = _merge_heap(s.top_k_examples, o.top_k_examples, self.top_k)
                for dom, oheap in o.domain_top_k_examples.items():
                    s.domain_top_k_examples[dom] = _merge_heap(
                        s.domain_top_k_examples[dom], oheap, self.domain_top_k
                    )
                for dom, c in o.domain_counts.items():
                    s.domain_counts[dom] += c
                for reg, c in o.region_counts.items():
                    s.region_counts[reg] += c
                for i in range(len(s.thinking_position_counts)):
                    s.thinking_position_counts[i] += o.thinking_position_counts[i]
                if o.token_counts:
                    for tid, c in o.token_counts.items():
                        s.token_counts[tid] = s.token_counts.get(tid, 0) + c
                    if self.token_count_cap and len(s.token_counts) > self.token_count_cap:
                        s.token_counts = dict(
                            heapq.nlargest(self.token_count_cap, s.token_counts.items(), key=lambda kv: kv[1])
                        )

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
        source_metadata: dict[str, Any],
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

        # Reservoir sampling decision for random (skip entirely when disabled: random.randint
        # per nonzero dominates dense collections, and n_random=0 wants no random samples).
        add_to_random = False
        random_replace_idx = None
        if self.n_random > 0:
            stats.random_seen_count += 1
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
            source_metadata=source_metadata,
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

    def _update_activation_histograms(
        self,
        *,
        layer_idx: int,
        domain: str,
        active_features: np.ndarray,
        active_values: np.ndarray,
    ) -> None:
        domain_idx = self.domain_to_index.get(domain)
        if domain_idx is None:
            raise ValueError(
                f"Unknown activation domain {domain!r}; known domains={self.domain_names}"
            )

        bin_indices = histogram_bin_indices(active_values, self.hist_bin_lower_bounds)
        np.add.at(self.run_hist_total, bin_indices, 1)
        np.add.at(self.run_hist_by_domain[domain_idx], bin_indices, 1)
        np.add.at(
            self.feature_hist_total_by_layer[layer_idx],
            (active_features, bin_indices),
            1,
        )
        np.add.at(
            self.feature_hist_by_domain_by_layer[layer_idx][domain_idx],
            (active_features, bin_indices),
            1,
        )

    def _maybe_add_activation_range_example(
        self,
        stats: FeatureStats,
        activation: float,
        tokens: list[int],
        position: int,
        features_gpu: torch.Tensor,
        feature_idx: int,
        domain: str,
        region: str,
        thinking_position: float | None,
        sequence_idx: int,
        source_metadata: dict[str, Any],
    ) -> None:
        if self.activation_range_examples_per_domain <= 0:
            return

        matched_range = None
        for activation_range in self.activation_example_ranges:
            lo, hi = activation_range
            if lo <= activation < hi:
                matched_range = activation_range
                break
        if matched_range is None:
            return

        label = format_activation_range_label(matched_range)
        key = f"{domain}|{label}"
        stats.activation_range_seen_counts[key] += 1
        kept_examples = stats.activation_range_examples[key]
        add_example = False
        replace_idx = None
        if len(kept_examples) < self.activation_range_examples_per_domain:
            add_example = True
        else:
            j = random.randint(0, stats.activation_range_seen_counts[key] - 1)
            if j < self.activation_range_examples_per_domain:
                add_example = True
                replace_idx = j
        if not add_example:
            return

        context_tokens, pos_in_ctx = self._get_context(tokens, position)
        ctx_start = max(0, position - self.context_before)
        ctx_end = min(len(tokens), position + self.context_after + 1)
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
            source_metadata=source_metadata,
        )
        if replace_idx is None:
            kept_examples.append(example)
        else:
            kept_examples[replace_idx] = example

    def process_batch(
        self,
        model,
        batch_tokens: list[list[int]],
        batch_domains: list[str],
        batch_markers: list[dict],
        batch_seq_idxs: list[int],
        batch_source_metadata: list[dict[str, Any]],
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

        self.accumulate_batch(
            batch_tokens=batch_tokens,
            batch_domains=batch_domains,
            batch_markers=batch_markers,
            batch_seq_idxs=batch_seq_idxs,
            batch_source_metadata=batch_source_metadata,
            layer_activations=self._layer_activations,
        )

        self._layer_activations = {}

    def accumulate_batch(
        self,
        *,
        batch_tokens: list[list[int]],
        batch_domains: list[str],
        batch_markers: list[dict],
        batch_seq_idxs: list[int],
        batch_source_metadata: list[dict[str, Any]],
        layer_activations: dict[int, torch.Tensor],
    ) -> None:
        """Accumulate feature stats from precomputed per-layer activations.

        ``layer_activations`` maps ``layer_idx -> tensor[B, >=seq_len, n_features]``
        of post-activation feature values (ReLU/JumpReLU already applied). This is
        the model-agnostic half of ``process_batch``: any front-end that can produce
        per-token feature activations can feed them here and reuse all top-k,
        histogram, and reservoir-sampling logic. ``process_batch`` supplies them via
        transcoder-adapter forward hooks; the base/GemmaScope collector supplies them
        from circuit-tracer ``ReplacementModel.get_activations``.
        """
        seq_lens = [len(t) for t in batch_tokens]

        # Process each item in the batch
        for b in range(len(batch_tokens)):
            tokens = batch_tokens[b]
            seq_len = seq_lens[b]
            domain = batch_domains[b]
            markers = batch_markers[b]
            sequence_idx = batch_seq_idxs[b]
            source_metadata = batch_source_metadata[b]

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
                features_gpu = layer_activations[layer_idx][b, :seq_len]  # [seq_len, n_features]

                nonzero = torch.nonzero(features_gpu > 0)  # [N, 2]
                if len(nonzero) == 0:
                    continue

                active_positions = nonzero[:, 0].cpu().numpy()
                active_features = nonzero[:, 1].cpu().numpy()
                active_values = features_gpu[nonzero[:, 0], nonzero[:, 1]].float().cpu().numpy()

                self._update_activation_histograms(
                    layer_idx=layer_idx,
                    domain=domain,
                    active_features=active_features,
                    active_values=active_values,
                )

                layer_stats = self.stats[layer_idx]

                # Pre-filter top-k example candidates. When n_random == 0 a nonzero can only be
                # kept if it beats its feature's current top-k / domain-top-k heap minimum (which
                # only rises within a layer), so we vectorize that check once and skip
                # _maybe_add_example for the dominated majority — the dominant cost in dense
                # (GemmaScope-style) collections. With n_random > 0 (e.g. the adapter collector)
                # every nonzero is a reservoir candidate, so candidate_mask stays None and behavior
                # is byte-for-byte unchanged.
                candidate_mask = None
                if self.n_random <= 0:
                    thresholds = np.full(self.n_features, -np.inf, dtype=np.float64)
                    for uf in np.unique(active_features).tolist():
                        s = layer_stats[uf]
                        global_thr = (
                            s.top_k_examples[0].activation
                            if len(s.top_k_examples) >= self.top_k
                            else -np.inf
                        )
                        domain_heap = s.domain_top_k_examples.get(domain)
                        domain_thr = (
                            domain_heap[0].activation
                            if domain_heap is not None and len(domain_heap) >= self.domain_top_k
                            else -np.inf
                        )
                        thresholds[uf] = min(global_thr, domain_thr)
                    candidate_mask = active_values > thresholds[active_features]

                track_ranges = (
                    self.activation_range_examples_per_domain > 0
                    and bool(self.activation_example_ranges)
                )

                for i in range(len(active_positions)):
                    pos = int(active_positions[i])
                    feature_idx = int(active_features[i])
                    act = float(active_values[i])

                    stats = layer_stats[feature_idx]
                    region = regions[pos]
                    think_pos = thinking_positions[pos]

                    stats.activation_count += 1
                    stats.domain_counts[domain] += 1
                    stats.region_counts[region] += 1
                    if self.track_token_counts and self.token_count_cap > 0:
                        token_id = tokens[pos]
                        token_counts = stats.token_counts
                        if token_id in token_counts:
                            token_counts[token_id] += 1
                        elif len(token_counts) < self.token_count_cap:
                            token_counts[token_id] = 1
                        else:
                            # Space-Saving: evict the lowest-count token, inherit its count + 1.
                            min_token = min(token_counts, key=token_counts.get)
                            token_counts[token_id] = token_counts.pop(min_token) + 1
                    if think_pos is not None:
                        bin_idx = min(9, int(think_pos * 10))
                        stats.thinking_position_counts[bin_idx] += 1

                    if track_ranges:
                        self._maybe_add_activation_range_example(
                            stats, act, tokens, pos, features_gpu, feature_idx,
                            domain, region, think_pos, sequence_idx, source_metadata,
                        )
                    if candidate_mask is None or candidate_mask[i]:
                        self._maybe_add_example(
                            stats, act, tokens, pos, features_gpu, feature_idx,
                            domain, region, think_pos, sequence_idx, source_metadata,
                        )


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


# Per-token-id decode cache. Export decodes the same handful of token ids (\n, common words,
# punctuation) tens of millions of times across hundreds of thousands of dense base features;
# tokenizer.decode is ~10-50us/call, so caching cuts packing time by ~10-50x. Output is identical
# (decode is deterministic). Keyed by token id; reset_decode_cache() is called at export start so
# a process that runs multiple collections never mixes tokenizers.
_DECODE_CACHE: dict[int, str] = {}


def reset_decode_cache() -> None:
    _DECODE_CACHE.clear()


def decode_token(tokenizer, token_id: int) -> str:
    cached = _DECODE_CACHE.get(token_id)
    if cached is None:
        cached = tokenizer.decode([token_id])
        _DECODE_CACHE[token_id] = cached
    return cached


def format_example_for_circuit_tracer(ex: ActivatingExample, tokenizer) -> dict:
    """Format an ActivatingExample for circuit tracer JSON."""
    tokens = [decode_token(tokenizer, tok_id) for tok_id in ex.context_tokens]
    formatted: dict[str, Any] = {
        "tokens": tokens,
        "tokens_acts_list": ex.context_activations,
        "train_token_ind": ex.position_in_context,
        "peak_activation": ex.activation,
        "is_repeated_datapoint": False,
    }
    if ex.source_metadata:
        formatted["source_metadata"] = dict(ex.source_metadata)
    return formatted


def format_top_tokens(stats: FeatureStats, tokenizer, *, k: int = 10) -> list[dict]:
    """Top tokens this feature activates on, with token-conditional fractions.

    Powers the graph-viz "of this feature's activations, what % land on this token"
    (token-conditional specificity) stat. Counts are bounded (Space-Saving, capped per
    feature), so fractions for the most frequent tokens are accurate while rare tokens may
    be undercounted or absent.
    """
    if not stats.token_counts:
        return []
    total = max(1, stats.activation_count)
    ranked = sorted(stats.token_counts.items(), key=lambda kv: -kv[1])[:k]
    return [
        {
            "token": decode_token(tokenizer, token_id),
            "token_id": int(token_id),
            "count": int(count),
            "fraction": count / total,
        }
        for token_id, count in ranked
    ]


def _build_examples_quantiles(
    stats: FeatureStats,
    tokenizer,
) -> tuple[list[dict], float, float]:
    top_examples = sorted(stats.top_k_examples, key=lambda x: -x.activation)
    random_examples = stats.random_examples

    top_formatted = [
        format_example_for_circuit_tracer(ex, tokenizer)
        for ex in top_examples
    ]
    random_formatted = [
        format_example_for_circuit_tracer(ex, tokenizer)
        for ex in random_examples
    ]

    domain_quantiles = []
    for domain_name, domain_heap in sorted(stats.domain_top_k_examples.items()):
        domain_sorted = sorted(domain_heap, key=lambda x: -x.activation)
        domain_formatted = [
            format_example_for_circuit_tracer(ex, tokenizer)
            for ex in domain_sorted
        ]
        domain_quantiles.append({
            "quantile_name": f"Top activations ({domain_name})",
            "examples": domain_formatted,
        })

    range_quantiles = []
    range_examples_for_scale = []
    for range_key, range_examples in sorted(stats.activation_range_examples.items()):
        domain_name, range_label = range_key.split("|", 1)
        range_sorted = sorted(range_examples, key=lambda x: x.activation)
        range_examples_for_scale.extend(range_sorted)
        range_formatted = [
            format_example_for_circuit_tracer(ex, tokenizer)
            for ex in range_sorted
        ]
        range_quantiles.append({
            "quantile_name": f"Activation range {range_label} ({domain_name})",
            "examples": range_formatted,
        })

    all_acts = []
    for ex in top_examples + random_examples + range_examples_for_scale:
        all_acts.extend(ex.context_activations)
    act_min = min(all_acts) if all_acts else 0.0
    act_max = max(all_acts) if all_acts else 1.0

    return (
        [
            {"quantile_name": "Top activations", "examples": top_formatted},
            *domain_quantiles,
            *range_quantiles,
            {"quantile_name": "Random samples", "examples": random_formatted},
        ],
        act_min,
        act_max,
    )


def _write_feature_json(args: tuple) -> None:
    """Write a single feature JSON file (for parallel execution)."""
    filepath, feature_json = args
    with open(filepath, 'w') as f:
        json.dump(feature_json, f)


def _pack_feature_for_circuit_tracer(feature_json: dict) -> bytes:
    """Pack one feature payload using circuit-tracer's local feature format."""
    json_bytes = json.dumps(feature_json, separators=(",", ":")).encode("utf-8")
    compressed = gzip.compress(json_bytes)
    return struct.pack("<I", len(compressed)) + compressed


def _drain_completed_writes(
    pending: set[Future],
    progress,
    block: bool = False,
) -> set[Future]:
    """Wait for completed JSON writes and surface any write errors."""
    while pending:
        done, pending = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            future.result()
        progress.update(len(done))
        if not block:
            break
    return pending


def _release_feature_examples(stats: FeatureStats) -> None:
    """Drop example buffers once the feature JSON has been handed to a writer."""
    stats.top_k_examples.clear()
    stats.random_examples.clear()
    stats.domain_top_k_examples.clear()
    stats.activation_range_examples.clear()
    stats.activation_range_seen_counts.clear()
    stats.token_counts.clear()


def _count_nonempty_features(collector: FeatureCollector) -> int:
    """Count feature JSON files that will be written."""
    return sum(
        1
        for layer_stats in collector.stats
        for stats in layer_stats
        if stats.activation_count > 0
    )


def _format_bytes(n_bytes: float) -> str:
    """Format a byte count for logs."""
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(max(0.0, n_bytes))
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"


def _estimate_token_json_bytes(
    items: list[tuple[list[int], str, dict, dict[str, Any]]],
    tokenizer,
    max_sample_tokens: int = 2048,
) -> float:
    """Estimate bytes for one decoded token string in feature JSON."""
    sampled_token_ids: list[int] = []
    for tokens, _, _, _ in items:
        remaining = max_sample_tokens - len(sampled_token_ids)
        if remaining <= 0:
            break
        sampled_token_ids.extend(tokens[:remaining])

    if not sampled_token_ids:
        return 8.0

    encoded_size = sum(
        len(json.dumps(tokenizer.decode([tok_id])))
        for tok_id in sampled_token_ids
    )
    return encoded_size / len(sampled_token_ids)


def log_startup_output_size_estimate(
    *,
    n_layers: int,
    n_features: int,
    items: list[tuple[list[int], str, dict, dict[str, Any]]],
    tokenizer,
    top_k: int,
    n_random: int,
    domain_top_k: int,
    context_before: int,
    context_after: int,
    domain_names: list[str],
    activation_example_ranges: list[tuple[float, float]],
    activation_range_examples_per_domain: int,
) -> None:
    """Log a conservative feature JSON size estimate before collection starts.

    The exact output size depends on which features fire and what examples are
    retained, both of which are only known after the model forward passes. This
    estimate is still useful as an early quota check because retention settings
    tightly bound the number and length of examples per feature.
    """
    total_features = n_layers * n_features
    max_context_tokens = context_before + 1 + context_after
    max_retained_examples_per_feature = (
        top_k
        + n_random
        + len(domain_names) * domain_top_k
        + len(domain_names)
        * len(activation_example_ranges)
        * max(0, activation_range_examples_per_domain)
    )

    token_json_bytes = _estimate_token_json_bytes(items, tokenizer)
    # Each saved token carries a decoded JSON string plus a float activation in
    # tokens_acts_list. Keep a little overhead for commas and object keys.
    bytes_per_saved_token = token_json_bytes + 18.0
    fixed_feature_bytes = 900.0
    bytes_per_example = 220.0 + max_context_tokens * bytes_per_saved_token
    bytes_per_feature = fixed_feature_bytes + (
        max_retained_examples_per_feature * bytes_per_example
    )
    feature_json_upper_bound = total_features * bytes_per_feature

    logger.info("Startup output-size estimate:")
    logger.info(
        "  Feature JSON upper bound: %s if all %s features fire",
        _format_bytes(feature_json_upper_bound),
        f"{total_features:,}",
    )
    logger.info(
        "  Estimate inputs: <=%s retained examples/feature, <=%s context tokens/example, "
        "%s domains, %s activation ranges, sampled token JSON %.1f bytes/token",
        f"{max_retained_examples_per_feature:,}",
        f"{max_context_tokens:,}",
        len(domain_names),
        len(activation_example_ranges),
        token_json_bytes,
    )
    logger.info(
        "  Exact feature count and final size are only known after activation collection; "
        "reduce --context_before/--context_after, --top_k, --domain_top_k, --n_random, "
        "or --activation_range_examples_per_domain to reduce output size."
    )


def export_circuit_tracer_json(
    collector: FeatureCollector,
    logit_lens_data: list[dict],
    tokenizer,
    output_dir: Path,
    n_workers: int = 16,
    export_circuit_tracer_features: bool = True,
    write_per_feature_json: bool = True,
):
    """Export feature data to circuit tracer JSON format.

    ``write_per_feature_json`` controls the per-feature ``features/{cantor_id}.json``
    files used by the dashboard. Set it False to write only the packed
    ``circuit_tracer_features/`` cache (which the attribution overlay/circuit-tracer
    frontend use). This matters for dense base/GemmaScope collections where nearly all
    of the hundreds of thousands of features fire, so writing one JSON per feature
    dominates wall-clock and inode count.
    """
    reset_decode_cache()
    features_dir = output_dir / "features"
    if write_per_feature_json:
        features_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Exporting features to {features_dir}...")
    else:
        logger.info("Skipping per-feature JSON files (packed circuit-tracer cache only).")
    packed_features_dir = output_dir / "circuit_tracer_features"
    packed_index: dict[str, Any] | None = None
    if export_circuit_tracer_features:
        packed_features_dir.mkdir(parents=True, exist_ok=True)
        packed_index = {"version": "1.0", "format": "variable_chunks"}
        logger.info(f"Exporting packed circuit-tracer features to {packed_features_dir}...")

    max_pending_writes = max(1, n_workers * 2)
    logger.info(
        f"Streaming feature JSON files with {n_workers} workers "
        f"(max {max_pending_writes} pending writes)..."
    )

    total_writes = _count_nonempty_features(collector)
    total_features = collector.n_layers * collector.n_features
    write_count = 0
    skipped = total_features - total_writes

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        pending_writes: set[Future] = set()
        with tqdm(
            total=total_writes,
            desc="Writing feature files",
            unit="file",
        ) as write_progress:
            for layer_idx in tqdm(range(collector.n_layers), desc="Building JSON"):
                layer_logit_lens = logit_lens_data[layer_idx]
                packed_bin_f = None
                packed_offsets = [0]
                packed_count = 0
                if export_circuit_tracer_features:
                    packed_bin_f = (packed_features_dir / f"layer_{layer_idx}.bin").open("wb")

                try:
                    for feature_idx in range(collector.n_features):
                        stats = collector.stats[layer_idx][feature_idx]

                        # Skip empty features
                        if stats.activation_count == 0:
                            if packed_bin_f is not None:
                                packed_offsets.append(packed_bin_f.tell())
                            continue

                        examples_quantiles, act_min, act_max = _build_examples_quantiles(
                            stats,
                            tokenizer,
                        )

                        # Get logit lens tokens
                        top_logits = [
                            decode_token(tokenizer, tok_id)
                            for tok_id in layer_logit_lens['top_ids'][feature_idx]
                        ]
                        bottom_logits = [
                            decode_token(tokenizer, tok_id)
                            for tok_id in layer_logit_lens['bot_ids'][feature_idx]
                        ]

                        # Build JSON
                        feature_json = {
                            "top_logits": top_logits,
                            "bottom_logits": bottom_logits,
                            "act_min": act_min,
                            "act_max": act_max,
                            "examples_quantiles": examples_quantiles,
                            # activation_frequency = "% of tokens this feature fires on" (metric 1).
                            "activation_frequency": stats.activation_count / max(1, collector.total_tokens),
                            # token_specificity = "% of this feature's activations on each token" (metric 2).
                            "token_specificity": format_top_tokens(stats, tokenizer),
                            "layer": layer_idx,
                            "feature": feature_idx,
                        }

                        if packed_bin_f is not None:
                            packed_bin_f.write(_pack_feature_for_circuit_tracer(feature_json))
                            packed_offsets.append(packed_bin_f.tell())
                            packed_count += 1

                        if write_per_feature_json:
                            cantor_id = cantor_pair(layer_idx, feature_idx)
                            filepath = features_dir / f"{cantor_id}.json"
                            pending_writes.add(executor.submit(_write_feature_json, (filepath, feature_json)))
                            write_count += 1
                        _release_feature_examples(stats)

                        if len(pending_writes) >= max_pending_writes:
                            pending_writes = _drain_completed_writes(
                                pending_writes, write_progress
                            )
                finally:
                    if packed_bin_f is not None:
                        packed_bin_f.close()

                if packed_index is not None:
                    packed_index[str(layer_idx)] = {
                        "filename": f"layer_{layer_idx}.bin",
                        "offsets": packed_offsets,
                    }
                    packed_size = (packed_features_dir / f"layer_{layer_idx}.bin").stat().st_size / 1e6
                    logger.info(
                        f"Packed layer {layer_idx}: {packed_size:.1f} MB, "
                        f"{packed_count} features, {collector.n_features - packed_count} missing"
                    )

            _drain_completed_writes(pending_writes, write_progress, block=True)

    if packed_index is not None:
        packed_index_path = packed_features_dir / "index.json.gz"
        with gzip.open(packed_index_path, "wt") as f:
            json.dump(packed_index, f)
        logger.info(f"Wrote packed circuit-tracer index to {packed_index_path}")

    logger.info(f"Generated {write_count} feature files, skipped {skipped} empty features")


def export_activation_histograms(collector: FeatureCollector, output_dir: Path) -> Path:
    hist_path = output_dir / "activation_histograms.npz"
    logger.info(f"Saving activation histograms to {hist_path}...")
    save_activation_histograms_npz(
        path=hist_path,
        bin_lower_bounds=collector.hist_bin_lower_bounds,
        domain_names=collector.domain_names,
        run_hist_total=collector.run_hist_total,
        run_hist_by_domain=collector.run_hist_by_domain,
        feature_hist_total_by_layer=collector.feature_hist_total_by_layer,
        feature_hist_by_domain_by_layer=collector.feature_hist_by_domain_by_layer,
    )
    logger.info(f"Saved activation histograms to {hist_path}")
    return hist_path


def build_feature_metadata_entry(
    *,
    collector: FeatureCollector,
    stats: FeatureStats,
    layer_idx: int,
    feature_idx: int,
    target_domain: str,
    baseline_domain: str,
) -> dict[str, Any]:
    total_acts = stats.activation_count

    domain_density = {}
    domain_fraction = {}
    for domain, count in stats.domain_counts.items():
        domain_tokens = collector.tokens_per_domain.get(domain, 0)
        domain_density[domain] = count / domain_tokens if domain_tokens > 0 else 0
        domain_fraction[domain] = count / total_acts if total_acts > 0 else 0

    region_density = {}
    region_fraction = {}
    for region, count in stats.region_counts.items():
        region_tokens = collector.tokens_per_region.get(region, 0)
        region_density[region] = count / region_tokens if region_tokens > 0 else 0
        region_fraction[region] = count / total_acts if total_acts > 0 else 0

    thinking_density = []
    thinking_fraction = []
    thinking_total = sum(stats.thinking_position_counts)
    for bin_idx in range(10):
        bin_acts = stats.thinking_position_counts[bin_idx]
        bin_tokens = collector.tokens_per_thinking_bin[bin_idx]
        thinking_density.append(bin_acts / bin_tokens if bin_tokens > 0 else 0)
        thinking_fraction.append(bin_acts / thinking_total if thinking_total > 0 else 0)

    feature_meta: dict[str, Any] = {
        "layer": layer_idx,
        "feature": feature_idx,
        "cantor_id": cantor_pair(layer_idx, feature_idx),
        "activation_count": total_acts,
        "activation_freq": total_acts / max(collector.total_tokens, 1),
        "domain_density": domain_density,
        "domain_fraction": domain_fraction,
        "region_density": region_density,
        "region_fraction": region_fraction,
        "thinking_position_density": thinking_density,
        "thinking_position_fraction": thinking_fraction,
    }
    relative_scores = compute_relative_domain_scores(
        domain_counts=dict(stats.domain_counts),
        tokens_per_domain=dict(collector.tokens_per_domain),
        domain_names=collector.domain_names,
        feature_hist_by_domain=collector.feature_hist_by_domain_by_layer[layer_idx][
            :, feature_idx, :
        ],
        bin_lower_bounds=collector.hist_bin_lower_bounds,
        target_domain=target_domain,
        baseline_domain=baseline_domain,
    )
    if relative_scores is not None:
        feature_meta["relative_domain_scores"] = relative_scores
    return feature_meta


def build_feature_frequency_summary(collector: FeatureCollector) -> dict[str, Any]:
    frequency_bin_lower_bounds = np.array(
        [
            0.0,
            1e-9,
            3e-9,
            1e-8,
            3e-8,
            1e-7,
            3e-7,
            1e-6,
            3e-6,
            1e-5,
            3e-5,
            1e-4,
            3e-4,
            1e-3,
            3e-3,
            1e-2,
            3e-2,
            1e-1,
            3e-1,
            1.0,
        ],
        dtype=np.float32,
    )
    n_total_features = collector.n_layers * collector.n_features
    total_counts = np.zeros(n_total_features, dtype=np.uint64)
    counts_by_domain = {
        domain: np.zeros(n_total_features, dtype=np.uint64)
        for domain in collector.domain_names
    }

    flat_idx = 0
    for layer_idx in range(collector.n_layers):
        for feature_idx in range(collector.n_features):
            stats = collector.stats[layer_idx][feature_idx]
            total_counts[flat_idx] = int(stats.activation_count)
            for domain in collector.domain_names:
                counts_by_domain[domain][flat_idx] = int(stats.domain_counts.get(domain) or 0)
            flat_idx += 1

    total_tokens = max(int(collector.total_tokens), 1)
    total_freqs = total_counts.astype(np.float64) / total_tokens
    total_bins = histogram_bin_indices(total_freqs, frequency_bin_lower_bounds)
    feature_frequency_hist_total = np.bincount(
        total_bins,
        minlength=len(frequency_bin_lower_bounds),
    ).astype(int).tolist()

    feature_frequency_hist_by_domain: dict[str, list[int]] = {}
    global_nonzero_density_by_domain: dict[str, float] = {}
    active_feature_count_by_domain: dict[str, int] = {}
    for domain, counts in counts_by_domain.items():
        domain_tokens = int(collector.tokens_per_domain.get(domain) or 0)
        if domain_tokens > 0:
            domain_freqs = counts.astype(np.float64) / domain_tokens
            domain_bins = histogram_bin_indices(domain_freqs, frequency_bin_lower_bounds)
            feature_frequency_hist_by_domain[domain] = np.bincount(
                domain_bins,
                minlength=len(frequency_bin_lower_bounds),
            ).astype(int).tolist()
            global_nonzero_density_by_domain[domain] = (
                float(counts.sum()) / (domain_tokens * max(n_total_features, 1))
            )
        else:
            feature_frequency_hist_by_domain[domain] = [0] * len(frequency_bin_lower_bounds)
            global_nonzero_density_by_domain[domain] = 0.0
        active_feature_count_by_domain[domain] = int(np.count_nonzero(counts))

    return {
        "frequency_bin_lower_bounds": frequency_bin_lower_bounds.tolist(),
        "domain_names": collector.domain_names,
        "all_features_total": n_total_features,
        "active_features_total": int(np.count_nonzero(total_counts)),
        "feature_frequency_hist_total": feature_frequency_hist_total,
        "feature_frequency_hist_by_domain": feature_frequency_hist_by_domain,
        "global_nonzero_density_by_domain": global_nonzero_density_by_domain,
        "active_feature_count_by_domain": active_feature_count_by_domain,
    }


def build_metadata_payload(
    *,
    collector: FeatureCollector,
    target_domain: str,
    baseline_domain: str,
    model_path: str,
    tokenizer_path: str | None,
    data_sources: list[FeatureDataSourceSettings],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "tokenization_settings": {
            "model_path": model_path,
            "tokenizer_path": tokenizer_path,
            "data_sources": [source.to_json() for source in data_sources],
        },
        "total_tokens": collector.total_tokens,
        "tokens_per_domain": dict(collector.tokens_per_domain),
        "tokens_per_region": dict(collector.tokens_per_region),
        "tokens_per_thinking_bin": collector.tokens_per_thinking_bin,
        "activation_histograms_file": "activation_histograms.npz",
        "activation_histogram_summary": {
            "bin_lower_bounds": collector.hist_bin_lower_bounds.tolist(),
            "domain_names": collector.domain_names,
            "run_hist_total": collector.run_hist_total.astype(int).tolist(),
            "run_hist_by_domain": collector.run_hist_by_domain.astype(int).tolist(),
        },
        "feature_frequency_summary": build_feature_frequency_summary(collector),
        "relative_score_config": {
            "target_domain": target_domain,
            "baseline_domain": baseline_domain,
            "smoothing": "max(raw_density, 1 / domain_token_count) for target and baseline",
            "strength_quantiles": [0.95, 0.99],
        },
        "features": [],
    }

    for layer_idx in range(collector.n_layers):
        for feature_idx in range(collector.n_features):
            stats = collector.stats[layer_idx][feature_idx]
            if stats.activation_count == 0:
                continue
            metadata["features"].append(
                build_feature_metadata_entry(
                    collector=collector,
                    stats=stats,
                    layer_idx=layer_idx,
                    feature_idx=feature_idx,
                    target_domain=target_domain,
                    baseline_domain=baseline_domain,
                )
            )
    return metadata


def export_metadata(
    collector: FeatureCollector,
    output_dir: Path,
    target_domain: str,
    baseline_domain: str,
    model_path: str,
    tokenizer_path: str | None,
    data_sources: list[FeatureDataSourceSettings],
) -> None:
    """Export rich metadata to JSON for analysis."""
    logger.info("Exporting metadata...")
    if target_domain not in collector.domain_names or baseline_domain not in collector.domain_names:
        logger.warning(
            "Relative scores skipped for missing domains: target=%r baseline=%r available=%s",
            target_domain,
            baseline_domain,
            collector.domain_names,
        )
    metadata = build_metadata_payload(
        collector=collector,
        target_domain=target_domain,
        baseline_domain=baseline_domain,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        data_sources=data_sources,
    )
    with open(output_dir / "feature_metadata.json", 'w') as f:
        json.dump(metadata, f)

    logger.info(f"Saved metadata for {len(metadata['features'])} features")


def annotate_collected_features(output_dir: Path) -> None:
    """Run fast metadata-based feature annotations for the dashboard."""
    annotations_path = output_dir / "feature_annotations.json"
    logger.info("Running assistant-response feature annotations...")
    logger.info(f"Annotation output file: {annotations_path}")
    run_annotation(
        data_dir=output_dir,
        annotations_file=annotations_path,
        annotator=AssistantResponseFeatureAnnotator(),
    )
    logger.info(f"Feature annotations saved to {annotations_path}")


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


def _source_row_metadata(dataset: Any, idx: int) -> dict[str, Any]:
    """Extract source IDs from a dataset row without retaining full transcripts."""
    row = None
    if hasattr(dataset, "ds"):
        row = dataset.ds[idx]
    else:
        try:
            row = dataset[idx]
        except Exception:
            row = None

    if not isinstance(row, dict):
        return {}

    metadata = {}
    for key in ("conversation_id", "id"):
        value = row.get(key)
        if value is not None:
            metadata[key] = value
    return metadata


def _build_example_source_metadata(
    *,
    dataset: Any,
    examples_meta: list[dict] | None,
    item: dict,
    dataset_row_idx: int,
    source_idx: int,
    source_path: str,
    domain_label: str | None,
    domain: str,
    prepared_item_idx: int,
) -> dict[str, Any]:
    """Build locator metadata for a retained activation example."""
    metadata: dict[str, Any] = {
        "source_idx": source_idx,
        "source_path": source_path,
        "dataset_row_idx": dataset_row_idx,
        "prepared_item_idx": prepared_item_idx,
        "domain": domain,
    }
    if domain_label is not None:
        metadata["domain_label"] = domain_label

    if examples_meta is not None:
        metadata.update(examples_meta[dataset_row_idx])

    for candidate in (item, _source_row_metadata(dataset, dataset_row_idx)):
        for key in ("conversation_id", "id"):
            value = candidate.get(key)
            if value is not None:
                metadata[key] = value

    return metadata


class ArgumentDefaultsRawTextHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawTextHelpFormatter,
):
    """Preserve intentional help formatting while still showing defaults."""


def _value_to_flag_string(action: argparse.Action, value: Any) -> str | None:
    """Format one argparse value as a pasteable CLI flag, or omit non-values."""
    if value is None:
        return None
    if isinstance(action, argparse._StoreTrueAction):
        return action.option_strings[0] if value else None
    if isinstance(action, argparse._StoreFalseAction):
        return action.option_strings[0] if not value else None
    if isinstance(action, argparse.BooleanOptionalAction):
        for option in action.option_strings:
            if value and not option.startswith("--no-"):
                return option
            if not value and option.startswith("--no-"):
                return option
        return None

    option = action.option_strings[0] if action.option_strings else None
    if option is None:
        return None

    if isinstance(value, (list, tuple)):
        if not value:
            return None
        quoted_values = " ".join(shlex.quote(str(item)) for item in value)
        return f"{option} {quoted_values}"

    return f"{option}={shlex.quote(str(value))}"


def build_replay_command(parser: argparse.ArgumentParser, args: argparse.Namespace) -> str:
    """Build a pasteable collector command from the fully parsed args namespace."""
    parts = ["uv run python -m analysis.features.collect_feature_activations"]
    for action in parser._actions:
        if action.dest in {"help", argparse.SUPPRESS}:
            continue
        if not action.option_strings:
            continue
        flag = _value_to_flag_string(action, getattr(args, action.dest, None))
        if flag is not None:
            parts.append(flag)
    return " \\\n    ".join(parts)


def export_run_arguments(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    """Write parsed run settings and a pasteable replay command."""
    args_payload = {
        key: value
        for key, value in vars(args).items()
    }
    args_json_path = output_dir / "collect_feature_activations_args.json"
    command_path = output_dir / "collect_feature_activations_command.sh"
    command = build_replay_command(parser, args)

    with open(args_json_path, "w") as f:
        json.dump(args_payload, f, indent=2, sort_keys=True)
        f.write("\n")
    with open(command_path, "w") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("set -euo pipefail\n\n")
        f.write(command)
        f.write("\n")

    logger.info(f"Saved parsed run arguments to {args_json_path}")
    logger.info(f"Saved pasteable replay command to {command_path}")


def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Collect transcoder feature activations for visualization",
        formatter_class=ArgumentDefaultsRawTextHelpFormatter,
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="""HF repo ID or local path to transcoder checkpoint""")
    parser.add_argument("--val_data", type=str, nargs='+', required=True,
                        help=textwrap.dedent("""
                            Validation data source(s). Each entry is either 'path' or 'domain:path'.
                            Examples:
                              chat:siddharthmb/lmsys-splits
                              fineweb:hf://org/dataset/data/val.jsonl
                        """).strip())
    parser.add_argument("--output_dir", type=str, default=None,
                        help="""Output directory (default: PRODUCTS_DIR/feature_data/<model>_<timestamp>)""")
    parser.add_argument(
        "--no-export_circuit_tracer_features",
        dest="export_circuit_tracer_features",
        action="store_false",
        default=True,
        help=textwrap.dedent("""
            Skip writing the packed circuit-tracer local feature cache while exporting feature JSON.
            By default, the packed cache is written.

            Output path:
              {output_dir}/circuit_tracer_features/index.json.gz
              {output_dir}/circuit_tracer_features/layer_N.bin

            This avoids the later CPU-only conversion pass that rereads features/*.json and repacks them.
        """).strip(),
    )
    parser.add_argument(
        "--upload_circuit_tracer_features_to_hub",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=textwrap.dedent("""
            Upload the packed circuit-tracer feature cache to Hugging Face Hub.
            Defaults to true when --export_circuit_tracer_features is enabled, and false
            when --no-export_circuit_tracer_features is used.

            The uploaded repo contains:
              features/index.json.gz
              features/layer_N.bin
              feature_collection_config.json
              feature_collection_summary.json

            Circuit-tracer can then use the Hugging Face repo ID as the feature scan name.
            Before model loading and feature collection, the script verifies write access and
            reserves the deterministic repo. If the same collection repo already exists, the
            script exits early and logs the existing Hub URL to avoid duplicate work.
        """).strip(),
    )
    parser.add_argument(
        "--hf_feature_repo_id",
        type=str,
        default=None,
        help="""Explicit Hugging Face model repo ID for uploaded circuit-tracer features. Default: deterministic repo name from model, data, and collection hyperparameters.""",
    )
    parser.add_argument(
        "--hub_org",
        type=str,
        default=None,
        help="""Hugging Face namespace/org for uploaded feature repos. Defaults to the authenticated user.""",
    )

    # Optional args
    parser.add_argument("--max_samples", type=int, default=None,
                        help="""Max samples to process per data source (default: all)""")
    parser.add_argument("--shuffle", action="store_true",
                        help=f"""Shuffle which rows are used (torch.Generator + torch.randperm). Uses seed {DEFAULT_SHUFFLE_SEED} unless --shuffle_seed is set.""")
    parser.add_argument("--shuffle_seed", type=int, default=None,
                        help=f"""If set, sample rows with this seed instead of sequential order (same mechanism as --shuffle). Overrides the default seed from --shuffle ({DEFAULT_SHUFFLE_SEED}). Each --val_data source uses an independent seed offset.""")
    parser.add_argument("--top_k", type=int, default=20,
                        help="""Number of global top activating examples per feature""")
    parser.add_argument("--domain_top_k", type=int, default=10,
                        help="""Number of top activating examples per feature per domain""")
    parser.add_argument("--n_random", type=int, default=10,
                        help="""Number of random samples per feature""")
    parser.add_argument(
        "--activation_example_ranges",
        type=str,
        default=DEFAULT_ACTIVATION_EXAMPLE_RANGES,
        help=textwrap.dedent("""
            Comma-separated activation bands in lo:hi format, for example:
              --activation_example_ranges 0.5:1.0,1.0:2.0,2.5:3.0

            For each positive feature activation, the collector checks whether the activation falls in one configured half-open range:
              lo <= activation < hi

            When it matches, the collector may save a small bounded text example for that feature, domain, and range. These examples appear in the dashboard as tabs like:
              Activation range 2.5-3.0 (chat)

            Why this exists:
              Histograms show how often activation magnitudes occur, but they do not show the text that caused those activations.
              Top-activation examples are useful, but they are biased toward extreme outliers and can hide the common medium-strength behavior of a feature.
              Random examples are useful, but they can miss a specific activation-strength band entirely.
              Activation ranges let you inspect representative text from chosen magnitude bands without saving every activation.

            Set to an empty string to disable matching all activation bands:
              --activation_example_ranges ''
        """).strip(),
    )
    parser.add_argument(
        "--activation_range_examples_per_domain",
        type=int,
        default=4,
        help=textwrap.dedent("""
            Maximum reservoir-sampled examples to keep for each feature/domain/range bucket.

            The cap is applied independently to every tuple:
              (layer, feature, domain, activation range)

            For example, with two domains, six activation ranges, and this value set to 4, one feature can keep up to:
              2 domains * 6 ranges * 4 examples = 48 activation-range examples

            Why this exists:
              The collector may see many positive activations for every feature across many tokens. Saving every matching text context would make feature JSONs enormous and slow down collection/export.
              Reservoir sampling keeps the output bounded while still giving each matching activation in the bucket a chance to be represented.
              Increase this when you want richer dashboard inspection for each band; decrease it when output size or collection time matters.

            Set to 0 to disable activation-range example tabs entirely.
        """).strip(),
    )
    parser.add_argument("--relative_target_domain", type=str, default="chat",
                        help="""Target domain for relative feature scores""")
    parser.add_argument("--relative_baseline_domain", type=str, default="fineweb",
                        help="""Baseline domain for relative feature scores""")
    parser.add_argument("--context_before", type=int, default=75,
                        help="""Number of tokens to save before each retained activation example. This only affects saved feature JSON snippet size, CPU transfer/serialization, and dashboard/auto-interp context; it does not change model forward tensor sizes or GPU activation tensor shapes.""")
    parser.add_argument("--context_after", type=int, default=20,
                        help="""Number of tokens to save after each retained activation example. This only affects saved feature JSON snippet size, CPU transfer/serialization, and dashboard/auto-interp context; it does not change model forward tensor sizes or GPU activation tensor shapes.""")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="""Max sequences per forward pass. GPU memory budget is auto-computed; this caps CPU-side work (per-token bookkeeping) per batch. Lower if the CPU bottleneck stalls the GPU on short sequences.""")
    parser.add_argument(
        "--shuffle_batches",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="""After length-sorting and packing into batches (padding-efficient), randomly reorder **which batch runs first** using torch.randperm with --shuffle_batches_seed. Total compute is identical; only iteration order changes, so per-step progress/time estimates (e.g. tqdm) are less skewed by many cheap short batches at the start. Default: on. Use --no-shuffle_batches to run batches in strict length order (shortest batches first).""",
    )
    parser.add_argument(
        "--shuffle_batches_seed",
        type=int,
        default=DEFAULT_BATCH_SHUFFLE_SEED,
        help=f"""Seed for --shuffle_batches (ignored with --no-shuffle_batches). Default: {DEFAULT_BATCH_SHUFFLE_SEED}.""",
    )
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="""Explicit tokenizer path (default: resolved from model_type)""")
    parser.add_argument("--max_length", type=int, default=4096,
                        help="""Max sequence length (longer sequences truncated)""")
    parser.add_argument(
        "--no_per_feature_json",
        action="store_true",
        help=(
            "Skip writing per-feature features/{cantor_id}.json files (and the dashboard "
            "annotation pass); write only the packed circuit_tracer_features/ cache used by "
            "the attribution overlay. Use this for large runs where per-feature JSON writing "
            "dominates wall-clock and fills scratch disk (hundreds of thousands of tiny files)."
        ),
    )

    add_collection_wandb_args(parser)

    args = parser.parse_args()
    if args.upload_circuit_tracer_features_to_hub is None:
        args.upload_circuit_tracer_features_to_hub = bool(args.export_circuit_tracer_features)
    if args.upload_circuit_tracer_features_to_hub and not args.export_circuit_tracer_features:
        parser.error(
            "--upload_circuit_tracer_features_to_hub requires --export_circuit_tracer_features "
            "(remove --no-export_circuit_tracer_features or disable upload with "
            "--no-upload_circuit_tracer_features_to_hub)."
        )

    try:
        activation_example_ranges = parse_activation_example_ranges(
            args.activation_example_ranges
        )
    except ValueError as exc:
        parser.error(str(exc))

    hf_feature_repo_id = None
    hf_feature_config = None
    if args.upload_circuit_tracer_features_to_hub:
        from analysis.features.hub_upload import (
            build_feature_collection_repo_id,
            check_feature_collection_exists,
            reserve_feature_collection_repo,
        )

        hf_feature_repo_id, hf_feature_config = build_feature_collection_repo_id(args)
        existing_repo_id = check_feature_collection_exists(hf_feature_repo_id, hf_feature_config)
        if existing_repo_id is not None:
            raise RuntimeError(
                "Feature collection already exists for these settings: "
                f"https://huggingface.co/{existing_repo_id}"
            )
        if reserve_feature_collection_repo(hf_feature_repo_id, hf_feature_config):
            raise RuntimeError(
                "Feature collection was reserved by another process before this run started: "
                f"https://huggingface.co/{hf_feature_repo_id}"
            )

    if args.shuffle_seed is not None:
        shuffle_seed: int | None = args.shuffle_seed
    elif args.shuffle:
        shuffle_seed = DEFAULT_SHUFFLE_SEED
    else:
        shuffle_seed = None

    if args.output_dir is None:
        model_slug = args.model_path.rstrip("/").split("/")[-1]
        output_dir = generate_output_path("feature_data", model_slug)
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    export_run_arguments(parser, args, output_dir)

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
    data_source_settings = [
        FeatureDataSourceSettings(
            source_path=path,
            max_length=args.max_length,
            model_type=model_type,
            domain=domain_label,
        )
        for domain_label, path in val_data_sources
    ]
    logger.info(f"Loading {len(val_data_sources)} data source(s):")
    loaded_sources: list[tuple[Any, list[dict] | None]] = []
    for source_settings in data_source_settings:
        logger.info(f"  {source_settings.source_path!r} (domain={source_settings.domain!r})")
        dataset, examples_meta = load_val_data_from_settings(source_settings, tokenizer)
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

    # Pre-extract all items across sources, filtering malformed ones
    items: list[tuple[list[int], str, dict, dict[str, Any]]] = []
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
            source_metadata = _build_example_source_metadata(
                dataset=dataset,
                examples_meta=examples_meta,
                item=item,
                dataset_row_idx=idx,
                source_idx=source_idx,
                source_path=val_data_sources[source_idx][1],
                domain_label=domain_label,
                domain=domain,
                prepared_item_idx=len(items),
            )

            if has_thinking and (markers['think_start'] is None or markers['think_end'] is None):
                if skipped < 5:
                    logger.warning(f"Skipping sample {idx} (source {source_idx}) - missing <think> tags")
                skipped += 1
                continue

            items.append((tokens, domain, markers, source_metadata))

    if skipped > 0:
        logger.warning(f"Skipped {skipped} samples due to missing <think> tags")

    if not items:
        logger.error(
            "No sequences left to process after filtering "
            f"(has_thinking={has_thinking}, skipped={skipped}). Exiting."
        )
        return

    domain_names = sorted({domain for _, domain, _, _ in items})
    logger.info(f"Activation histogram domains: {domain_names}")
    log_startup_output_size_estimate(
        n_layers=n_layers,
        n_features=n_features,
        items=items,
        tokenizer=tokenizer,
        top_k=args.top_k,
        n_random=args.n_random,
        domain_top_k=args.domain_top_k,
        context_before=args.context_before,
        context_after=args.context_after,
        domain_names=domain_names,
        activation_example_ranges=activation_example_ranges,
        activation_range_examples_per_domain=args.activation_range_examples_per_domain,
    )

    # Create collector now that all concrete domain labels are known.
    collector = FeatureCollector(
        n_layers=n_layers,
        n_features=n_features,
        top_k=args.top_k,
        n_random=args.n_random,
        domain_top_k=args.domain_top_k,
        context_before=args.context_before,
        context_after=args.context_after,
        domain_names=domain_names,
        activation_example_ranges=activation_example_ranges,
        activation_range_examples_per_domain=args.activation_range_examples_per_domain,
    )

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    # Token budget from GPU memory. Collection uses the transformer backbone only (no lm_head
    # logits, see _run_backbone_for_hooks), so the dominant term is the per-layer hook tensors.
    max_batch_tokens = compute_max_batch_tokens(
        device=model.device,
        n_layers=n_layers,
        n_features=n_features,
        hidden=int(getattr(model.config, "hidden_size", 0) or 0),
    )
    batches = form_length_packed_batches(
        items,
        batch_size=args.batch_size,
        max_batch_tokens=max_batch_tokens,
        shuffle=args.shuffle_batches,
        shuffle_seed=args.shuffle_batches_seed,
    )

    wandb_logger = CollectionWandbLogger.from_args(
        args,
        run_name=output_dir.name,
        config={
            **{k: v for k, v in vars(args).items() if not k.startswith("_")},
            "collector": "adapter",
            "n_layers": n_layers,
            "n_features": n_features,
            "n_sequences": len(items),
            "n_batches": len(batches),
        },
        total_sequences=len(items),
    )
    wandb_logger.reset_timer()

    # Process batches
    collector.register_hooks(model)
    for batch in tqdm(batches, desc="Processing batches"):
        batch_tokens = [t for t, _, _, _ in batch]
        batch_domains = [d for _, d, _, _ in batch]
        batch_markers = [m for _, _, m, _ in batch]
        batch_source_metadata = [source_meta for _, _, _, source_meta in batch]
        batch_seq_idxs = [
            int(source_meta["prepared_item_idx"])
            for source_meta in batch_source_metadata
        ]

        collector.process_batch(model, batch_tokens, batch_domains, batch_markers,
                                batch_seq_idxs, batch_source_metadata, pad_token_id)
        wandb_logger.log_batch(n_seqs=len(batch_tokens), n_tokens=sum(len(t) for t in batch_tokens))

    collector.remove_hooks()

    # Summary stats
    logger.info("Collection summary:")
    logger.info(f"  Total tokens: {collector.total_tokens:,}")
    logger.info(f"  Domains: {dict(collector.tokens_per_domain)}")
    logger.info(f"  Regions: {dict(collector.tokens_per_region)}")
    wandb_logger.finish(
        summary={
            "final/total_tokens": collector.total_tokens,
            "final/n_sequences": len(items),
            "final/tokens_per_domain": dict(collector.tokens_per_domain),
        }
    )

    # Compute logit lens
    logit_lens_data = compute_logit_lens(model, tokenizer)

    # Export
    export_circuit_tracer_json(
        collector,
        logit_lens_data,
        tokenizer,
        output_dir,
        export_circuit_tracer_features=args.export_circuit_tracer_features,
        write_per_feature_json=not args.no_per_feature_json,
    )
    export_activation_histograms(collector, output_dir)
    export_metadata(
        collector,
        output_dir,
        target_domain=args.relative_target_domain,
        baseline_domain=args.relative_baseline_domain,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer,
        data_sources=data_source_settings,
    )
    # Annotations are derived from feature_metadata.json (always written), not the
    # per-feature features/{id}.json files, so they run even with --no_per_feature_json.
    # They are also a required sidecar for the Hub upload below.
    annotate_collected_features(output_dir)

    if hf_feature_repo_id and hf_feature_config:
        from analysis.features.hub_upload import upload_circuit_tracer_features_to_hub

        upload_circuit_tracer_features_to_hub(
            repo_id=hf_feature_repo_id,
            output_dir=output_dir,
            config=hf_feature_config,
        )

    logger.info(f"Done! Output written to {output_dir}")
    logger.info("  features/: Circuit tracer JSON files")
    if args.export_circuit_tracer_features:
        logger.info("  circuit_tracer_features/: Packed circuit-tracer feature cache")
    logger.info("  activation_histograms.npz: Exact activation histogram sidecar")
    logger.info("  feature_metadata.json: Rich metadata for analysis")
    logger.info("  feature_annotations.json: Automatic feature annotations")
    logger.info("  collect_feature_activations_args.json: Full parsed CLI settings")
    logger.info("  collect_feature_activations_command.sh: Pasteable replay command")
    if hf_feature_repo_id:
        logger.info(f"  Hugging Face feature repo: https://huggingface.co/{hf_feature_repo_id}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.error("Unhandled exception in collect_feature_activations", exc_info=True)
        raise
