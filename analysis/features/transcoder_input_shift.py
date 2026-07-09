"""Measure how GemmaScope transcoder sparsity/reconstruction shift with the input model.

<!-- Created by Claude Code session "implement: new 07/02 gemmascope transcoder experiments". 2026-07-09. -->

Experiment 1 (see plan cached-doodling-horizon.md). Pretrained GemmaScope transcoders
(``google/gemma-scope-2b-pt-transcoders``) were trained to replace each *base-model* MLP
inside the *full base model* -- i.e. on the base model's per-layer input distribution. In
our circuit-tracing use case we instead feed them the *instruction-tuned* (or hybrid-adapter)
hidden states. Switching the source model shifts the distribution of inputs to each layer,
which can change which features fire (L0 sparsity) and how well the transcoder reconstructs
the MLP output (reconstruction error).

For each layer L and each source model, this script measures, over many tokens:
  * L0     -- mean number of active transcoder features per token (post-JumpReLU).
  * FVU    -- fraction of variance unexplained of the transcoder reconstruction, plus raw
              MSE and mean cosine similarity.
  * per-feature fire frequency -- for the "does the same feature fire" / re-finetune decision.

The transcoder's target is ALWAYS the base MLP: ``target = MLP_base(x)``. We only vary the
source of ``x`` (the post-``ln2`` normalized hidden state feeding the transcoder). Because the
gemma-2 MLP sub-block (``ln2.hook_normalized`` -> ``hook_mlp_out``) is strictly position-wise,
we obtain ``MLP_base(x)`` for foreign ``x`` with a *patched* base-model forward: overwrite
``blocks.L.ln2.hook_normalized`` with the source model's ``x`` and read ``blocks.L.hook_mlp_out``.
This is exactly the input->output map the transcoder was trained to imitate, with no
assumptions about pre/post layernorm folding.

Models are loaded as plain TransformerLens ``HookedTransformer``s with the SAME processing that
circuit-tracer's ``ReplacementModel.from_pretrained_and_transcoders`` uses
(``fold_ln=False, center_writing_weights=False, center_unembed=False``); this guarantees
``ln2.hook_normalized`` matches the convention the GemmaScope transcoders expect.

Sanity check: the ``base`` source at ``average_l0_76`` should reproduce L0 ~= 76 and a small
FVU, validating the pipeline before the instruct numbers are trusted.

Usage (smoke, single layer, no upload/wandb):
    uv run --extra viz python -m analysis.features.transcoder_input_shift \
        --gemmascope_width width_16k --gemmascope_l0 average_l0_76 \
        --sources base instruct --layers 0 --max_tokens 2000 --no-wandb

Full first read:
    uv run --extra viz python -m analysis.features.transcoder_input_shift \
        --gemmascope_width width_16k --gemmascope_l0 average_l0_76 \
        --sources base instruct --layers 0 6 12 18 25 --max_tokens 1000000 --wandb
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from helpers.log import logger, setup_logging
from helpers.paths.output_path import generate_output_path

from analysis.attribution.run_base_adapter_comparison import (
    _load_gemmascope_transcoders,
    build_gemmascope_transcoder_config,
    resolve_gemmascope_l0_values,
)
from analysis.features.batching import compute_max_batch_tokens, form_length_packed_batches
from analysis.features.collect_feature_activations import _parse_val_data_entry
from analysis.features.collection_wandb import CollectionWandbLogger
from analysis.features.load_val_data import load_val_data


def _torch_dtype(name: str) -> torch.dtype:
    normalized = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}.get(name, name)
    return getattr(torch, normalized)


def parse_layers(values: list[str], n_layers: int) -> list[int]:
    """Parse ``--layers`` (list of ints, or the single token ``all``)."""
    if len(values) == 1 and values[0].lower() == "all":
        return list(range(n_layers))
    layers = sorted({int(v) for v in values})
    for layer in layers:
        if not 0 <= layer < n_layers:
            raise ValueError(f"Layer {layer} out of range [0, {n_layers})")
    return layers


class ShiftAccumulator:
    """Streaming per-(source, layer) sums so the run scales to ~1M tokens without storing acts."""

    def __init__(self, d_model: int, n_features: int, device: torch.device) -> None:
        self.n_tokens = 0
        self.sum_l0 = 0.0
        self.sum_sqerr = 0.0
        self.sum_target_sq = 0.0
        self.sum_cos = 0.0
        self.sum_target = torch.zeros(d_model, dtype=torch.float64, device=device)
        self.fire_count = torch.zeros(n_features, dtype=torch.float64, device=device)

    def update(self, feats: torch.Tensor, recon: torch.Tensor, target: torch.Tensor) -> None:
        """feats/recon/target are flat [N, .] tensors over the valid (non-BOS, non-pad) tokens."""
        feats = feats.float()
        recon = recon.float()
        target = target.float()
        n = target.shape[0]
        self.n_tokens += n
        self.sum_l0 += (feats > 0).sum(dim=-1).sum().item()
        self.sum_sqerr += (recon - target).pow(2).sum().item()
        self.sum_target_sq += target.pow(2).sum().item()
        self.sum_target += target.double().sum(dim=0)
        self.sum_cos += torch.nn.functional.cosine_similarity(recon, target, dim=-1).sum().item()
        self.fire_count += (feats > 0).double().sum(dim=0)

    def metrics(self) -> dict[str, float]:
        n = max(1, self.n_tokens)
        # denom = sum_t ||target_t - mean_target||^2 = sum||t||^2 - n * ||mean||^2
        mean_sq = (self.sum_target / n).pow(2).sum().item()
        denom = self.sum_target_sq - n * mean_sq
        fvu = self.sum_sqerr / denom if denom > 0 else float("nan")
        return {
            "n_tokens": self.n_tokens,
            "l0_mean": self.sum_l0 / n,
            "fvu": fvu,
            "mse_mean": self.sum_sqerr / (n * self.sum_target.numel()),
            "cosine_mean": self.sum_cos / n,
        }


def load_hooked_model(model_name: str, *, device: torch.device, dtype: torch.dtype):
    """Load a plain HookedTransformer with the GemmaScope-matching processing convention."""
    from transformer_lens import HookedTransformer

    logger.info(f"Loading HookedTransformer: {model_name}")
    model = HookedTransformer.from_pretrained(
        model_name,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        dtype=dtype,
        device=device,
    )
    model.eval()
    return model


def resolve_source_model(source: str, base_model, instruct_model):
    """Map a ``--sources`` token to a loaded model. base/instruct supported; adapter deferred."""
    if source == "base":
        return base_model
    if source == "instruct":
        return instruct_model
    raise NotImplementedError(
        f"Source {source!r} is not implemented. Use 'base' or 'instruct'. "
        "adapter:<ckpt> (hybrid-model hidden states) is a planned Experiment-1 extension."
    )


def _input_hook_names(layers: list[int], feature_input_hook: str) -> list[str]:
    return [f"blocks.{layer}.{feature_input_hook}" for layer in layers]


def _output_hook_names(layers: list[int], feature_output_hook: str) -> list[str]:
    return [f"blocks.{layer}.{feature_output_hook}" for layer in layers]


@torch.no_grad()
def compute_source_x(
    model, tokens: torch.Tensor, layers: list[int], feature_input_hook: str, feature_output_hook: str, *, capture_mlp_out: bool,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor] | None]:
    """Run ``model`` and cache per-layer transcoder input x (and optionally true mlp_out)."""
    names = _input_hook_names(layers, feature_input_hook)
    if capture_mlp_out:
        names = names + _output_hook_names(layers, feature_output_hook)
    names_set = set(names)
    _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in names_set)
    x_by_layer = {layer: cache[f"blocks.{layer}.{feature_input_hook}"] for layer in layers}
    mlp_out = None
    if capture_mlp_out:
        mlp_out = {layer: cache[f"blocks.{layer}.{feature_output_hook}"] for layer in layers}
    return x_by_layer, mlp_out


@torch.no_grad()
def compute_base_target(
    base_model, tokens: torch.Tensor, x_by_layer: dict[int, torch.Tensor], layers: list[int], feature_input_hook: str, feature_output_hook: str,
) -> dict[int, torch.Tensor]:
    """Patched base forward: overwrite ln2.hook_normalized with x and read hook_mlp_out = MLP_base(x)."""
    captured: dict[int, torch.Tensor] = {}
    fwd_hooks = []
    for layer in layers:
        def set_input(acts, hook, _layer=layer):
            return x_by_layer[_layer].to(acts.dtype)

        def grab_output(acts, hook, _layer=layer):
            captured[_layer] = acts
            return acts

        fwd_hooks.append((f"blocks.{layer}.{feature_input_hook}", set_input))
        fwd_hooks.append((f"blocks.{layer}.{feature_output_hook}", grab_output))
    base_model.run_with_hooks(tokens, fwd_hooks=fwd_hooks, return_type=None)
    return captured


def build_valid_mask(tokens: torch.Tensor, pad_token_id: int | None) -> torch.Tensor:
    """[batch, seq] bool mask: real tokens excluding position 0 (BOS artifact) and padding."""
    mask = torch.ones_like(tokens, dtype=torch.bool)
    mask[:, 0] = False  # BOS / position-0 high-norm artifact (transcoders zero it too)
    if pad_token_id is not None:
        mask &= tokens != pad_token_id
    return mask


def pad_batch(item_tokens: list[list[int]], pad_token_id: int, device: torch.device) -> torch.Tensor:
    """Right-pad a batch of token lists to a [batch, max_len] tensor (causal => pad is safe)."""
    max_len = max(len(t) for t in item_tokens)
    out = torch.full((len(item_tokens), max_len), pad_token_id, dtype=torch.long, device=device)
    for i, toks in enumerate(item_tokens):
        out[i, : len(toks)] = torch.tensor(toks, dtype=torch.long, device=device)
    return out


def prepare_items(args: argparse.Namespace, tokenizer) -> list[tuple[list[int], str]]:
    """Load + tokenize all --val_data sources into (tokens, domain) items (chat via it template)."""
    items: list[tuple[list[int], str]] = []
    for entry in args.val_data:
        domain, path = _parse_val_data_entry(entry)
        logger.info(f"Loading source {path!r} (domain={domain!r})")
        dataset, _ = load_val_data(path, tokenizer, args.max_length, domain=domain, model_type="gemma2")
        for i in range(len(dataset)):
            toks = dataset[i]["input_ids"]
            if hasattr(toks, "tolist"):
                toks = toks.tolist()
            toks = list(toks)[: args.max_length]
            if len(toks) >= 2:
                items.append((toks, domain or "unknown"))
    logger.info(f"Prepared {len(items)} sequences across {len(args.val_data)} source(s)")
    return items


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base_model", default="google/gemma-2-2b", help="Base model. Provides the MLP_base target and the 'base' source distribution.")
    parser.add_argument("--instruct_model", default="google/gemma-2-2b-it", help="Instruction-tuned model providing the shifted 'instruct' source hidden states.")
    parser.add_argument("--sources", nargs="+", default=["base", "instruct"], help="Which input distributions to evaluate: 'base', 'instruct' (adapter:<ckpt> planned).")
    parser.add_argument("--layers", nargs="+", default=["0", "6", "12", "18", "25"], help="Layers to evaluate: space-separated ints, or the single token 'all' (0..n_layers-1).")

    parser.add_argument("--gemmascope_repo", default="google/gemma-scope-2b-pt-transcoders", help="HF repo of GemmaScope PT transcoders.")
    parser.add_argument("--gemmascope_width", required=True, help="GemmaScope width folder, e.g. width_16k.")
    parser.add_argument("--gemmascope_l0", required=True, help="GemmaScope L0 folder / target, e.g. average_l0_76.")
    parser.add_argument("--gemmascope_l0_match", default="nearest", choices=["exact", "nearest"], help="How to interpret a single --gemmascope_l0 value across layers. 'nearest' picks the closest available average_l0_* folder per layer (GemmaScope's available L0s differ by layer, so 'exact' 404s on layers lacking that exact folder).")
    parser.add_argument("--gemmascope_n_layers", type=int, default=26, help="Number of base-model layers / GemmaScope transcoders (gemma-2-2b = 26).")
    parser.add_argument("--feature_input_hook", default="ln2.hook_normalized", help="TransformerLens hook the transcoder reads (its input x).")
    parser.add_argument("--feature_output_hook", default="hook_mlp_out", help="TransformerLens hook the transcoder reconstructs (the MLP output).")

    parser.add_argument("--val_data", nargs="+", default=["chat:siddharthmb/2026.transcoder-adapters.lmsys-chat-1m-splits"], help="Data sources as domain:path (chat rendered with the it chat template). Add e.g. 'fineweb:science-of-finetuning/fineweb-1m-sample' for a web mix.")
    parser.add_argument("--prompt_tokenizer_model", default="google/gemma-2-2b-it", help="Tokenizer used to render chat data (base + it share vocab).")
    parser.add_argument("--max_tokens", type=int, default=200_000, help="Approximate number of (valid) tokens to evaluate per source. 2000 for a smoke test, ~1e6 for the real run.")
    parser.add_argument("--max_length", type=int, default=512, help="Max tokens per sequence.")
    parser.add_argument("--batch_size", type=int, default=16, help="Max sequences per batch (also capped by a GPU-memory token budget).")

    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument("--dtype", default="bf16", help="Torch dtype for the models (bf16/fp16/fp32).")
    parser.add_argument("--output_dir", type=Path, default=None, help="Output dir (default: PRODUCTS_DIR/transcoder_input_shift/<tag>_<timestamp>).")

    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False, help="Log throughput/summary to wandb (--wandb / --no-wandb).")
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_log_every", type=int, default=5)
    return parser


def main() -> None:
    setup_logging()
    args = build_parser().parse_args()

    device = torch.device(args.device)
    dtype = _torch_dtype(args.dtype)
    layers = parse_layers(args.layers, args.gemmascope_n_layers)

    tag = f"gemmascope_{args.gemmascope_width}_{args.gemmascope_l0}_{'-'.join(args.sources)}"
    output_dir = args.output_dir or generate_output_path("transcoder_input_shift", tag)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Invocation: %s", " ".join(shlex.quote(a) for a in [sys.executable, *sys.argv]))
    logger.info("Output dir: %s", output_dir)
    logger.info("Layers: %s | sources: %s", layers, args.sources)

    # --- Load transcoders (base GemmaScope) ---
    l0_values = resolve_gemmascope_l0_values(
        repo=args.gemmascope_repo, width=args.gemmascope_width, l0=args.gemmascope_l0,
        n_layers=args.gemmascope_n_layers, l0_match=args.gemmascope_l0_match,
    )
    ts_config = build_gemmascope_transcoder_config(
        repo=args.gemmascope_repo, width=args.gemmascope_width, l0=args.gemmascope_l0,
        n_layers=args.gemmascope_n_layers, model_name=args.base_model,
        feature_input_hook=args.feature_input_hook, feature_output_hook=args.feature_output_hook,
        l0_values=l0_values, l0_match=args.gemmascope_l0_match,
    )
    (output_dir / "gemmascope_config.json").write_text(json.dumps(ts_config, indent=2) + "\n")
    logger.info("Loading GemmaScope transcoders (%s)...", ts_config["scan_name"])
    transcoders = _load_gemmascope_transcoders(ts_config, device=device, dtype=dtype)

    # --- Load models ---
    base_model = load_hooked_model(args.base_model, device=device, dtype=dtype)
    instruct_model = None
    if "instruct" in args.sources:
        instruct_model = load_hooked_model(args.instruct_model, device=device, dtype=dtype)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.prompt_tokenizer_model)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    d_model = base_model.cfg.d_model
    n_features = transcoders[layers[0]].W_enc.shape[0]

    # --- Data ---
    items4 = [(toks, dom, {}, {}) for toks, dom in prepare_items(args, tokenizer)]
    max_batch_tokens = compute_max_batch_tokens(
        device=device, n_layers=len(layers), n_features=n_features, hidden=d_model,
    )
    batches = form_length_packed_batches(
        items4, batch_size=args.batch_size, max_batch_tokens=max_batch_tokens,
        shuffle=True, shuffle_seed=0,
    )

    accs = {
        source: {layer: ShiftAccumulator(d_model, n_features, device) for layer in layers}
        for source in args.sources
    }

    wandb_logger = CollectionWandbLogger(
        enabled=args.wandb, project=args.wandb_project, entity=args.wandb_entity,
        run_name=args.wandb_run_name or output_dir.name, config=vars(args) | {"tag": tag},
        log_every=args.wandb_log_every,
    )
    wandb_logger.reset_timer()

    # --- Main loop: stop each source once it reaches max_tokens ---
    done = {source: False for source in args.sources}
    t0 = time.time()
    for bi, batch in enumerate(batches):
        if all(done.values()):
            break
        batch_tokens = pad_batch([it[0] for it in batch], pad_id, device)
        valid_mask = build_valid_mask(batch_tokens, pad_id)
        flat_valid = valid_mask.reshape(-1)

        for source in args.sources:
            if done[source]:
                continue
            model = resolve_source_model(source, base_model, instruct_model)
            is_base = model is base_model
            x_by_layer, mlp_out = compute_source_x(
                model, batch_tokens, layers, args.feature_input_hook, args.feature_output_hook,
                capture_mlp_out=is_base,
            )
            if is_base:
                target_by_layer = mlp_out
            else:
                target_by_layer = compute_base_target(
                    base_model, batch_tokens, x_by_layer, layers,
                    args.feature_input_hook, args.feature_output_hook,
                )
            for layer in layers:
                x = x_by_layer[layer].reshape(-1, d_model)[flat_valid]
                target = target_by_layer[layer].reshape(-1, d_model)[flat_valid]
                feats = transcoders[layer].encode(x)
                recon = transcoders[layer].decode(feats, x)
                accs[source][layer].update(feats, recon, target)

        n_valid = int(flat_valid.sum().item())
        wandb_logger.log_batch(n_seqs=len(batch), n_tokens=n_valid)
        for source in args.sources:
            reached = min(accs[source][layer].n_tokens for layer in layers)
            if reached >= args.max_tokens:
                done[source] = True
        if bi % 10 == 0:
            some = accs[args.sources[0]][layers[0]]
            m = some.metrics()
            logger.info(
                "batch %d | %s L%d: n=%d L0=%.1f FVU=%.3f | %.0f tok/s",
                bi, args.sources[0], layers[0], some.n_tokens, m["l0_mean"], m["fvu"],
                some.n_tokens / max(1e-6, time.time() - t0),
            )

    # --- Results ---
    results = {"config": vars(args) | {"scan_name": ts_config["scan_name"], "layers": layers}, "by_source": {}}
    results["config"]["output_dir"] = str(output_dir)
    fire_freq: dict[str, Any] = {}
    for source in args.sources:
        results["by_source"][source] = {}
        for layer in layers:
            acc = accs[source][layer]
            m = acc.metrics()
            results["by_source"][source][str(layer)] = m
            n = max(1, acc.n_tokens)
            fire_freq[f"{source}__layer_{layer}"] = (acc.fire_count / n).cpu().numpy().astype(np.float32)
            logger.info("%s L%d: n=%d L0=%.2f FVU=%.4f MSE=%.5f cos=%.4f", source, layer, m["n_tokens"], m["l0_mean"], m["fvu"], m["mse_mean"], m["cosine_mean"])

    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2, default=str) + "\n")
    np.savez_compressed(output_dir / "per_feature_fire_freq.npz", **fire_freq)
    write_summary(output_dir, results, args)
    logger.info("Artifacts: %s", output_dir)
    logger.info("  results.json: %s", results_path)
    logger.info("  per_feature_fire_freq.npz: %s", output_dir / "per_feature_fire_freq.npz")
    logger.info("  summary.md: %s", output_dir / "summary.md")

    wandb_logger.finish(summary={
        f"final/{source}/L{layer}/{k}": v
        for source in args.sources for layer in layers
        for k, v in results["by_source"][source][str(layer)].items()
    })


def write_summary(output_dir: Path, results: dict, args: argparse.Namespace) -> None:
    """Write a compact markdown table (source x layer -> L0, FVU) + reproduction command."""
    lines = [
        f"# Transcoder input-shift eval — {results['config']['scan_name']}",
        "",
        f"GemmaScope `{args.gemmascope_width}` / `{args.gemmascope_l0}` transcoders on "
        f"`{args.base_model}`; target = MLP_base(x). Reconstruction FVU and L0 per source × layer.",
        "",
        "| source | layer | n_tokens | L0 | FVU | MSE | cosine |",
        "|---|---|---|---|---|---|---|",
    ]
    for source in args.sources:
        for layer in results["config"]["layers"]:
            m = results["by_source"][source][str(layer)]
            lines.append(
                f"| {source} | {layer} | {m['n_tokens']} | {m['l0_mean']:.1f} | "
                f"{m['fvu']:.4f} | {m['mse_mean']:.5f} | {m['cosine_mean']:.4f} |"
            )
    lines += [
        "",
        "**Reading it:** if `instruct` L0/FVU ≈ `base`, the GemmaScope transcoders transfer; "
        "if materially higher, the input shift warrants re-fine-tuning (or a pivot to "
        "adapter-only / ReLP). Per-feature fire frequencies are in `per_feature_fire_freq.npz`.",
        "",
        f"Invocation: `{' '.join(shlex.quote(a) for a in sys.argv)}`",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
