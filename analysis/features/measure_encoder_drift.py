"""Measure how far fine-tuning rotated each transcoder feature's encoder direction.

<!-- Created by Claude Code session "implement: new 07/02 gemmascope transcoder experiments". 2026-07-09. -->

A feature's top activating examples are determined by its ENCODER row ``W_enc[i]`` (the direction
that triggers it). So the question "do the already-collected (original-weight) feature examples
still apply after fine-tuning?" reduces to "how much did ``W_enc[i]`` rotate?". This loads the
pretrained GemmaScope transcoders and the fine-tuned ``finetuned_layer_*.safetensors`` and reports,
per fine-tuned layer, the per-feature cosine similarity ``cos(W_enc_orig[i], W_enc_ft[i])``.

If cosine ≈ 1 for essentially all features (as observed: mean ≈ 0.9997, none < 0.99), the feature
detectors are unchanged, so the original examples are EXACT and no re-collection is needed. CPU-only.

Usage:
    uv run --no-sync python -m analysis.features.measure_encoder_drift \
        --finetune_dir <ft run dir> --layers 0 24 25 \
        --gemmascope_width width_16k --gemmascope_l0 average_l0_76
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from helpers.log import logger, setup_logging

from analysis.attribution.run_base_adapter_comparison import (
    _load_gemmascope_transcoders,
    build_gemmascope_transcoder_config,
    resolve_gemmascope_l0_values,
)


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--finetune_dir", type=Path, required=True)
    ap.add_argument("--layers", nargs="+", type=int, default=[0, 24, 25])
    ap.add_argument("--base_model", default="google/gemma-2-2b")
    ap.add_argument("--gemmascope_repo", default="google/gemma-scope-2b-pt-transcoders")
    ap.add_argument("--gemmascope_width", required=True)
    ap.add_argument("--gemmascope_l0", required=True)
    ap.add_argument("--gemmascope_l0_match", default="nearest")
    ap.add_argument("--gemmascope_n_layers", type=int, default=26)
    ap.add_argument("--output", type=Path, default=None, help="Where to write encoder_drift.json (default: <finetune_dir>/encoder_drift.json).")
    args = ap.parse_args()

    l0v = resolve_gemmascope_l0_values(repo=args.gemmascope_repo, width=args.gemmascope_width, l0=args.gemmascope_l0, n_layers=args.gemmascope_n_layers, l0_match=args.gemmascope_l0_match)
    cfg = build_gemmascope_transcoder_config(repo=args.gemmascope_repo, width=args.gemmascope_width, l0=args.gemmascope_l0, n_layers=args.gemmascope_n_layers, model_name=args.base_model, feature_input_hook="ln2.hook_normalized", feature_output_hook="hook_mlp_out", l0_values=l0v, l0_match=args.gemmascope_l0_match)
    ts = _load_gemmascope_transcoders(cfg, device=torch.device("cpu"), dtype=torch.float32)

    result: dict[str, dict] = {}
    logger.info("layer | mean cos | median | frac<0.99 | frac<0.9")
    for layer in args.layers:
        orig = ts[layer].W_enc.float()
        ft = load_file(str(args.finetune_dir / f"finetuned_layer_{layer}.safetensors"))["W_enc"].float()
        cos = F.cosine_similarity(orig, ft, dim=1)
        stats = {
            "mean_cos": cos.mean().item(),
            "median_cos": cos.median().item(),
            "min_cos": cos.min().item(),
            "frac_below_0.99": (cos < 0.99).float().mean().item(),
            "frac_below_0.9": (cos < 0.9).float().mean().item(),
        }
        result[str(layer)] = stats
        logger.info("%5d | %.4f | %.4f | %.3f | %.3f", layer, stats["mean_cos"], stats["median_cos"], stats["frac_below_0.99"], stats["frac_below_0.9"])

    out = args.output or (args.finetune_dir / "encoder_drift.json")
    out.write_text(json.dumps({"scan": cfg["scan_name"], "layers": args.layers, "per_layer": result}, indent=2) + "\n")
    logger.info("Wrote %s", out)
    logger.info("Interpretation: cos≈1 => feature detectors unchanged => already-collected examples are exact (no re-collection needed).")


if __name__ == "__main__":
    main()
