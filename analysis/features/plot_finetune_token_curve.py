"""Plot per-layer FVU and L0 vs training tokens from a fine-tune run's intermediate evals.

<!-- Created by Claude Code session "implement: new 07/02 gemmascope transcoder experiments". 2026-07-15. -->

Reads ``finetune_report.json['eval_history']`` (written when ``--eval_every_tokens`` is set) and
plots, per layer, FVU and L0 against training tokens with dashed base-target lines. Answers "does
this layer keep improving with more tokens, or plateau?" — e.g. L25 flattens above its base FVU.

Usage:
    uv run --no-sync python -m analysis.features.plot_finetune_token_curve \
        --finetune_dir <10M run dir> --all_layers_dir <Exp1 26-layer run> \
        --out_dir .claude/products/transcoder_input_shift
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#2563eb", "#dc2626", "#16a34a", "#7c3aed", "#f59e0b", "#0891b2"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--finetune_dir", type=Path, required=True)
    ap.add_argument("--all_layers_dir", type=Path, required=True, help="Exp1 26-layer run (base target FVU/L0).")
    ap.add_argument("--out_dir", type=Path, default=Path(".claude/products/transcoder_input_shift"))
    ap.add_argument("--out_name", default="transcoder_finetune_token_curve.png")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rep = json.load(open(args.finetune_dir / "finetune_report.json"))
    hist = rep["eval_history"]
    layers = [int(x) for x in rep["config"]["layers"]]
    tokens = [e["tokens"] / 1e6 for e in hist]
    base = json.load(open(args.all_layers_dir / "results.json"))["by_source"]["base"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    tot = rep["config"].get("train_tokens", 0) / 1e6
    fig.suptitle(f"Fine-tune progress vs training tokens (per-layer, {tot:.0f}M-token run)\n"
                 "dashed = base operating point; FVU flattens above base at the deepest layer (L25)",
                 fontsize=12, fontweight="bold")
    for key, base_key, title, ylabel, ax in (
        ("fvu", "fvu", "Reconstruction error (FVU)  —  lower is better", "FVU", axes[0]),
        ("l0_mean", "l0_mean", "Sparsity (active features/token, a.k.a. L0)", "active features/token", axes[1]),
    ):
        for i, L in enumerate(layers):
            c = COLORS[i % len(COLORS)]
            ys = [e["per_layer"][str(L)][key] for e in hist]
            ax.plot(tokens, ys, "o-", color=c, label=f"layer {L}", ms=4)
            ax.axhline(base[str(L)][base_key], color=c, ls="--", lw=1, alpha=0.7)
        ax.set_xlabel("training tokens (millions)")
        ax.set_ylabel(ylabel); ax.set_title(title, fontweight="bold")
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = args.out_dir / args.out_name
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
