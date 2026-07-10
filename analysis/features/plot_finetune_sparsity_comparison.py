"""Compare transcoder fine-tune WITHOUT vs WITH a sparsity (L1) penalty — FVU and L0.

<!-- Created by Claude Code session "implement: new 07/02 gemmascope transcoder experiments". 2026-07-10. -->

Reconstruction-only fine-tuning improves FVU but lets L0 drift up (the frozen JumpReLU threshold
no longer matches the shifted encoder). Adding a decoder-norm-weighted L1 penalty holds L0 near the
base operating point. This renders a 4-bars-per-layer comparison for both metrics:

    before FT  ·  after FT (no penalty)  ·  after FT (+L1)  ·  base (target level)

so you can see the +L1 run keeps L0 near base (blue) without giving up the FVU gain. This is a NEW
figure; it does not touch transcoder_finetune_before_after.png.

Usage:
    uv run --no-sync python -m analysis.features.plot_finetune_sparsity_comparison \
        --nopenalty_dir <ft run, l1_coeff=0> --l1_dir <ft run, l1_coeff>0> \
        --all_layers_dir <Exp1 26-layer run, for base target> \
        --out_dir .claude/products/transcoder_input_shift
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BEFORE_C, NOPEN_C, L1_C, BASE_C = "#dc2626", "#f59e0b", "#7c3aed", "#2563eb"  # red / orange / purple / blue


def _report(run_dir: Path) -> dict:
    return json.load(open(run_dir / "finetune_report.json"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nopenalty_dir", type=Path, required=True, help="Fine-tune run dir with l1_coeff=0.")
    ap.add_argument("--l1_dir", type=Path, required=True, help="Fine-tune run dir with l1_coeff>0.")
    ap.add_argument("--all_layers_dir", type=Path, required=True, help="Exp1 26-layer run dir (base target FVU/L0).")
    ap.add_argument("--out_dir", type=Path, default=Path(".claude/products/transcoder_input_shift"))
    ap.add_argument("--out_name", default="transcoder_finetune_sparsity_comparison.png")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    nopen, l1 = _report(args.nopenalty_dir), _report(args.l1_dir)
    layers = [int(x) for x in nopen["config"]["layers"]]
    lc = l1["config"].get("l1_coeff", "?")
    l1_coeff = dict(zip(layers, lc)) if isinstance(lc, list) and len(lc) == len(layers) else lc

    exp1 = json.load(open(args.all_layers_dir / "results.json"))
    base_src = exp1["by_source"]["base"]

    def series(metric_key: str, base_key: str):
        before = [nopen["before"][str(L)][metric_key] for L in layers]      # instruct, original transcoder
        after_nopen = [nopen["after"][str(L)][metric_key] for L in layers]
        after_l1 = [l1["after"][str(L)][metric_key] for L in layers]
        base = [base_src[str(L)][base_key] for L in layers]                 # base model + original transcoder
        return before, after_nopen, after_l1, base

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        f"Transcoder fine-tune on instruct inputs: no sparsity penalty vs +L1 (l1_coeff={l1_coeff})\n"
        "L1 holds L0 near the base operating point while keeping the reconstruction (FVU) gain",
        fontsize=13, fontweight="bold",
    )
    specs = [("fvu", "fvu", "Reconstruction error (FVU)  ↓ better", axes[0]),
             ("l0_mean", "l0_mean", "Sparsity L0 (active features/token)  → match base", axes[1])]
    x = range(len(layers))
    w = 0.2
    for metric_key, base_key, title, ax in specs:
        before, after_nopen, after_l1, base = series(metric_key, base_key)
        ax.bar([xi - 1.5 * w for xi in x], before, w, color=BEFORE_C, label="instruct, before FT")
        ax.bar([xi - 0.5 * w for xi in x], after_nopen, w, color=NOPEN_C, label="after FT (no penalty)")
        ax.bar([xi + 0.5 * w for xi in x], after_l1, w, color=L1_C, label="after FT (+L1)")
        ax.bar([xi + 1.5 * w for xi in x], base, w, color=BASE_C, alpha=0.75, label="base (target level)")
        ax.set_xticks(list(x)); ax.set_xticklabels([f"L{L}" for L in layers])
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(metric_key)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = args.out_dir / args.out_name
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
