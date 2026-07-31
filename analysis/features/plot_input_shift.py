"""Plot the GemmaScope transcoder input-shift results (base vs instruct hidden states).

<!-- Created by Claude Code session "implement: new 07/02 gemmascope transcoder experiments". 2026-07-09. -->

Reads results.json (+ fire_freq_drift.json) from transcoder_input_shift run dirs and renders
a multi-panel figure: reconstruction error (FVU) and sparsity (L0) vs layer for base vs
instruct inputs, the per-layer FVU gap, the chat-vs-web comparison, and firing-frequency
drift. Saves PNG(s) to the given output dir (default .claude/products/transcoder_input_shift).

Usage:
    uv run --no-sync python -m analysis.features.plot_input_shift \
        --all_layers_dir <26-layer chat run> --chat_dir <5-layer chat> --web_dir <5-layer web> \
        --out_dir .claude/products/transcoder_input_shift
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_C, INST_C, WEB_C = "#2563eb", "#dc2626", "#16a34a"  # blue / red / green


def load(run_dir: Path) -> dict:
    r = json.load(open(run_dir / "results.json"))
    layers = r["config"]["layers"]
    out = {"layers": layers, "base": {}, "instruct": {}}
    for src in ("base", "instruct"):
        out[src]["fvu"] = [r["by_source"][src][str(L)]["fvu"] for L in layers]
        out[src]["l0"] = [r["by_source"][src][str(L)]["l0_mean"] for L in layers]
    drift_p = run_dir / "fire_freq_drift.json"
    if drift_p.exists():
        d = json.load(open(drift_p))
        out["freq_r"] = [d["per_layer"][str(L)]["freq_pearson_r"] for L in layers]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all_layers_dir", type=Path, required=True, help="26-layer chat run dir.")
    ap.add_argument("--chat_dir", type=Path, required=True, help="5-layer chat run dir.")
    ap.add_argument("--web_dir", type=Path, required=True, help="5-layer web (fineweb) run dir.")
    ap.add_argument("--finetune_dir", type=Path, default=None, help="Optional re-finetune run dir (finetune_report.json) for a before/after panel.")
    ap.add_argument("--out_dir", type=Path, default=Path(".claude/products/transcoder_input_shift"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    al = load(args.all_layers_dir)
    chat = load(args.chat_dir)
    web = load(args.web_dir)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        "GemmaScope width_16k transcoders on gemma-2-2b: behavior on base vs. instruct hidden states\n"
        "Transcoders imitate the base MLP; we only change whose hidden states feed them (target = MLP_base(x))",
        fontsize=13, fontweight="bold",
    )
    L = al["layers"]

    # (0,0) FVU vs layer, all 26 layers
    ax = axes[0, 0]
    ax.plot(L, al["base"]["fvu"], "o-", color=BASE_C, label="base inputs", ms=4)
    ax.plot(L, al["instruct"]["fvu"], "s-", color=INST_C, label="instruct inputs", ms=4)
    ax.set_title("Reconstruction error (FVU) vs layer", fontweight="bold")
    ax.set_xlabel("layer"); ax.set_ylabel("FVU (frac. variance unexplained)")
    ax.legend(); ax.grid(alpha=0.3)

    # (0,1) FVU gap %
    ax = axes[0, 1]
    gap = [100 * (i - b) / b for b, i in zip(al["base"]["fvu"], al["instruct"]["fvu"])]
    colors = [INST_C if g > 0 else BASE_C for g in gap]
    ax.bar(L, gap, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("FVU increase on instruct inputs (%)", fontweight="bold")
    ax.set_xlabel("layer"); ax.set_ylabel("Δ FVU vs base (%)")
    for x in (0, 24, 25):
        ax.annotate(f"+{gap[x]:.0f}%", (x, gap[x]), ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")

    # (0,2) L0 vs layer
    ax = axes[0, 2]
    ax.plot(L, al["base"]["l0"], "o-", color=BASE_C, label="base inputs", ms=4)
    ax.plot(L, al["instruct"]["l0"], "s-", color=INST_C, label="instruct inputs", ms=4)
    ax.set_title("Sparsity L0 (active features/token) vs layer", fontweight="bold")
    ax.set_xlabel("layer"); ax.set_ylabel("mean L0")
    ax.legend(); ax.grid(alpha=0.3)

    # (1,0) firing-frequency Pearson r vs layer
    ax = axes[1, 0]
    if "freq_r" in al:
        ax.plot(L, al["freq_r"], "d-", color="#7c3aed", ms=4)
        ax.set_ylim(0.6, 1.02)
    ax.set_title("Per-feature firing-rate correlation (base vs instruct)", fontweight="bold")
    ax.set_xlabel("layer"); ax.set_ylabel("Pearson r")
    ax.axhline(1.0, color="k", lw=0.6, ls=":")
    ax.grid(alpha=0.3)
    ax.text(0.5, 0.05, "same features fire (Jaccard≈1); r<1 = drifted rates", transform=ax.transAxes,
            ha="center", fontsize=8, style="italic", color="#555")

    # (1,1) chat vs web FVU gap (5 layers) — is the shift chat-specific?
    ax = axes[1, 1]
    cl = chat["layers"]
    chat_gap = [100 * (i - b) / b for b, i in zip(chat["base"]["fvu"], chat["instruct"]["fvu"])]
    web_gap = [100 * (i - b) / b for b, i in zip(web["base"]["fvu"], web["instruct"]["fvu"])]
    x = range(len(cl)); w = 0.38
    ax.bar([xi - w / 2 for xi in x], chat_gap, w, color="#0891b2", label="chat")
    ax.bar([xi + w / 2 for xi in x], web_gap, w, color=WEB_C, label="web (fineweb)")
    ax.set_xticks(list(x)); ax.set_xticklabels(cl)
    ax.set_title("FVU increase (%): chat vs web — shift is not chat-specific", fontweight="bold")
    ax.set_xlabel("layer"); ax.set_ylabel("Δ FVU vs base (%)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")

    # (1,2) base absolute FVU (context: mid-layers reconstruct worst even for base)
    ax = axes[1, 2]
    ax.fill_between(L, al["base"]["fvu"], color=BASE_C, alpha=0.15)
    ax.plot(L, al["base"]["fvu"], "o-", color=BASE_C, ms=4)
    ax.set_title("Base-input FVU (transcoder quality baseline)", fontweight="bold")
    ax.set_xlabel("layer"); ax.set_ylabel("FVU")
    ax.grid(alpha=0.3)
    ax.text(0.5, 0.9, "mid-layers reconstruct worst even for base", transform=ax.transAxes,
            ha="center", fontsize=8, style="italic", color="#555")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = args.out_dir / "transcoder_input_shift_overview.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")

    # Separate before/after re-finetune figure.
    if args.finetune_dir is not None:
        rep = json.load(open(args.finetune_dir / "finetune_report.json"))
        fl = rep["config"]["layers"]
        before = [rep["before"][str(x)]["fvu"] for x in fl]
        after = [rep["after"][str(x)]["fvu"] for x in fl]
        # base-input FVU (Exp 1 target) at those layers, from the all-layers run
        base_ref = [al["base"]["fvu"][al["layers"].index(x)] for x in fl]
        fig2, ax = plt.subplots(figsize=(8, 5))
        xi = range(len(fl)); w = 0.28
        ax.bar([x - w for x in xi], before, w, color=INST_C, label="instruct, before FT")
        ax.bar([x for x in xi], after, w, color="#f59e0b", label="instruct, after FT")
        ax.bar([x + w for x in xi], base_ref, w, color=BASE_C, alpha=0.7, label="base (target level)")
        ax.set_xticks(list(xi)); ax.set_xticklabels([f"L{x}" for x in fl])
        ax.set_ylabel("FVU"); ax.set_title("Re-fine-tuning transcoders on instruct inputs closes the gap\n(2M tokens; target = MLP_base(x))", fontweight="bold")
        for i in xi:
            d = 100 * (after[i] - before[i]) / before[i]
            ax.annotate(f"{d:+.0f}%", (i, after[i]), ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.legend(); ax.grid(alpha=0.3, axis="y")
        fig2.tight_layout()
        out2 = args.out_dir / "transcoder_finetune_before_after.png"
        fig2.savefig(out2, dpi=130, bbox_inches="tight")
        print(f"wrote {out2}")

        # L0 (sparsity) twin of the before/after figure — is L0 still comparable after FT?
        l0_before = [rep["before"][str(x)]["l0_mean"] for x in fl]
        l0_after = [rep["after"][str(x)]["l0_mean"] for x in fl]
        l0_base = [al["base"]["l0"][al["layers"].index(x)] for x in fl]
        fig3, ax = plt.subplots(figsize=(8, 5))
        ax.bar([x - w for x in xi], l0_before, w, color=INST_C, label="instruct, before FT")
        ax.bar([x for x in xi], l0_after, w, color="#f59e0b", label="instruct, after FT")
        ax.bar([x + w for x in xi], l0_base, w, color=BASE_C, alpha=0.7, label="base (target level)")
        ax.set_xticks(list(xi)); ax.set_xticklabels([f"L{x}" for x in fl])
        ax.set_ylabel("L0 (active features / token)")
        ax.set_title("Sparsity (L0) before vs after re-fine-tune\n(threshold frozen; L0 drifts up as W_enc/b_enc shift)", fontweight="bold")
        for i in xi:
            ax.annotate(f"{l0_after[i]:.0f}", (i, l0_after[i]), ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.legend(); ax.grid(alpha=0.3, axis="y")
        fig3.tight_layout()
        out3 = args.out_dir / "transcoder_finetune_l0_before_after.png"
        fig3.savefig(out3, dpi=130, bbox_inches="tight")
        print(f"wrote {out3}")


if __name__ == "__main__":
    main()
