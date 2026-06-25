"""Plot the AGREE-vs-DIVERGE circuit-trace results from aggregate_results.json.

Reads the aggregate_results.json produced by analyze_graphs.py and renders a 2x3 figure:
  A) adapter-feature count (mean +/- std) per bucket
  B) error-node count (mean +/- std) per bucket
  C) adapter features split into template (format floor) vs content work (stacked)
  D) adapter_fraction = adapter/(base+adapter) (mean +/- std) per bucket
  E) adapter_content_fraction = content / adapter features (mean +/- std) per bucket
  F) per-prompt scatter of KL(instruct||base) vs adapter-feature count, colored by bucket

This runs automatically at the end of analyze_graphs.py (the script that writes
aggregate_results.json); run it standalone only to re-render from an existing JSON. The
plot() function is the reusable entry point and is what analyze_graphs imports.

Usage:
  uv run --no-sync python experiments/base_vs_adapter_circuit_trace/scripts/plot_results.py \
    --results experiments/base_vs_adapter_circuit_trace/aggregate_results.json \
    --out .claude/products/base_vs_adapter_circuit_trace.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from helpers.log import logger, setup_logging

BUCKETS = ["agree", "diverge"]
COLORS = {"agree": "#2c7fb8", "diverge": "#d95f02"}


def _bar(ax, agg, metric, title, ylabel):
    means = [agg[b][metric]["mean"] for b in BUCKETS]
    stds = [agg[b][metric]["std"] for b in BUCKETS]
    ax.bar(BUCKETS, means, yerr=stds, capsize=6, color=[COLORS[b] for b in BUCKETS], alpha=0.85)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s, f"{m:.2f}" if m < 5 else f"{m:.1f}", ha="center", va="bottom", fontsize=10)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.margins(y=0.18)


def _stacked_template_content(ax, agg):
    """Stacked bars: adapter features split into template (format floor) vs content work."""
    tmpl = [agg[b]["adapter_template"]["mean"] for b in BUCKETS]
    cont = [agg[b]["adapter_content"]["mean"] for b in BUCKETS]
    ax.bar(BUCKETS, tmpl, color="#9ecae1", label="template (format floor)")
    ax.bar(BUCKETS, cont, bottom=tmpl, color="#fd8d3c", label="content (instruct work)")
    for i, (t, c) in enumerate(zip(tmpl, cont)):
        if t > 0:
            ax.text(i, t / 2, f"{t:.1f}", ha="center", va="center", fontsize=9)
        if c > 0.05:
            ax.text(i, t + c / 2, f"{c:.1f}", ha="center", va="center", fontsize=9, color="white")
    ax.set_title("Adapter features: format floor vs content")
    ax.set_ylabel("# adapter feature nodes")
    ax.legend(fontsize=8, loc="upper left")
    ax.margins(y=0.15)


def plot(results: Path, out: Path) -> Path:
    """Render the 2x2 figure from an aggregate_results.json and save it to `out`.

    Reusable entry point shared by the CLI (main) and analyze_graphs.py's auto-plot step.
    Returns the absolute output path and logs it (via helpers.log) so the figure location
    lands in the run log / sbatch log for easy discovery.
    """
    results, out = Path(results), Path(out)
    d = json.loads(results.read_text())
    agg = d["aggregates"]
    per = d["per_prompt"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    n = {b: agg[b]["n_graphs"] for b in BUCKETS}
    fig.suptitle(
        f"Base+adapter circuit tracing: AGREE (n={n['agree']}) vs DIVERGE (n={n['diverge']}) prompts",
        fontsize=14, fontweight="bold",
    )

    _bar(axes[0, 0], agg, "adapter_features", "Adapter features recruited", "# adapter feature nodes")
    _bar(axes[0, 1], agg, "error_nodes", "Error nodes (unexplained residual)", "# error nodes")
    _stacked_template_content(axes[0, 2], agg)

    _bar(axes[1, 0], agg, "adapter_fraction", "Adapter fraction = adapter/(base+adapter)", "fraction")
    _bar(axes[1, 1], agg, "adapter_content_fraction", "Adapter content fraction", "content / adapter features")

    ax = axes[1, 2]
    for b in BUCKETS:
        xs = [r["kl_instruct_base"] for r in per[b]]
        ys = [r["adapter_features"] for r in per[b]]
        ax.scatter(xs, ys, color=COLORS[b], label=f"{b} (n={len(xs)})", alpha=0.8, edgecolor="k", linewidth=0.4)
    ax.set_xscale("log")
    ax.set_xlabel("KL(instruct || base)  [log]")
    ax.set_ylabel("# adapter features")
    ax.set_title("Per-prompt: divergence vs adapter recruitment")
    ax.legend(fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    abs_out = out.resolve()
    logger.info(f"PLOT WRITTEN: {abs_out}")
    return abs_out


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, required=True, help="aggregate_results.json from analyze_graphs.py")
    p.add_argument("--out", type=Path, required=True, help="Output image path (PNG); parent dirs created.")
    args = p.parse_args()
    plot(args.results, args.out)


if __name__ == "__main__":
    main()
