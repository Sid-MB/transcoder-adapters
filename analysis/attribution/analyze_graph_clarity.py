"""Quantify circuit-tracer graph clarity: error-node vs feature-node participation.

<!-- Created by Claude Code session "implement: new 07/02 gemmascope transcoder experiments". 2026-07-09. -->

Tasks 2 & 4 of the input-shift follow-up. Experiment 1 showed GemmaScope transcoders
reconstruct the base MLP worse on instruct hidden states (higher FVU). The graph-level analog
of reconstruction failure is the **error node**: circuit-tracer inserts one MLP-reconstruction-
error node per (layer, position) carrying `true_mlp_out - transcoder_out`. If error nodes carry
much of a graph's structure, the graph is "muddy" (computation not explained by interpretable
features); if the graph is dominated by feature nodes, it is clean.

This reads an existing base-vs-adapter combined-attribution aggregate (produced by
``experiments/base_vs_adapter_circuit_trace/scripts/analyze_graphs.py``) and reports, per bucket:
  * error-node fraction   = error_nodes / (base_features + error_nodes)
  * adapter fraction       = adapter_features / (base_features + adapter_features)
These answer, respectively, "are the base (GemmaScope) graphs clean?" (task 4) and "how much
does the adapter / difference circuit contribute?" (task 2).

Usage:
    uv run --no-sync python -m analysis.attribution.analyze_graph_clarity \
        --aggregate experiments/base_vs_adapter_circuit_trace/aggregate_results.json \
        --out experiments/transcoder_input_shift/graph_clarity.json
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from helpers.log import logger, setup_logging


def summarize_bucket(rows: list[dict]) -> dict:
    err_frac = [r["error_nodes"] / max(1, r["base_features"] + r["error_nodes"]) for r in rows]
    adapt_frac = [r["adapter_fraction"] for r in rows]
    return {
        "n_graphs": len(rows),
        "error_node_fraction_mean": st.mean(err_frac),
        "error_node_fraction_max": max(err_frac),
        "adapter_fraction_mean": st.mean(adapt_frac),
        "base_features_mean": st.mean(r["base_features"] for r in rows),
        "error_nodes_mean": st.mean(r["error_nodes"] for r in rows),
        "adapter_content_mean": st.mean(r.get("adapter_content", 0) for r in rows),
        "adapter_template_mean": st.mean(r.get("adapter_template", 0) for r in rows),
    }


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aggregate", type=Path, default=Path("experiments/base_vs_adapter_circuit_trace/aggregate_results.json"))
    ap.add_argument("--out", type=Path, default=Path("experiments/transcoder_input_shift/graph_clarity.json"))
    args = ap.parse_args()

    data = json.load(open(args.aggregate))
    result = {"source_aggregate": str(args.aggregate), "by_bucket": {}}
    for bucket, rows in data["per_prompt"].items():
        s = summarize_bucket(rows)
        result["by_bucket"][bucket] = s
        logger.info(
            "%-8s n=%2d | error-node frac %.3f (max %.3f) | adapter frac %.3f | base_feat %.0f err %.0f | adapter content/template %.1f/%.1f",
            bucket, s["n_graphs"], s["error_node_fraction_mean"], s["error_node_fraction_max"],
            s["adapter_fraction_mean"], s["base_features_mean"], s["error_nodes_mean"],
            s["adapter_content_mean"], s["adapter_template_mean"],
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    logger.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
