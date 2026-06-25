"""STEP 5 + 6: Aggregate base/adapter/error node participation across AGREE vs DIVERGE graphs.

For each combined-attribution graph JSON we record two parallel measures of graph composition:

  1. metadata-counts: metadata.comparison.node_counts {base_features, adapter_features, error_nodes}.
     These are what tag_combined_graph recorded for the nodes that survived --max_feature_nodes
     (and the --max_error_nodes cap, which here was -1 = keep all, so error counts are uncapped).

  2. raw-node-counts: recomputed directly from each graph's nodes. Feature nodes have
     feature_type == 'cross layer transcoder'; we split them into base vs adapter using the
     tagged source_model field (which tag_combined_graph set from the original combined index,
     base = combined index < 16384, adapter = combined index >= 16384). Error nodes are any node
     whose feature_type contains 'error' ('mlp reconstruction error').

The two should match here because the graphs were tagged in this same run with --max_error_nodes -1;
we report both and flag any mismatch. Per graph we record base #, adapter #, error #, and
adapter_fraction = adapter / (base + adapter). We then aggregate mean +/- std per bucket and join
each prompt's base-vs-instruct agreement metric from agreement_metrics.json.

Outputs: aggregate_results.json (machine-readable) and a printed per-prompt + summary table.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from helpers.log import logger, setup_logging

N_BASE = 16384  # GemmaScope width_16k features per layer = base/adapter combined-index split.

# Gemma chat-template / structural tokens. Adapter features that fire on these are
# "format-handling" overhead (the instruct model processing the turn scaffolding) and are
# present even on converge prompts; adapter features on other (content) tokens reflect
# instruct-specific behavioral work. We split adapter participation along this line so the
# converge-prompt adapter floor (≈all template) is distinguishable from divergent content work.
TEMPLATE_TOKENS = {"<bos>", "<eos>", "<start_of_turn>", "<end_of_turn>", "<pad>", "user", "model"}


def _is_template_token(tok: str) -> bool:
    """True for chat-scaffolding tokens: special turn markers, role words, or pure whitespace."""
    return tok in TEMPLATE_TOKENS or tok.strip() == ""


def _is_feature_node(node: dict) -> bool:
    return str(node.get("feature_type") or "") == "cross layer transcoder"


def _is_error_node(node: dict) -> bool:
    return "error" in str(node.get("feature_type") or "")


CORE_COUNT_KEYS = ("base_features", "adapter_features", "error_nodes")


def raw_counts_from_nodes(payload: dict) -> dict:
    """Recompute base/adapter/error node counts directly from graph nodes.

    Adapter feature nodes are additionally split by the token they fire on:
    ``adapter_template`` (chat-scaffolding tokens) vs ``adapter_content`` (everything else),
    with the content tokens themselves collected for inspection. The three CORE_COUNT_KEYS
    are what metadata.comparison.node_counts records; the adapter_template/adapter_content
    split is the extra measure (it does not change base/adapter/error totals).
    """
    prompt_tokens = (payload.get("metadata") or {}).get("prompt_tokens") or []
    base = adapter = error = adapter_template = adapter_content = 0
    content_tokens: list[str] = []
    for node in payload["nodes"]:
        if _is_error_node(node):
            error += 1
        elif _is_feature_node(node):
            if node.get("source_model") == "adapter":
                adapter += 1
                ctx_idx = node.get("ctx_idx")
                tok = prompt_tokens[ctx_idx] if isinstance(ctx_idx, int) and 0 <= ctx_idx < len(prompt_tokens) else ""
                if _is_template_token(tok):
                    adapter_template += 1
                else:
                    adapter_content += 1
                    content_tokens.append(tok)
            else:
                base += 1
    return {
        "base_features": base,
        "adapter_features": adapter,
        "error_nodes": error,
        "adapter_template": adapter_template,
        "adapter_content": adapter_content,
        "adapter_content_tokens": content_tokens,
    }


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def analyze_bucket(graph_dir: Path, agreement_by_id: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []
    for graph_path in sorted(graph_dir.glob("*.json")):
        if graph_path.name in {"graph-metadata.json", "combined_attribution_manifest.json"}:
            continue
        payload = json.loads(graph_path.read_text())
        if not {"metadata", "nodes", "links"}.issubset(payload):
            continue
        meta_counts = payload["metadata"]["comparison"]["node_counts"]
        raw_counts = raw_counts_from_nodes(payload)

        # slug form: {run}__{prompt_id}__h{hash}; prompt_id may contain underscores, hash starts with h.
        slug = payload["metadata"].get("slug", graph_path.stem)
        parts = slug.split("__")
        prompt_id = parts[1] if len(parts) >= 2 else slug
        agreement = agreement_by_id.get(prompt_id, {})

        base = raw_counts["base_features"]
        adapter = raw_counts["adapter_features"]
        error = raw_counts["error_nodes"]
        denom = base + adapter
        adapter_fraction = adapter / denom if denom else float("nan")
        adapter_template = raw_counts["adapter_template"]
        adapter_content = raw_counts["adapter_content"]
        # Of this prompt's adapter features, what fraction fire on non-template (content) tokens.
        adapter_content_fraction = adapter_content / adapter if adapter else float("nan")
        # meta node_counts only carries the 3 core totals, so compare on those keys.
        core_raw = {k: raw_counts[k] for k in CORE_COUNT_KEYS}

        rows.append(
            {
                "prompt_id": prompt_id,
                "slug": slug,
                "graph_path": str(graph_path),
                "meta_counts": meta_counts,
                "raw_counts": raw_counts,
                "counts_match": meta_counts == core_raw,
                "base_features": base,
                "adapter_features": adapter,
                "error_nodes": error,
                "adapter_fraction": adapter_fraction,
                "adapter_template": adapter_template,
                "adapter_content": adapter_content,
                "adapter_content_fraction": adapter_content_fraction,
                "adapter_content_tokens": raw_counts["adapter_content_tokens"],
                "kl_instruct_base": agreement.get("kl_instruct_base"),
                "top1_match": agreement.get("top1_match"),
                "topk_overlap": agreement.get("topk_overlap"),
                "user_text": agreement.get("user_text"),
            }
        )
    return rows


def aggregate(rows: list[dict]) -> dict:
    return {
        "n_graphs": len(rows),
        "base_features": summarize([r["base_features"] for r in rows]),
        "adapter_features": summarize([r["adapter_features"] for r in rows]),
        "adapter_template": summarize([r["adapter_template"] for r in rows]),
        "adapter_content": summarize([r["adapter_content"] for r in rows]),
        "error_nodes": summarize([r["error_nodes"] for r in rows]),
        "adapter_fraction": summarize([r["adapter_fraction"] for r in rows]),
        # nan when a graph had 0 adapter features; drop those (v == v is False for nan).
        "adapter_content_fraction": summarize([r["adapter_content_fraction"] for r in rows if r["adapter_content_fraction"] == r["adapter_content_fraction"]]),
        "kl_instruct_base": summarize([r["kl_instruct_base"] for r in rows if r["kl_instruct_base"] is not None]),
    }


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Aggregate base/adapter/error participation per bucket.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=Path(__file__).resolve().parents[3] / ".claude" / "products" / "base_vs_adapter_circuit_trace.png",
        help="Where to write the summary figure after aggregating. Defaults to the repo's .claude/products/ dir (the convention for generated media, for easy discovery). Used unless --no-plot is given.",
    )
    parser.add_argument(
        "--no-plot",
        dest="plot",
        action="store_false",
        help="Skip auto-rendering the summary figure. By default analyze_graphs renders plot_results.py's figure right after writing aggregate_results.json so the plot is always fresh; pass this when you only want the JSON/table (e.g. headless batch aggregation).",
    )
    args = parser.parse_args()

    agreement_payload = json.loads((args.root / "agreement_metrics.json").read_text())
    agreement_by_id = {c["id"]: c for c in agreement_payload["candidates"]}

    buckets = {
        "agree": analyze_bucket(args.root / "graphs" / "agree", agreement_by_id),
        "diverge": analyze_bucket(args.root / "graphs" / "diverge", agreement_by_id),
    }

    # Report per-prompt table.
    header = f"{'bucket':8s} {'prompt_id':24s} {'KL':>7s} {'t1':>3s} {'base':>5s} {'adapt':>6s} {'err':>5s} {'adapt_frac':>10s} {'match':>6s}"
    logger.info(header)
    logger.info("-" * len(header))
    for bucket, rows in buckets.items():
        for r in rows:
            kl = r["kl_instruct_base"]
            logger.info(
                f"{bucket:8s} {r['prompt_id']:24s} {kl:7.2f} {str(r['top1_match'])[0]:>3s} "
                f"{r['base_features']:5d} {r['adapter_features']:6d} {r['error_nodes']:5d} "
                f"{r['adapter_fraction']:10.4f} {str(r['counts_match'])[0]:>6s}"
            )

    aggregates = {bucket: aggregate(rows) for bucket, rows in buckets.items()}

    logger.info("")
    logger.info("=== AGGREGATE (mean +/- std) ===")
    metric_header = f"{'metric':18s} {'AGREE':>20s} {'DIVERGE':>20s}"
    logger.info(metric_header)
    logger.info("-" * len(metric_header))
    for metric in [
        "base_features",
        "adapter_features",
        "adapter_template",
        "adapter_content",
        "error_nodes",
        "adapter_fraction",
        "adapter_content_fraction",
        "kl_instruct_base",
    ]:
        a = aggregates["agree"][metric]
        d = aggregates["diverge"][metric]
        if metric in ("adapter_fraction", "adapter_content_fraction"):
            fmt = lambda s: f"{s['mean']:.4f} +/- {s['std']:.4f}"
        elif metric == "kl_instruct_base":
            fmt = lambda s: f"{s['mean']:.2f} +/- {s['std']:.2f}"
        else:
            fmt = lambda s: f"{s['mean']:.1f} +/- {s['std']:.1f}"
        logger.info(f"{metric:18s} {fmt(a):>20s} {fmt(d):>20s}")

    all_match = all(r["counts_match"] for rows in buckets.values() for r in rows)
    logger.info("")
    logger.info(f"raw-node-counts == metadata node_counts for all graphs: {all_match}")

    out = {
        "n_base_split": N_BASE,
        "per_prompt": buckets,
        "aggregates": aggregates,
        "raw_equals_metadata_for_all": all_match,
    }
    out_path = args.root / "aggregate_results.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    logger.info(f"Wrote {out_path}")

    # Auto-render the summary figure from the JSON we just wrote (unless --no-plot).
    # plot() logs "PLOT WRITTEN: <abspath>" so the figure location appears in this run's log.
    if args.plot:
        from plot_results import plot  # local import: only pull in matplotlib when actually plotting

        plot(out_path, args.plot_out)


if __name__ == "__main__":
    main()
