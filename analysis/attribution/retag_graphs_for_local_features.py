"""Re-tag circuit-tracer graphs so the local server shows feature examples, + UI labels.

<!-- Created by Claude Code session "implement: new 07/02 gemmascope transcoder experiments". 2026-07-09. -->

The circuit-tracer frontend only fetches feature examples from the LOCAL server's ``/features/``
endpoint when a graph's ``metadata.scan`` starts with ``/`` (or ``./``); otherwise it builds a
remote HuggingFace URL and (for our scans) 404s, so no examples appear
(see ``frontend/assets/feature_examples/init-feature-examples.js: featureUrl``). Graphs built by
``compare_finetuned_transcoder_graphs`` / ``run_base_attribution`` are tagged with the long
GemmaScope scan name, so their examples never load locally.

This rewrites, in place, for every graph JSON in a dir (and ``graph-metadata.json``):
  * ``metadata.scan`` -> a LOCAL scan (leading ``/``) so examples load from ``--features_dir``.
    The scan string is also shown in the graph-picker dropdown, so pick something readable.
  * ``metadata.title_prefix`` -> a dropdown tag (e.g. ``[ORIGINAL]`` / ``[FINE-TUNED L0/24/25]``).
  * node ``clerp`` for nodes on ``--mark_layers`` -> a marker (e.g. re-collected-on-fine-tuned),
    so those nodes are visibly flagged in the graph.

Usage:
    uv run --no-sync python -m analysis.attribution.retag_graphs_for_local_features \
        --graph_dir <dir> --scan /gemmascope_original --title_prefix "[ORIGINAL GemmaScope]"
    # fine-tuned side, flag the re-collected layers:
    uv run --no-sync python -m analysis.attribution.retag_graphs_for_local_features \
        --graph_dir <dir> --scan "/gemmascope_finetuned_L0-24-25" \
        --title_prefix "[FINE-TUNED L0/24/25]" --mark_layers 0 24 25 \
        --mark_text "re-collected on fine-tuned transcoder"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from helpers.log import logger, setup_logging

SKIP = {"graph-metadata.json", "combined_attribution_manifest.json", "run_attribution_args.json"}


def retag_graph(path: Path, *, scan: str, title_prefix: str | None, mark_layers: set[int], mark_text: str) -> int:
    payload = json.loads(path.read_text())
    meta = payload.setdefault("metadata", {})
    meta["scan"] = scan
    if title_prefix is not None:
        meta["title_prefix"] = title_prefix
    marked = 0
    if mark_layers:
        for node in payload.get("nodes", []):
            try:
                layer = int(node.get("layer"))
            except (TypeError, ValueError):
                continue
            if layer in mark_layers and "transcoder" in str(node.get("feature_type") or ""):
                existing = str(node.get("clerp") or "").strip()
                tag = f"⟳ {mark_text} (L{layer})"
                node["clerp"] = tag if not existing or existing.startswith("⟳") else f"{tag} — {existing}"
                marked += 1
    path.write_text(json.dumps(payload))
    return marked


def retag_metadata_index(path: Path, *, scan: str, title_prefix: str | None) -> None:
    payload = json.loads(path.read_text())
    for entry in payload.get("graphs", []):
        entry["scan"] = scan
        if title_prefix is not None:
            entry["title_prefix"] = title_prefix
    path.write_text(json.dumps(payload))


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph_dir", type=Path, required=True, help="Directory of circuit-tracer graph JSONs (+ graph-metadata.json).")
    ap.add_argument("--scan", required=True, help="Local scan name (MUST start with '/'), e.g. /gemmascope_original.")
    ap.add_argument("--title_prefix", default=None, help="Dropdown tag, e.g. '[ORIGINAL GemmaScope]'.")
    ap.add_argument("--mark_layers", nargs="*", type=int, default=[], help="Layers whose feature nodes get a clerp marker (e.g. 0 24 25).")
    ap.add_argument("--mark_text", default="re-collected on fine-tuned transcoder", help="Marker text for --mark_layers nodes.")
    args = ap.parse_args()

    if not args.scan.startswith(("/", "./")):
        raise SystemExit(f"--scan must start with '/' (local features), got {args.scan!r}")

    mark_layers = set(args.mark_layers)
    graph_jsons = [p for p in sorted(args.graph_dir.glob("*.json")) if p.name not in SKIP]
    total_marked = 0
    for path in graph_jsons:
        total_marked += retag_graph(path, scan=args.scan, title_prefix=args.title_prefix, mark_layers=mark_layers, mark_text=args.mark_text)

    index_path = args.graph_dir / "graph-metadata.json"
    if index_path.exists():
        retag_metadata_index(index_path, scan=args.scan, title_prefix=args.title_prefix)

    logger.info("Re-tagged %d graphs in %s: scan=%s title_prefix=%s (marked %d nodes on layers %s)", len(graph_jsons), args.graph_dir, args.scan, args.title_prefix, total_marked, sorted(mark_layers) or "-")


if __name__ == "__main__":
    main()
