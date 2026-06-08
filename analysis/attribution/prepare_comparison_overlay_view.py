"""Write compact display variants for existing base-vs-adapter overlay graphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.attribution.run_base_adapter_comparison import (
    DEFAULT_COMPACT_BASE_ERROR_NODES,
    DEFAULT_COMPACT_BASE_FEATURE_NODES,
    LOCAL_BASE_FEATURE_SCAN,
    LOCAL_FEATURE_SCAN,
    default_base_feature_scan_for_overlay_base_scan,
    write_compact_overlay_graphs,
)
from helpers.log import logger, setup_logging


def cap_from_arg(value: int | None) -> int | None:
    return None if value is not None and value < 0 else value


def default_output_dir(
    overlay_dir: Path,
    *,
    max_base_feature_nodes: int | None,
    max_base_error_nodes: int | None,
    max_adapter_feature_nodes: int | None,
) -> Path:
    def cap_label(name: str, value: int | None) -> str:
        return f"{name}{value}" if value is not None else f"{name}all"

    suffix_parts = [
        "compact",
        cap_label("base", max_base_feature_nodes),
        cap_label("error", max_base_error_nodes),
    ]
    if max_adapter_feature_nodes is not None:
        suffix_parts.append(cap_label("adapter", max_adapter_feature_nodes))
    return overlay_dir.parent / f"{overlay_dir.name}_{'_'.join(suffix_parts)}"


def infer_base_feature_scan(overlay_dir: Path) -> str | None:
    for graph_path in sorted(overlay_dir.glob("*.json")):
        if graph_path.name in {"graph-metadata.json", "run_attribution_args.json"}:
            continue
        try:
            payload = json.loads(graph_path.read_text())
        except json.JSONDecodeError:
            continue
        comparison = payload.get("metadata", {}).get("comparison", {})
        inferred = default_base_feature_scan_for_overlay_base_scan(comparison.get("base_scan"))
        if inferred is not None:
            return inferred
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create compact display overlays from an existing overlay graph directory."
    )
    parser.add_argument("--overlay_dir", required=True, type=Path)
    parser.add_argument(
        "--output_dir",
        default=None,
        type=Path,
        help=(
            "Directory for compact overlay JSONs. Defaults to a sibling of "
            "--overlay_dir named "
            "<overlay_dir.name>_compact_base{max_base_feature_nodes}_"
            "error{max_base_error_nodes}, with _adapter{max_adapter_feature_nodes} "
            "appended when an adapter cap is set. A disabled cap (-1) is written as 'all'."
        ),
    )
    parser.add_argument(
        "--max_base_feature_nodes",
        type=int,
        default=DEFAULT_COMPACT_BASE_FEATURE_NODES,
        help="Maximum base cross-layer feature nodes to keep per graph. Use -1 for no base cap.",
    )
    parser.add_argument(
        "--max_base_error_nodes",
        type=int,
        default=DEFAULT_COMPACT_BASE_ERROR_NODES,
        help="Maximum base reconstruction-error nodes to keep per graph. Use -1 for no base error cap.",
    )
    parser.add_argument(
        "--max_adapter_feature_nodes",
        type=int,
        default=None,
        help="Optional adapter cross-layer feature-node cap per graph.",
    )
    parser.add_argument(
        "--base_feature_scan",
        default="auto",
        help=(
            "Scan for base feature examples. Default 'auto' uses the standard "
            "Gemma-2 feature repo for google/gemma-scope-2b-pt-transcoders "
            f"overlays. Use a Hugging Face repo, {LOCAL_BASE_FEATURE_SCAN} for a "
            "local base feature directory, or an empty string to leave existing metadata unchanged."
        ),
    )
    parser.add_argument(
        "--adapter_feature_scan",
        default=None,
        help=(
            "Optional scan for adapter feature examples. Defaults to existing metadata; "
            f"use {LOCAL_FEATURE_SCAN} for comparison local serving."
        ),
    )
    return parser


def main() -> None:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args()
    max_base_feature_nodes = cap_from_arg(args.max_base_feature_nodes)
    max_base_error_nodes = cap_from_arg(args.max_base_error_nodes)
    max_adapter_feature_nodes = cap_from_arg(args.max_adapter_feature_nodes)
    output_dir = args.output_dir or default_output_dir(
        args.overlay_dir,
        max_base_feature_nodes=max_base_feature_nodes,
        max_base_error_nodes=max_base_error_nodes,
        max_adapter_feature_nodes=max_adapter_feature_nodes,
    )
    base_feature_scan = args.base_feature_scan
    if base_feature_scan == "auto":
        base_feature_scan = infer_base_feature_scan(args.overlay_dir)
    elif base_feature_scan == "":
        base_feature_scan = None
    paths = write_compact_overlay_graphs(
        overlay_graph_dir=args.overlay_dir,
        compact_overlay_graph_dir=output_dir,
        max_base_feature_nodes=max_base_feature_nodes,
        max_base_error_nodes=max_base_error_nodes,
        max_adapter_feature_nodes=max_adapter_feature_nodes,
        base_feature_scan=base_feature_scan,
        adapter_feature_scan=args.adapter_feature_scan,
    )
    logger.info(f"Wrote {len(paths)} compact overlay graph(s) to {output_dir}")


if __name__ == "__main__":
    main()
