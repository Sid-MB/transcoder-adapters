"""Write compact display variants for existing base-vs-adapter overlay graphs."""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis.attribution.run_base_adapter_comparison import (
    DEFAULT_COMPACT_BASE_ERROR_NODES,
    DEFAULT_COMPACT_BASE_FEATURE_NODES,
    write_compact_overlay_graphs,
)
from helpers.log import logger, setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create compact display overlays from an existing overlay graph directory."
    )
    parser.add_argument("--overlay_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
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
    return parser


def main() -> None:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args()
    max_base_feature_nodes = (
        None if args.max_base_feature_nodes < 0 else args.max_base_feature_nodes
    )
    max_base_error_nodes = None if args.max_base_error_nodes < 0 else args.max_base_error_nodes
    paths = write_compact_overlay_graphs(
        overlay_graph_dir=args.overlay_dir,
        compact_overlay_graph_dir=args.output_dir,
        max_base_feature_nodes=max_base_feature_nodes,
        max_base_error_nodes=max_base_error_nodes,
        max_adapter_feature_nodes=args.max_adapter_feature_nodes,
    )
    logger.info(f"Wrote {len(paths)} compact overlay graph(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
