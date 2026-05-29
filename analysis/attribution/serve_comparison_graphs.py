"""Serve comparison overlay graphs with repo-local circuit-tracer UI overrides."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from analysis.attribution.comparison_frontend import prepare_comparison_frontend
from helpers.log import logger, setup_logging


def serve_until_interrupted(
    *,
    graph_file_dir: Path,
    features_dir: str | Path | None,
    port: int,
) -> None:
    from circuit_tracer.frontend.local_server import serve

    graph_file_dir = Path(graph_file_dir)
    if not graph_file_dir.is_dir():
        raise FileNotFoundError(f"Graph file directory does not exist: {graph_file_dir}")
    resolved_features_dir = str(Path(features_dir).resolve()) if features_dir is not None else None
    frontend_dir = prepare_comparison_frontend(graph_file_dir)

    logger.info(f"Starting comparison graph server on port {port}")
    logger.info(f"Serving graph directory: {graph_file_dir.resolve()}")
    if resolved_features_dir is not None:
        logger.info(f"Serving adapter feature directory: {resolved_features_dir}")

    server = serve(
        data_dir=str(graph_file_dir),
        frontend_dir=str(frontend_dir),
        features_dir=resolved_features_dir,
        port=port,
    )
    try:
        logger.info("Press Ctrl+C to stop the server.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping comparison graph server...")
        server.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve base-vs-adapter overlay graph JSONs with node-shape UI overrides."
    )
    parser.add_argument("--graph_file_dir", required=True, type=Path)
    parser.add_argument(
        "--features_dir",
        default=None,
        help="Optional adapter circuit_tracer_features directory for local feature examples.",
    )
    parser.add_argument("--port", type=int, default=8044)
    return parser


def main() -> None:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args()
    try:
        serve_until_interrupted(
            graph_file_dir=args.graph_file_dir,
            features_dir=args.features_dir,
            port=args.port,
        )
    except Exception as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
