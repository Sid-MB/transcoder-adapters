"""Run GPU attribution from a CPU-prepared circuit-tracer manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.attribution.run_attribution import run_attribution
from helpers.log import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run RelP attribution using paths from prepare_circuit_tracer_assets.py."
    )
    parser.add_argument("--manifest_path", required=True, type=Path)
    parser.add_argument("--prompt_format", choices=["auto", "raw", "chat"], default="auto")
    parser.add_argument("--max_n_logits", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_feature_nodes", type=int, default=10000)
    parser.add_argument("--node_threshold", type=float, default=0.8)
    parser.add_argument("--edge_threshold", type=float, default=0.98)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--device_map", type=str, default=None)
    parser.add_argument("--auto_shard_gpus", action="store_true")
    parser.add_argument("--serve", action="store_true", help="Start a circuit-tracer server after attribution")
    parser.add_argument("--port", type=int, default=8041)
    return parser


def attribution_args_from_manifest(args: argparse.Namespace) -> argparse.Namespace:
    manifest = json.loads(args.manifest_path.read_text())
    return argparse.Namespace(
        checkpoint=manifest["checkpoint"],
        run_name=manifest["run_name"],
        prompts=Path(manifest["prompts"]),
        output_dir=Path(manifest["output_dir"]),
        scan=manifest["scan"],
        prompt_format=args.prompt_format,
        max_n_logits=args.max_n_logits,
        batch_size=args.batch_size,
        max_feature_nodes=args.max_feature_nodes,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
        device=args.device,
        device_map=args.device_map,
        auto_shard_gpus=args.auto_shard_gpus,
        num_shards=1,
        shard_index=0,
        serve=args.serve,
        port=args.port,
        features_dir=manifest.get("features_dir"),
    )


def main() -> None:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args()
    try:
        run_attribution(attribution_args_from_manifest(args))
    except Exception as e:
        parser.error(str(e))


if __name__ == "__main__":
    main()
