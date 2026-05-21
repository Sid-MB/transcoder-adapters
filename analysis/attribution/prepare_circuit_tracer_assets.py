"""Prepare circuit-tracer assets without allocating a GPU.

This script performs the CPU/file-I/O part of the circuit-tracer workflow and
writes a JSON manifest for a later GPU attribution job.

Inputs:
    --transcoder_model_path:
        Hugging Face repo ID or local checkpoint for the transcoder-adapted
        model.
    --base_model:
        Base model name to write into the circuit-tracer transcoder config.
    --prompts:
        Prompt file or prompt directory that will be used by the later
        attribution job.  This script records the path in the manifest but does
        not tokenize prompts or load the model.
    --feature_data_dir:
        Optional collected feature-data run directory.  When provided, this
        script packs its ``features/*.json`` files into circuit-tracer local
        feature-example files unless the run already contains a complete
        ``circuit_tracer_features/`` packed cache from
        ``collect_feature_activations --export_circuit_tracer_features``.

Outputs:
    PRODUCTS_DIR/circuit_tracer_transcoders/<model>/
        ``config.yaml`` and ``layer_N.safetensors`` files, created only if the
        consistent output directory is missing.
    PRODUCTS_DIR/circuit_tracer_features/<feature-run>/
        ``index.json.gz`` and ``layer_N.bin`` files, created only if
        ``--feature_data_dir`` is provided and no packed cache already exists
        inside the collected feature-data directory.
    --manifest_path:
        JSON file containing exact paths for the GPU attribution job:
        ``checkpoint``, ``prompts``, ``output_dir``, ``scan``, and
        ``features_dir``.

Example:
    uv run --extra viz python -m analysis.attribution.prepare_circuit_tracer_assets --transcoder_model_path siddharthmb/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860 --base_model google/gemma-2-2b --feature_data_dir /nlp/scr/siddharth/sparse-adaptation/feature_data/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860_20260519_171751_15493160 --prompts analysis/attribution/prompts/interesting_small --run_name interesting_small --manifest_path logs/attribution/interesting_small_manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from analysis.attribution.run_circuit_tracer_pipeline import (
    default_feature_output_dir,
    default_graph_output_dir,
    default_transcoder_output_dir,
    ensure_feature_data_conversion,
    ensure_transcoder_conversion,
    scan_name_for_feature_output,
)
from helpers.log import logger, setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare CPU-only circuit-tracer assets and write a GPU attribution manifest."
    )
    parser.add_argument("--transcoder_model_path", required=True, help="HF repo ID or local transcoder checkpoint")
    parser.add_argument("--base_model", required=True, help="Base model name for circuit-tracer transcoder config")
    parser.add_argument("--prompts", required=True, type=Path, help="Directory of .txt prompts, or one .txt file")
    parser.add_argument("--feature_data_dir", default=None, help="Optional collected feature-data run directory")
    parser.add_argument("--run_name", default="circuit_tracer", help="Graph run name and slug prefix")
    parser.add_argument("--transcoder_output_dir", type=Path, default=None)
    parser.add_argument("--feature_output_dir", type=Path, default=None)
    parser.add_argument("--graph_output_dir", type=Path, default=None)
    parser.add_argument("--manifest_path", type=Path, required=True)
    parser.add_argument("--n_layers", type=int, default=None, help="Override feature-data layer count")
    parser.add_argument("--n_features", type=int, default=None, help="Override feature-data feature count")
    parser.add_argument("--feature_input_hook", default="ln2.hook_normalized")
    parser.add_argument("--feature_output_hook", default="hook_mlp_out")
    parser.add_argument("--activation", default="relu", choices=["relu"])
    return parser


def prepare_assets(args: argparse.Namespace) -> dict[str, Any]:
    transcoder_output_dir = args.transcoder_output_dir or default_transcoder_output_dir(
        args.transcoder_model_path
    )
    ensure_transcoder_conversion(
        args.transcoder_model_path,
        transcoder_output_dir,
        base_model=args.base_model,
        feature_input_hook=args.feature_input_hook,
        feature_output_hook=args.feature_output_hook,
        activation=args.activation,
    )

    feature_output_dir = ensure_feature_data_conversion(
        args.feature_data_dir,
        args.feature_output_dir,
        n_layers=args.n_layers,
        n_features=args.n_features,
    )
    if args.feature_data_dir is not None and feature_output_dir is None:
        feature_output_dir = default_feature_output_dir(args.feature_data_dir)

    graph_output_dir = args.graph_output_dir or default_graph_output_dir(
        transcoder_model_path=args.transcoder_model_path,
        run_name=args.run_name,
    )
    graph_output_dir.mkdir(parents=True, exist_ok=True)
    scan = scan_name_for_feature_output(feature_output_dir, args.run_name)
    features_dir = str(feature_output_dir) if feature_output_dir is not None else None

    manifest = {
        "transcoder_model_path": args.transcoder_model_path,
        "base_model": args.base_model,
        "prompts": str(args.prompts),
        "run_name": args.run_name,
        "checkpoint": args.transcoder_model_path,
        "transcoder_output_dir": str(transcoder_output_dir),
        "feature_data_dir": args.feature_data_dir,
        "feature_output_dir": str(feature_output_dir) if feature_output_dir is not None else None,
        "graph_output_dir": str(graph_output_dir),
        "output_dir": str(graph_output_dir),
        "scan": scan,
        "features_dir": features_dir,
    }
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info(f"Wrote circuit-tracer pipeline manifest: {args.manifest_path}")
    logger.info(f"Graph output directory: {graph_output_dir}")
    return manifest


def main() -> None:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args()
    try:
        prepare_assets(args)
    except Exception as e:
        parser.error(str(e))


if __name__ == "__main__":
    main()
