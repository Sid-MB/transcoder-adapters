"""Run the local circuit-tracer workflow for transcoder-adapter checkpoints.

Skip behavior:
    - Transcoder conversion is skipped when the target converted directory
      already contains ``config.yaml``.
    - Feature-data conversion is skipped when the target feature directory
      already contains ``index.json.gz``.  If ``--feature_output_dir`` is not
      provided, the pipeline first auto-detects and reuses
      ``<feature_data_dir>/circuit_tracer_features/index.json.gz`` before
      falling back to the deterministic ``PRODUCTS_DIR/circuit_tracer_features``
      conversion path.
    - Attribution skips any prompt whose graph JSON already exists.  When all
      requested graph JSONs exist, ``run_attribution`` skips model loading and
      attribution entirely.  Graph paths are keyed by
      ``<graph_output_dir>/<run_name>__<prompt_file_stem>.json``; if a prompt
      directory mixes old and new prompts, the existing graph JSONs are left
      alone and only missing prompt graphs are computed.

This is the all-in-one entrypoint for producing circuit-tracer graph JSONs from
raw prompt files.  It reuses the lower-level conversion and attribution modules:

1. Convert a transcoder-adapter checkpoint into circuit-tracer's
   ``transcoder_set`` format if the consistent output directory does not already
   exist.
2. Optionally convert collected feature-dashboard data into circuit-tracer's
   local feature-example format if ``--feature_data_dir`` is provided and the
   consistent output directory does not already exist.
3. Run RelP attribution for each ``.txt`` prompt, skipping graph JSON files that
   already exist.
4. Optionally start the circuit-tracer web server when ``--serve`` is passed.

Required inputs:
    --transcoder_model_path:
        Hugging Face repo ID or local checkpoint directory for the
        transcoder-adapted model.
    --base_model:
        Hugging Face base model name to write into the converted transcoder
        ``config.yaml``.
    --prompts:
        A directory of ``.txt`` files, or a single ``.txt`` file.  Each file must
        include the target token at the end; attribution runs on the file content
        before that final token and scores the final token.

Optional input:
    --feature_data_dir:
        A collected feature-data run directory, such as
        ``/nlp/scr/.../feature_data/<run>/``.  The directory must contain
        ``features/*.json`` and should contain ``feature_metadata.json`` so the
        exporter can infer the layer and feature counts.  If the directory also
        contains a complete ``circuit_tracer_features/`` packed cache produced
        by ``collect_feature_activations --export_circuit_tracer_features``,
        that cache is reused automatically unless ``--feature_output_dir`` is
        set explicitly.

Outputs:
    PRODUCTS_DIR/circuit_tracer_transcoders/<model>/
        ``config.yaml`` plus ``layer_N.safetensors`` files.
    PRODUCTS_DIR/circuit_tracer_features/<feature-run>/
        ``index.json.gz`` plus ``layer_N.bin`` files, only when
        ``--feature_data_dir`` is provided and no packed cache already exists
        inside the collected feature-data directory.
    PRODUCTS_DIR/attribution_graphs/<run-name>_<model>/
        ``graph-metadata.json`` plus one ``{run_name}__{prompt}.json`` per
        prompt.

Example without feature examples:
    uv run --extra viz python -m analysis.attribution.run_circuit_tracer_pipeline --transcoder_model_path siddharthmb/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860 --base_model google/gemma-2-2b --prompts analysis/attribution/prompts/interesting_small --run_name interesting_small --prompt_format raw --max_feature_nodes 256 --batch_size 4 --max_n_logits 5 --port 8042

Example with local feature examples and serving:
    uv run --extra viz python -m analysis.attribution.run_circuit_tracer_pipeline --transcoder_model_path siddharthmb/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860 --base_model google/gemma-2-2b --feature_data_dir /nlp/scr/siddharth/sparse-adaptation/feature_data/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860_20260519_171751_15493160 --prompts analysis/attribution/prompts/interesting_small --run_name interesting_small --prompt_format raw --max_feature_nodes 256 --batch_size 4 --max_n_logits 5 --serve --port 8042
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from analysis.attribution.export_circuit_tracer_feature_data import (
    default_output_dir as default_feature_output_dir,
    export_circuit_tracer_feature_data,
    normalize_feature_data_dir,
)
from analysis.attribution.export_circuit_tracer_transcoders import (
    default_output_dir as default_transcoder_output_dir,
    export_circuit_tracer_transcoders,
)
from analysis.attribution.run_attribution import run_attribution
from helpers.log import logger, setup_logging
from helpers.paths.output_path import generate_output_path

_SLUG_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _slugify(value: str) -> str:
    return _SLUG_RE.sub("_", Path(value.rstrip("/")).name or value).strip("_") or "run"


def default_graph_output_dir(*, transcoder_model_path: str, run_name: str) -> Path:
    return generate_output_path(
        "attribution_graphs",
        f"{_slugify(run_name)}_{_slugify(transcoder_model_path)}",
        consistent=True,
    )


def ensure_transcoder_conversion(
    transcoder_model_path: str,
    output_dir: Path,
    *,
    base_model: str,
    feature_input_hook: str,
    feature_output_hook: str,
    activation: str,
) -> Path:
    completion_marker = output_dir / "config.yaml"
    if completion_marker.exists():
        logger.info(f"Skipping transcoder conversion; output already exists: {output_dir}")
        return output_dir
    if output_dir.exists():
        raise RuntimeError(
            f"Transcoder conversion directory exists but is incomplete: {output_dir}. "
            "Remove it before rerunning conversion."
        )

    export_circuit_tracer_transcoders(
        transcoder_model_path,
        output_dir,
        model_name=base_model,
        feature_input_hook=feature_input_hook,
        feature_output_hook=feature_output_hook,
        activation=activation,
    )
    return output_dir


def ensure_feature_data_conversion(
    feature_data_dir: str | None,
    output_dir: Path | None,
    *,
    n_layers: int | None,
    n_features: int | None,
) -> Path | None:
    if feature_data_dir is None:
        return None
    if output_dir is None:
        collected_packed_dir = normalize_feature_data_dir(feature_data_dir) / "circuit_tracer_features"
        if (collected_packed_dir / "index.json.gz").exists():
            default_output_dir = default_feature_output_dir(feature_data_dir)
            logger.info(
                "Auto-detected packed circuit-tracer features in the collected feature-data "
                f"directory; using {collected_packed_dir} instead of {default_output_dir}"
            )
            return collected_packed_dir
        output_dir = default_feature_output_dir(feature_data_dir)
    completion_marker = output_dir / "index.json.gz"
    if completion_marker.exists():
        logger.info(f"Skipping feature-data conversion; output already exists: {output_dir}")
        return output_dir
    if output_dir.exists():
        raise RuntimeError(
            f"Feature-data conversion directory exists but is incomplete: {output_dir}. "
            "Remove it before rerunning conversion."
        )

    export_circuit_tracer_feature_data(
        feature_data_dir,
        output_dir,
        n_layers=n_layers,
        n_features=n_features,
    )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export circuit-tracer assets if needed, run RelP attribution for prompt files, "
            "and optionally serve the resulting graph directory."
        )
    )
    parser.add_argument("--transcoder_model_path", required=True, help="HF repo ID or local transcoder checkpoint")
    parser.add_argument("--base_model", required=True, help="Base model name for circuit-tracer transcoder config")
    parser.add_argument("--prompts", required=True, type=Path, help="Directory of .txt prompts, or one .txt file")
    parser.add_argument("--feature_data_dir", default=None, help="Optional collected feature-data run directory")
    parser.add_argument("--run_name", default="circuit_tracer", help="Graph run name and slug prefix")
    parser.add_argument("--transcoder_output_dir", type=Path, default=None)
    parser.add_argument("--feature_output_dir", type=Path, default=None)
    parser.add_argument("--graph_output_dir", type=Path, default=None)
    parser.add_argument("--prompt_format", choices=["auto", "raw", "chat"], default="auto")
    parser.add_argument("--max_n_logits", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_feature_nodes", type=int, default=10000)
    parser.add_argument("--node_threshold", type=float, default=0.8)
    parser.add_argument("--edge_threshold", type=float, default=0.98)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--device_map", type=str, default=None)
    parser.add_argument("--auto_shard_gpus", action="store_true")
    parser.add_argument("--n_layers", type=int, default=None, help="Override feature-data layer count")
    parser.add_argument("--n_features", type=int, default=None, help="Override feature-data feature count")
    parser.add_argument("--feature_input_hook", default="ln2.hook_normalized")
    parser.add_argument("--feature_output_hook", default="hook_mlp_out")
    parser.add_argument("--activation", default="relu", choices=["relu"])
    parser.add_argument("--port", type=int, default=8041)
    parser.add_argument("--serve", action="store_true", help="Start the circuit-tracer server at the end")
    return parser


def run_pipeline(args: argparse.Namespace) -> dict[str, str]:
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

    graph_output_dir = args.graph_output_dir or default_graph_output_dir(
        transcoder_model_path=args.transcoder_model_path,
        run_name=args.run_name,
    )
    scan = str(feature_output_dir) if feature_output_dir is not None else args.run_name
    features_dir = str(feature_output_dir) if feature_output_dir is not None else None

    attribution_args = argparse.Namespace(
        checkpoint=args.transcoder_model_path,
        run_name=args.run_name,
        prompts=args.prompts,
        output_dir=graph_output_dir,
        scan=scan,
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
        features_dir=features_dir,
    )
    return run_attribution(attribution_args)


def main() -> None:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args()
    try:
        run_pipeline(args)
    except Exception as e:
        parser.error(str(e))


if __name__ == "__main__":
    main()
