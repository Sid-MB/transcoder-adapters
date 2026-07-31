"""Export collected feature examples to circuit-tracer local feature format.

This converts the per-feature JSON files produced by
``analysis.features.collect_feature_activations`` into the packed feature format
expected by the circuit-tracer frontend when serving local feature examples.

Required input:
    --feature_data_path:
        A collected feature-data run directory, or its ``features/`` subdirectory.
        The run directory must contain ``features/<cantor_id>.json`` files.  A
        ``feature_metadata.json`` file is used, when present, to infer
        ``n_layers`` and ``n_features``.

Optional inputs:
    --output_dir:
        Destination for the packed circuit-tracer feature files.  If omitted,
        this uses a consistent path under
        ``PRODUCTS_DIR/circuit_tracer_features/<feature-data-run>/``.
    --n_layers / --n_features:
        Explicit shape overrides.  These are required only when the shape cannot
        be inferred from ``feature_metadata.json`` or the run/checkpoint name.

Outputs:
    <output_dir>/index.json.gz:
        Compressed index mapping each layer to a ``layer_N.bin`` file and byte
        offsets for every feature in that layer.
    <output_dir>/layer_N.bin:
        Concatenated compressed feature JSON records for layer ``N``.  This is
        the path served by circuit-tracer's ``--features_dir`` option.

Existing output behavior:
    The default output path is deterministic.  If the output directory already
    exists, this script raises an error before doing any work, because that means
    the conversion has already been done.  The combined pipeline script catches
    that case earlier by skipping existing conversion directories.

Example:
    uv run --extra viz python -m analysis.attribution.export_circuit_tracer_feature_data --feature_data_path $LARGE_ARTIFACTS_DIR/transcoder-adapters/feature_data/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860_20260519_171751_15493160

Example with explicit shape:
    uv run --extra viz python -m analysis.attribution.export_circuit_tracer_feature_data --feature_data_path $LARGE_ARTIFACTS_DIR/transcoder-adapters/feature_data/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860_20260519_171751_15493160 --n_layers 26 --n_features 8192
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path
from typing import Any

from tqdm import tqdm

from analysis.attribution.export_circuit_tracer_transcoders import (
    require_new_conversion_output_dir,
)
from analysis.features.pack_features import pack_layer
from helpers.log import logger, setup_logging
from helpers.paths.output_path import generate_output_path

_SLUG_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_FEATURE_WIDTH_RE = re.compile(r"(?:^|[_-])tc(\d+)(?:[_-]|$)")


def normalize_feature_data_dir(feature_data_dir: str | Path) -> Path:
    path = Path(feature_data_dir).expanduser()
    if path.name == "features":
        return path.parent
    return path


def feature_json_dir(feature_data_dir: str | Path) -> Path:
    path = normalize_feature_data_dir(feature_data_dir)
    features_dir = path / "features"
    if not features_dir.is_dir():
        raise FileNotFoundError(f"Could not find feature JSON directory: {features_dir}")
    return features_dir


def default_output_dir(feature_data_dir: str | Path) -> Path:
    normalized_dir = normalize_feature_data_dir(feature_data_dir)
    slug = _SLUG_RE.sub("_", normalized_dir.name).strip("_") or "feature_data"
    return generate_output_path("circuit_tracer_features", slug, consistent=True)


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r") as f:
        return json.load(f)


def _infer_n_features_from_name(name: str) -> int | None:
    match = _FEATURE_WIDTH_RE.search(name)
    if match is None:
        return None
    return int(match.group(1))


def infer_feature_shape(
    feature_data_dir: str | Path,
    *,
    n_layers: int | None = None,
    n_features: int | None = None,
) -> tuple[int, int]:
    normalized_dir = normalize_feature_data_dir(feature_data_dir)
    metadata = _load_json_if_exists(normalized_dir / "feature_metadata.json")
    args = _load_json_if_exists(normalized_dir / "collect_feature_activations_args.json")

    metadata_features = metadata.get("features", []) if metadata is not None else []
    if metadata_features:
        if n_layers is None:
            n_layers = max(int(feature["layer"]) for feature in metadata_features) + 1
        if n_features is None:
            n_features = max(int(feature["feature"]) for feature in metadata_features) + 1

    if n_features is None:
        model_path = str(args.get("model_path", "")) if args is not None else ""
        n_features = _infer_n_features_from_name(normalized_dir.name) or _infer_n_features_from_name(model_path)

    if n_layers is None or n_features is None:
        raise ValueError(
            "Could not infer n_layers/n_features from feature_metadata.json or run name. "
            "Pass --n_layers and --n_features explicitly."
        )

    return n_layers, n_features


def export_circuit_tracer_feature_data(
    feature_data_dir: str | Path,
    output_dir: Path,
    *,
    n_layers: int | None = None,
    n_features: int | None = None,
) -> None:
    require_new_conversion_output_dir(
        output_dir,
        conversion_name="Circuit-tracer feature-data",
    )

    features_dir = feature_json_dir(feature_data_dir)
    n_layers, n_features = infer_feature_shape(
        feature_data_dir,
        n_layers=n_layers,
        n_features=n_features,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    index: dict[str, Any] = {"version": "1.0", "format": "variable_chunks"}

    logger.info(f"Packing feature data from {features_dir}")
    logger.info(f"Writing circuit-tracer feature files to {output_dir}")
    logger.info(f"Shape: {n_layers} layers x {n_features} features")

    total_found = 0
    total_possible = n_layers * n_features
    for layer in tqdm(range(n_layers), desc="Packing layers"):
        bin_path = output_dir / f"layer_{layer}.bin"
        offsets, n_found = pack_layer(str(features_dir), str(bin_path), layer, n_features)
        total_found += n_found
        index[str(layer)] = {
            "filename": f"layer_{layer}.bin",
            "offsets": offsets,
        }
        size_mb = bin_path.stat().st_size / 1e6
        tqdm.write(
            f"  Layer {layer}: {size_mb:.1f} MB, {n_found} features, {n_features - n_found} missing"
        )

    with gzip.open(output_dir / "index.json.gz", "wt") as f:
        json.dump(index, f)

    total_missing = total_possible - total_found
    logger.info(f"Done: {total_found}/{total_possible} features packed ({total_missing} missing)")
    logger.info(f"Output: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export collected feature JSONs to circuit-tracer local feature format."
    )
    parser.add_argument(
        "--feature_data_path",
        required=True,
        help="Feature-data run directory, or its features/ subdirectory.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help=(
            "Directory for index.json.gz and layer_N.bin. "
            "Default: consistent PRODUCTS_DIR/circuit_tracer_features/<feature-data-run>"
        ),
    )
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--n_features", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    output_dir = args.output_dir or default_output_dir(args.feature_data_path)
    export_circuit_tracer_feature_data(
        args.feature_data_path,
        output_dir,
        n_layers=args.n_layers,
        n_features=args.n_features,
    )


if __name__ == "__main__":
    main()
