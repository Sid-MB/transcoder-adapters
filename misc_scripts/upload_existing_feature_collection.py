"""Upload an already-collected feature-data directory to the Hugging Face Hub.

The feature collector (``analysis.features.collect_feature_activations`` /
``...collect_base_feature_activations``) normally uploads its packed
``circuit_tracer_features/`` cache to the Hub at the end of a run. This script
does that same upload for a collection directory that already exists on disk
(e.g. an older run produced before upload was wired in, or a run whose upload
was skipped). It reconstructs the deterministic repo id + dedup config from the
directory's saved ``collect_feature_activations_args.json`` and pushes the
packed cache plus required sidecar files.

Only directories that contain a packed cache (``circuit_tracer_features/index.json.gz``)
and the sidecar files can be uploaded; the per-feature ``features/*.json`` files
are never uploaded (they are a local dashboard artifact).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from analysis.features.hub_upload import (
    build_feature_collection_repo_id,
    check_feature_collection_exists,
    reserve_feature_collection_repo,
    upload_circuit_tracer_features_to_hub,
)
from helpers.log import logger

# Both collectors save their parsed run arguments, but under collector-specific
# filenames: the adapter collector (collect_feature_activations) and the base
# collector (collect_base_feature_activations).
ARGS_FILENAMES = (
    "collect_feature_activations_args.json",
    "collect_base_feature_activations_args.json",
)


def _load_args_namespace(output_dir: Path) -> SimpleNamespace:
    """Rebuild the argparse namespace from the saved run-arguments JSON.

    ``build_feature_collection_repo_id`` reads attributes off an argparse-style
    object, so a ``SimpleNamespace`` wrapping the saved dict is sufficient. Tries
    each collector's args filename so this works for base and adapter collections.
    """
    for name in ARGS_FILENAMES:
        args_path = output_dir / name
        if args_path.is_file():
            return SimpleNamespace(**json.loads(args_path.read_text()))
    raise FileNotFoundError(
        f"Cannot rebuild collection config; none of {ARGS_FILENAMES} found in {output_dir}. "
        "This directory was not produced by a feature collector."
    )


def upload_one(output_dir: Path, dry_run: bool) -> str:
    """Reserve (if needed) and upload one collection directory; return repo id."""
    args = _load_args_namespace(output_dir)
    repo_id, config = build_feature_collection_repo_id(args)

    packed_index = output_dir / "circuit_tracer_features" / "index.json.gz"
    if not packed_index.is_file():
        raise FileNotFoundError(
            f"No packed cache to upload ({packed_index} missing). "
            "This is an old per-feature-JSON-only collection and cannot be uploaded "
            "in the supported packed format."
        )

    if dry_run:
        logger.info(f"[dry-run] would upload {output_dir} -> https://huggingface.co/{repo_id}")
        return repo_id

    # Create + seed the repo when it does not already exist; uploading is then safe
    # whether the repo was freshly reserved or already present.
    if check_feature_collection_exists(repo_id, config) is None:
        reserve_feature_collection_repo(repo_id, config)
    upload_circuit_tracer_features_to_hub(repo_id=repo_id, output_dir=output_dir, config=config)
    logger.info(f"Uploaded {output_dir} -> https://huggingface.co/{repo_id}")
    return repo_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "output_dirs",
        nargs="+",
        help=(
            "One or more feature-data directories to upload (each must contain one of "
            f"{ARGS_FILENAMES}, circuit_tracer_features/index.json.gz, and the sidecar "
            "files). The Hub repo id is derived deterministically from each directory's "
            "saved collection arguments, so re-uploading the same collection is idempotent."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help=(
            "Print the destination repo id for each directory without creating repos or "
            "uploading. Use this to preview what will be pushed before committing."
        ),
    )
    args = parser.parse_args()

    results: list[tuple[str, str]] = []
    failures: list[tuple[str, str]] = []
    for raw in args.output_dirs:
        output_dir = Path(raw)
        try:
            repo_id = upload_one(output_dir, dry_run=args.dry_run)
            results.append((raw, repo_id))
        except Exception as exc:  # noqa: BLE001 - report per-dir and continue
            logger.error(f"FAILED to upload {raw}: {exc}")
            failures.append((raw, str(exc)))

    logger.info("=== upload summary ===")
    for raw, repo_id in results:
        logger.info(f"  OK   {raw} -> https://huggingface.co/{repo_id}")
    for raw, err in failures:
        logger.info(f"  FAIL {raw}: {err}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
