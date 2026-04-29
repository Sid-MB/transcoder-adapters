"""
Annotate features that concentrate around the assistant response boundary.

This is a lightweight post-processing step for ``collect_feature_activations.py``
outputs. It reads ``feature_metadata.json`` only, scores existing feature stats,
and writes a persistent ``feature_annotations.json`` file consumable by the local
dashboard. By default, an existing annotations file is moved into an ``archive/``
subdirectory before fresh annotations are written. Pass ``--merge`` to merge new
tags into an existing annotations file instead.

Example:
    python -m analysis.features.annotate_assistant_response_features \
        --data_dir /path/to/feature_run
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from helpers.log import logger, setup_logging

ASSISTANT_MARKER_REGION = "assistant_marker"
ANSWER_REGION = "answer"


def _load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(data, f, separators=(",", ":"))
        f.write("\n")
    tmp_path.replace(path)


def _archive_existing_annotations(path: Path) -> Path:
    archive_dir = path.parent / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = archive_dir / f"{path.stem}_{timestamp}{path.suffix}"
    suffix = 1
    while archive_path.exists():
        archive_path = archive_dir / f"{path.stem}_{timestamp}_{suffix}{path.suffix}"
        suffix += 1
    shutil.move(str(path), str(archive_path))
    return archive_path


def _region_fraction(feature: dict, region: str) -> float:
    return float((feature.get("region_fraction") or {}).get(region) or 0.0)


def _region_density(feature: dict, region: str) -> float:
    return float((feature.get("region_density") or {}).get(region) or 0.0)


def score_feature(feature: dict, tokens_per_region: dict[str, int]) -> dict[str, float]:
    """Return assistant-boundary scores from metadata for one feature."""
    marker_fraction = _region_fraction(feature, ASSISTANT_MARKER_REGION)
    answer_fraction = _region_fraction(feature, ANSWER_REGION)
    marker_density = _region_density(feature, ASSISTANT_MARKER_REGION)
    answer_density = _region_density(feature, ANSWER_REGION)

    non_assistant_density_vals = [
        float(d)
        for region, d in (feature.get("region_density") or {}).items()
        if region not in {ASSISTANT_MARKER_REGION, ANSWER_REGION}
    ]
    non_assistant_density = (
        sum(non_assistant_density_vals) / len(non_assistant_density_vals)
        if non_assistant_density_vals
        else 0.0
    )

    assistant_fraction = marker_fraction + answer_fraction
    assistant_density = marker_density + answer_density
    density_lift = assistant_density / max(non_assistant_density, 1e-12)

    marker_tokens = int(tokens_per_region.get(ASSISTANT_MARKER_REGION) or 0)
    has_marker_region = marker_tokens > 0

    return {
        "assistant_fraction": assistant_fraction,
        "assistant_marker_fraction": marker_fraction,
        "answer_fraction": answer_fraction,
        "assistant_density": assistant_density,
        "assistant_marker_density": marker_density,
        "answer_density": answer_density,
        "density_lift": density_lift,
        "has_marker_region": 1.0 if has_marker_region else 0.0,
    }


def classify_feature(
    feature: dict,
    scores: dict[str, float],
    *,
    assistant_fraction_threshold: float,
    marker_fraction_threshold: float,
    density_lift_threshold: float,
) -> list[str]:
    """Classify one feature into annotation tags."""
    tags: list[str] = []
    if (
        scores["assistant_fraction"] >= assistant_fraction_threshold
        and scores["density_lift"] >= density_lift_threshold
    ):
        tags.append("assistant_response")

    if (
        scores["assistant_marker_fraction"] >= marker_fraction_threshold
        and scores["has_marker_region"] > 0
    ):
        tags.extend(["assistant_response", "assistant_token"])

    if (
        scores["answer_fraction"] >= assistant_fraction_threshold
        and "assistant_token" not in tags
    ):
        tags.extend(["assistant_response", "near_assistant"])

    return sorted(set(tags))


def annotate_features(
    metadata: dict,
    annotations: dict,
    *,
    assistant_fraction_threshold: float,
    marker_fraction_threshold: float,
    density_lift_threshold: float,
) -> tuple[dict, list[dict]]:
    tokens_per_region = metadata.get("tokens_per_region") or {}
    hits: list[dict] = []

    for feature in metadata.get("features") or []:
        cantor_id = str(feature["cantor_id"])
        scores = score_feature(feature, tokens_per_region)
        tags = classify_feature(
            feature,
            scores,
            assistant_fraction_threshold=assistant_fraction_threshold,
            marker_fraction_threshold=marker_fraction_threshold,
            density_lift_threshold=density_lift_threshold,
        )
        if not tags:
            continue

        entry = annotations.get(cantor_id) or {}
        existing_tags = entry.get("tags") or []
        merged_tags = sorted({*existing_tags, *tags})
        annotations[cantor_id] = {
            **entry,
            "tags": merged_tags,
            "notes": entry.get("notes", ""),
            "auto_scores": {
                "assistant_response": scores,
            },
        }
        hits.append({
            "cantor_id": feature["cantor_id"],
            "layer": feature["layer"],
            "feature": feature["feature"],
            "tags": tags,
            **scores,
        })

    hits.sort(
        key=lambda h: (
            h["assistant_marker_fraction"],
            h["assistant_fraction"],
            h["density_lift"],
        ),
        reverse=True,
    )
    return annotations, hits


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Find and persist annotations for assistant-response features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument(
        "--annotations_file",
        type=Path,
        default=None,
        help=(
            "JSON file to write. Defaults to <data_dir>/feature_annotations.json. "
            "Unless --merge is passed, an existing file is archived next to itself "
            "under archive/ before the new file is written."
        ),
    )
    parser.add_argument(
        "--assistant_fraction_threshold",
        type=float,
        default=0.35,
        help="Minimum fraction of feature fires in assistant marker + answer regions",
    )
    parser.add_argument(
        "--marker_fraction_threshold",
        type=float,
        default=0.20,
        help="Minimum fraction of feature fires on assistant marker tokens",
    )
    parser.add_argument(
        "--density_lift_threshold",
        type=float,
        default=2.0,
        help="Minimum assistant-region density lift over non-assistant regions",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help=(
            "Preserve the existing annotations file and merge newly detected tags "
            "into it. By default, an existing annotations file is archived to "
            "<annotations_dir>/archive/<stem>_<timestamp><suffix> and replaced with "
            "a fresh file so threshold changes produce a clean result."
        ),
    )
    parser.add_argument("--top_k", type=int, default=25, help="Number of hits to print")
    args = parser.parse_args()

    metadata_path = args.data_dir / "feature_metadata.json"
    annotations_path = args.annotations_file or (args.data_dir / "feature_annotations.json")
    metadata = _load_json(metadata_path)
    if not args.merge and annotations_path.is_file():
        archive_path = _archive_existing_annotations(annotations_path)
        logger.info(
            "Overwriting annotations: archived existing %s to %s",
            annotations_path,
            archive_path,
        )
        annotations = {}
    else:
        annotations = _load_json(annotations_path) if annotations_path.is_file() else {}
    annotations, hits = annotate_features(
        metadata,
        annotations,
        assistant_fraction_threshold=args.assistant_fraction_threshold,
        marker_fraction_threshold=args.marker_fraction_threshold,
        density_lift_threshold=args.density_lift_threshold,
    )
    _save_json(annotations_path, annotations)

    print(f"Annotated {len(hits)} features in {annotations_path}")
    for hit in hits[: args.top_k]:
        tag_text = ",".join(hit["tags"])
        print(
            f"L{hit['layer']} F{hit['feature']} cantor={hit['cantor_id']} "
            f"tags={tag_text} assistant_frac={hit['assistant_fraction']:.3f} "
            f"marker_frac={hit['assistant_marker_fraction']:.3f} "
            f"density_lift={hit['density_lift']:.2f}"
        )


if __name__ == "__main__":
    main()
