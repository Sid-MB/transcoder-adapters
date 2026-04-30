"""
Shared harness for metadata-driven feature annotators.

Annotators own their scoring/classification state, while this module handles
the common disk IO, archive/merge semantics, and per-feature iteration.
"""

from __future__ import annotations

import json
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from tqdm import tqdm

from helpers.log import logger


@dataclass
class FeatureAnnotationResult:
    """Tags, persisted scores, and display fields returned by an annotator."""

    tags: list[str]
    scores: dict[str, Any] = field(default_factory=dict)
    hit_data: dict[str, Any] = field(default_factory=dict)


class FeatureAnnotator(ABC):
    """Base class for annotators that operate on feature metadata entries."""

    annotation_name: str
    # Set this for migration from older files without auto_tags ownership metadata.
    owned_tags: frozenset[str] | None = None

    def setup(self, metadata: dict[str, Any]) -> None:
        """Load any run-level state needed before per-feature annotation."""

    @abstractmethod
    def annotate_feature(
        self,
        feature: dict[str, Any],
        metadata: dict[str, Any],
    ) -> FeatureAnnotationResult:
        """Return scores/classifications for one feature and its run metadata."""

    def merge_annotation(
        self,
        entry: dict[str, Any],
        result: FeatureAnnotationResult,
    ) -> dict[str, Any]:
        existing_tags = entry.get("tags") or []
        auto_tags = dict(entry.get("auto_tags") or {})
        previous_tags = set(auto_tags.get(self.annotation_name) or [])
        if self.owned_tags is not None:
            previous_tags.update(self.owned_tags)

        merged_tags = sorted(({*existing_tags} - previous_tags) | set(result.tags))
        auto_scores = dict(entry.get("auto_scores") or {})
        if result.scores:
            auto_scores[self.annotation_name] = result.scores
        else:
            auto_scores.pop(self.annotation_name, None)

        if result.tags:
            auto_tags[self.annotation_name] = sorted(set(result.tags))
        else:
            auto_tags.pop(self.annotation_name, None)

        merged = {
            **entry,
            "tags": merged_tags,
            "notes": entry.get("notes", ""),
        }
        if auto_scores:
            merged["auto_scores"] = auto_scores
        else:
            merged.pop("auto_scores", None)
        if auto_tags:
            merged["auto_tags"] = auto_tags
        else:
            merged.pop("auto_tags", None)
        return merged

    def has_existing_annotation(self, entry: dict[str, Any]) -> bool:
        auto_tags = entry.get("auto_tags") or {}
        auto_scores = entry.get("auto_scores") or {}
        if self.annotation_name in auto_tags or self.annotation_name in auto_scores:
            return True
        if self.owned_tags is None:
            return False
        return bool(set(entry.get("tags") or []) & self.owned_tags)

    def make_hit(
        self,
        feature: dict[str, Any],
        result: FeatureAnnotationResult,
    ) -> dict[str, Any]:
        return {
            "cantor_id": feature["cantor_id"],
            "layer": feature["layer"],
            "feature": feature["feature"],
            "tags": result.tags,
            **result.hit_data,
        }

    def sort_key(self, hit: dict[str, Any]) -> Any:
        return ()

    def format_hit(self, hit: dict[str, Any]) -> str:
        tag_text = ",".join(hit["tags"])
        return (
            f"L{hit['layer']} F{hit['feature']} cantor={hit['cantor_id']} "
            f"tags={tag_text}"
        )


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(data, f, separators=(",", ":"))
        f.write("\n")
    tmp_path.replace(path)


def archive_existing_annotations(path: Path) -> Path:
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


def annotate_features(
    metadata: dict[str, Any],
    annotations: dict[str, Any],
    annotator: FeatureAnnotator,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    annotator.setup(metadata)
    hits: list[dict[str, Any]] = []

    for feature in tqdm(metadata.get("features") or []):
        result = annotator.annotate_feature(feature, metadata)
        cantor_id = str(feature["cantor_id"])
        entry = annotations.get(cantor_id) or {}
        should_update = bool(result.tags) or annotator.has_existing_annotation(entry)
        if not should_update:
            continue

        annotations[cantor_id] = annotator.merge_annotation(entry, result)
        if result.tags:
            hits.append(annotator.make_hit(feature, result))

    hits.sort(key=annotator.sort_key, reverse=True)
    return annotations, hits


def _annotator_list(
    annotator: FeatureAnnotator | Sequence[FeatureAnnotator],
) -> list[FeatureAnnotator]:
    if isinstance(annotator, FeatureAnnotator):
        return [annotator]
    annotators = list(annotator)
    if not annotators:
        raise ValueError("At least one annotator is required")
    annotation_names = [a.annotation_name for a in annotators]
    if len(annotation_names) != len(set(annotation_names)):
        raise ValueError(f"Annotator names must be unique: {annotation_names}")
    return annotators


def annotate_features_with_annotators(
    metadata: dict[str, Any],
    annotations: dict[str, Any],
    annotators: Sequence[FeatureAnnotator],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    hits_by_annotator: dict[str, list[dict[str, Any]]] = {}
    for annotator in annotators:
        annotations, hits = annotate_features(metadata, annotations, annotator)
        hits_by_annotator[annotator.annotation_name] = hits
    return annotations, hits_by_annotator


def run_annotation(
    *,
    data_dir: Path,
    annotations_file: Path | None,
    annotator: FeatureAnnotator | Sequence[FeatureAnnotator],
    replace_all: bool = False,
    top_k: int = 25,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    metadata_path = data_dir / "feature_metadata.json"
    annotations_path = annotations_file or (data_dir / "feature_annotations.json")
    metadata = load_json(metadata_path)
    annotators = _annotator_list(annotator)

    if replace_all and annotations_path.is_file():
        archive_path = archive_existing_annotations(annotations_path)
        logger.info(
            "Overwriting annotations: archived existing %s to %s",
            annotations_path,
            archive_path,
        )
        annotations: dict[str, Any] = {}
    else:
        annotations = load_json(annotations_path) if annotations_path.is_file() else {}

    annotations, hits_by_annotator = annotate_features_with_annotators(
        metadata,
        annotations,
        annotators,
    )
    save_json(annotations_path, annotations)

    for annotator in annotators:
        hits = hits_by_annotator[annotator.annotation_name]
        logger.info(
            "%s annotated %s features in %s",
            annotator.annotation_name,
            len(hits),
            annotations_path,
        )
        for hit in hits[:top_k]:
            logger.info(annotator.format_hit(hit))

    return annotations, hits_by_annotator


def add_common_annotation_args(parser: Any) -> None:
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument(
        "--annotations_file",
        type=Path,
        default=None,
        help=(
            "JSON file to write. Defaults to <data_dir>/feature_annotations.json. "
            "Existing annotations are updated in place unless --replace_all is passed."
        ),
    )
    parser.add_argument(
        "--replace_all",
        action="store_true",
        help=(
            "Archive the existing annotations file and start from an empty file. "
            "By default, annotators update only their own tags/scores in place."
        ),
    )
    parser.add_argument("--top_k", type=int, default=25, help="Number of hits to log")
