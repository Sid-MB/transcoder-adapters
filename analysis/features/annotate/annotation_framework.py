"""
Shared harness for metadata-driven feature annotators.

Annotators own their scoring/classification state, while this module handles
the common disk IO, archive/merge semantics, and per-feature iteration.
"""

from __future__ import annotations

import json
import math
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


def load_feature_json(data_dir: Path, cantor_id: int | str) -> dict[str, Any] | None:
    """Load ``features/{cantor_id}.json`` when it exists."""
    path = data_dir / "features" / f"{cantor_id}.json"
    if not path.is_file():
        return None
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object in feature JSON: {path}")
    return data


class FeatureJsonAnnotator(FeatureAnnotator):
    """Base class for annotators that optionally read per-feature JSON files."""

    def __init__(
        self,
        *,
        data_dir: Path,
        top_k: int = 20,
        feature_json_cache: dict[str, dict[str, Any] | None] | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.top_k = top_k
        self._feature_json_cache = (
            feature_json_cache if feature_json_cache is not None else {}
        )

    def get_feature_json(self, feature: dict[str, Any]) -> dict[str, Any] | None:
        cantor_id = str(feature["cantor_id"])
        if cantor_id not in self._feature_json_cache:
            self._feature_json_cache[cantor_id] = load_feature_json(
                self.data_dir,
                cantor_id,
            )
        return self._feature_json_cache[cantor_id]


def get_examples_quantile(
    feature_json: dict[str, Any] | None,
    quantile_name: str,
    *,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    if not feature_json:
        return []
    for quantile in feature_json.get("examples_quantiles") or []:
        if quantile.get("quantile_name") == quantile_name:
            examples = [
                ex
                for ex in quantile.get("examples") or []
                if isinstance(ex, dict)
            ]
            return examples[:top_k] if top_k is not None else examples
    return []


def get_top_activation_examples(
    feature_json: dict[str, Any] | None,
    *,
    top_k: int | None = None,
    include_domain_quantiles: bool = False,
) -> list[dict[str, Any]]:
    if not feature_json:
        return []

    examples = get_examples_quantile(feature_json, "Top activations")
    if include_domain_quantiles:
        for quantile in feature_json.get("examples_quantiles") or []:
            name = str(quantile.get("quantile_name") or "")
            if name.startswith("Top activations ("):
                examples.extend(
                    ex
                    for ex in quantile.get("examples") or []
                    if isinstance(ex, dict)
                )

    if not examples:
        for quantile in feature_json.get("examples_quantiles") or []:
            name = str(quantile.get("quantile_name") or "")
            if name.startswith("Top activations"):
                examples.extend(
                    ex
                    for ex in quantile.get("examples") or []
                    if isinstance(ex, dict)
                )
                break

    return examples[:top_k] if top_k is not None else examples


def get_highlighted_token(example: dict[str, Any]) -> str | None:
    tokens = example.get("tokens") or []
    if not isinstance(tokens, list):
        return None
    try:
        idx = int(example["train_token_ind"])
    except (KeyError, TypeError, ValueError):
        return None
    if idx < 0 or idx >= len(tokens):
        return None
    return str(tokens[idx])


def get_tokens_acts_list(example: dict[str, Any]) -> list[float]:
    values: list[float] = []
    for value in example.get("tokens_acts_list") or []:
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            values.append(0.0)
    return values


def get_local_token_window(
    example: dict[str, Any],
    *,
    before: int = 3,
    after: int = 3,
) -> list[str]:
    tokens = example.get("tokens") or []
    if not isinstance(tokens, list):
        return []
    try:
        idx = int(example["train_token_ind"])
    except (KeyError, TypeError, ValueError):
        return []
    start = max(0, idx - before)
    end = min(len(tokens), idx + after + 1)
    return [str(token) for token in tokens[start:end]]


def get_local_text_window(
    example: dict[str, Any],
    *,
    before: int = 8,
    after: int = 8,
) -> str:
    return "".join(get_local_token_window(example, before=before, after=after))


def get_logits(
    feature_json: dict[str, Any] | None,
    key: str,
    *,
    top_k: int | None = None,
) -> list[str]:
    if not feature_json:
        return []
    logits = [str(token) for token in feature_json.get(key) or []]
    return logits[:top_k] if top_k is not None else logits


def _positive_distribution_values(distribution: Any) -> list[float]:
    if isinstance(distribution, dict):
        raw_values = distribution.values()
    else:
        raw_values = distribution or []

    values: list[float] = []
    for value in raw_values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric > 0:
            values.append(numeric)
    return values


def _distribution_probabilities(distribution: Any) -> list[float]:
    values = _positive_distribution_values(distribution)
    total = sum(values)
    if total <= 0:
        return []
    return [value / total for value in values]


def concentration_entropy(distribution: Any) -> float:
    """Return normalized entropy in [0, 1] for a distribution."""
    probs = _distribution_probabilities(distribution)
    if len(probs) <= 1:
        return 0.0
    entropy = -sum(prob * math.log(prob) for prob in probs)
    return entropy / math.log(len(probs))


def concentration_herfindahl(distribution: Any) -> float:
    probs = _distribution_probabilities(distribution)
    return sum(prob * prob for prob in probs)


def concentration_max_fraction(distribution: Any) -> float:
    probs = _distribution_probabilities(distribution)
    return max(probs) if probs else 0.0


def concentration_lift_over_rest(distribution: Any) -> float:
    probs = _distribution_probabilities(distribution)
    if len(probs) <= 1:
        return 0.0
    top = max(probs)
    rest_mean = (1.0 - top) / max(1, len(probs) - 1)
    return top / max(rest_mean, 1e-12)


def concentration_scores(distribution: Any, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_entropy": concentration_entropy(distribution),
        f"{prefix}_herfindahl": concentration_herfindahl(distribution),
        f"{prefix}_max_fraction": concentration_max_fraction(distribution),
        f"{prefix}_lift_over_rest": concentration_lift_over_rest(distribution),
    }


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
    for annotator in annotators:
        annotator.setup(metadata)

    hits_by_annotator: dict[str, list[dict[str, Any]]] = {
        annotator.annotation_name: []
        for annotator in annotators
    }

    for feature in tqdm(metadata.get("features") or []):
        cantor_id = str(feature["cantor_id"])
        for annotator in annotators:
            result = annotator.annotate_feature(feature, metadata)
            entry = annotations.get(cantor_id) or {}
            should_update = bool(result.tags) or annotator.has_existing_annotation(entry)
            if not should_update:
                continue

            annotations[cantor_id] = annotator.merge_annotation(entry, result)
            if result.tags:
                hits_by_annotator[annotator.annotation_name].append(
                    annotator.make_hit(feature, result),
                )

    for annotator in annotators:
        hits_by_annotator[annotator.annotation_name].sort(
            key=annotator.sort_key,
            reverse=True,
        )
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
