"""
Annotate metadata shifts between two comparable feature collection runs.

The target run receives the annotations. Only features with matching
``cantor_id`` entries in source and target metadata are compared.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from helpers.log import setup_logging

from analysis.features.annotate.annotation_framework import (
    FeatureAnnotationResult,
    FeatureAnnotator,
    concentration_max_fraction,
    load_json,
    run_annotation,
)

CONTRASTIVE_TAGS = frozenset(
    {
        "it_amplified",
        "base_amplified",
        "checkpoint_emergent",
        "checkpoint_suppressed",
        "stable_feature",
    }
)

EPSILON = 1e-12


def _feature_by_cantor_id(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(feature["cantor_id"]): feature
        for feature in metadata.get("features") or []
        if "cantor_id" in feature
    }


def _activation_freq(feature: dict[str, Any]) -> float:
    return float(
        feature.get("activation_freq")
        or feature.get("activation_frequency")
        or 0.0
    )


def _activation_count(feature: dict[str, Any]) -> float:
    return float(feature.get("activation_count") or 0.0)


def _normalize_distribution(distribution: Any) -> dict[str, float]:
    if isinstance(distribution, dict):
        items = distribution.items()
    elif isinstance(distribution, list):
        items = enumerate(distribution)
    else:
        items = []

    values: dict[str, float] = {}
    for key, value in items:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric > 0:
            values[str(key)] = numeric

    total = sum(values.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in values.items()}


def _distribution_shift(source: Any, target: Any) -> float:
    source_dist = _normalize_distribution(source)
    target_dist = _normalize_distribution(target)
    keys = set(source_dist) | set(target_dist)
    if not keys:
        return 0.0
    return 0.5 * sum(
        abs(source_dist.get(key, 0.0) - target_dist.get(key, 0.0))
        for key in keys
    )


class ContrastiveRunAnnotator(FeatureAnnotator):
    """Annotate activation and metadata shifts across two runs."""

    annotation_name = "contrastive_run"
    owned_tags = CONTRASTIVE_TAGS

    def __init__(
        self,
        *,
        source_metadata: dict[str, Any],
        amplified_ratio: float = 2.0,
        emergent_ratio: float = 5.0,
        rare_frequency: float = 1e-12,
        min_frequency: float = 1e-8,
        stable_ratio_min: float = 0.67,
        stable_ratio_max: float = 1.5,
        shift_threshold: float = 0.25,
    ) -> None:
        self.source_features = _feature_by_cantor_id(source_metadata)
        self.amplified_ratio = amplified_ratio
        self.emergent_ratio = emergent_ratio
        self.rare_frequency = rare_frequency
        self.min_frequency = min_frequency
        self.stable_ratio_min = stable_ratio_min
        self.stable_ratio_max = stable_ratio_max
        self.shift_threshold = shift_threshold

    def score_feature(
        self,
        source_feature: dict[str, Any],
        target_feature: dict[str, Any],
    ) -> dict[str, float]:
        source_freq = _activation_freq(source_feature)
        target_freq = _activation_freq(target_feature)
        ratio = (target_freq + EPSILON) / (source_freq + EPSILON)
        inverse_ratio = (source_freq + EPSILON) / (target_freq + EPSILON)
        region_shift = _distribution_shift(
            source_feature.get("region_fraction") or {},
            target_feature.get("region_fraction") or {},
        )
        domain_shift = _distribution_shift(
            source_feature.get("domain_fraction") or {},
            target_feature.get("domain_fraction") or {},
        )

        return {
            "source_activation_freq": source_freq,
            "target_activation_freq": target_freq,
            "activation_frequency_ratio": ratio,
            "inverse_activation_frequency_ratio": inverse_ratio,
            "source_activation_count": _activation_count(source_feature),
            "target_activation_count": _activation_count(target_feature),
            "region_distribution_shift": region_shift,
            "domain_distribution_shift": domain_shift,
            "source_region_max_fraction": concentration_max_fraction(
                source_feature.get("region_fraction") or {},
            ),
            "target_region_max_fraction": concentration_max_fraction(
                target_feature.get("region_fraction") or {},
            ),
            "source_domain_max_fraction": concentration_max_fraction(
                source_feature.get("domain_fraction") or {},
            ),
            "target_domain_max_fraction": concentration_max_fraction(
                target_feature.get("domain_fraction") or {},
            ),
        }

    def classify_feature(self, scores: dict[str, float]) -> list[str]:
        tags: list[str] = []
        source_freq = scores["source_activation_freq"]
        target_freq = scores["target_activation_freq"]
        ratio = scores["activation_frequency_ratio"]
        inverse_ratio = scores["inverse_activation_frequency_ratio"]
        max_shift = max(
            scores["region_distribution_shift"],
            scores["domain_distribution_shift"],
        )

        if (
            source_freq <= self.rare_frequency
            and target_freq >= self.min_frequency
            and ratio >= self.emergent_ratio
        ):
            tags.append("checkpoint_emergent")
        elif (
            target_freq <= self.rare_frequency
            and source_freq >= self.min_frequency
            and inverse_ratio >= self.emergent_ratio
        ):
            tags.append("checkpoint_suppressed")
        elif target_freq >= self.min_frequency and ratio >= self.amplified_ratio:
            tags.append("it_amplified")
        elif source_freq >= self.min_frequency and inverse_ratio >= self.amplified_ratio:
            tags.append("base_amplified")

        if (
            self.stable_ratio_min <= ratio <= self.stable_ratio_max
            and max_shift <= self.shift_threshold
        ):
            tags.append("stable_feature")

        return sorted(set(tags))

    def annotate_feature(
        self,
        feature: dict[str, Any],
        metadata: dict[str, Any],
    ) -> FeatureAnnotationResult:
        source_feature = self.source_features.get(str(feature["cantor_id"]))
        if source_feature is None:
            return FeatureAnnotationResult(tags=[])
        scores = self.score_feature(source_feature, feature)
        tags = self.classify_feature(scores)
        return FeatureAnnotationResult(tags=tags, scores=scores, hit_data=scores)

    def sort_key(self, hit: dict[str, Any]) -> tuple[float, float, float]:
        ratio = max(
            hit.get("activation_frequency_ratio", 0.0),
            hit.get("inverse_activation_frequency_ratio", 0.0),
        )
        shift = max(
            hit.get("region_distribution_shift", 0.0),
            hit.get("domain_distribution_shift", 0.0),
        )
        return (ratio, shift, hit.get("target_activation_freq", 0.0))

    def format_hit(self, hit: dict[str, Any]) -> str:
        tag_text = ",".join(hit["tags"])
        return (
            f"L{hit['layer']} F{hit['feature']} cantor={hit['cantor_id']} "
            f"tags={tag_text} ratio={hit.get('activation_frequency_ratio', 0.0):.2f} "
            f"region_shift={hit.get('region_distribution_shift', 0.0):.2f} "
            f"domain_shift={hit.get('domain_distribution_shift', 0.0):.2f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Annotate contrastive metadata shifts between comparable runs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source_data_dir", type=Path, required=True)
    parser.add_argument("--target_data_dir", type=Path, required=True)
    parser.add_argument(
        "--annotations_file",
        type=Path,
        default=None,
        help=(
            "JSON file to write. Defaults to "
            "<target_data_dir>/feature_annotations.json."
        ),
    )
    parser.add_argument(
        "--replace_all",
        action="store_true",
        help=(
            "Archive the existing annotations file and start from an empty file. "
            "By default, only contrastive_run tags/scores are updated."
        ),
    )
    parser.add_argument("--top_k", type=int, default=25, help="Number of hits to log")
    parser.add_argument("--amplified_ratio", type=float, default=2.0)
    parser.add_argument("--emergent_ratio", type=float, default=5.0)
    parser.add_argument("--rare_frequency", type=float, default=1e-12)
    parser.add_argument("--min_frequency", type=float, default=1e-8)
    parser.add_argument("--stable_ratio_min", type=float, default=0.67)
    parser.add_argument("--stable_ratio_max", type=float, default=1.5)
    parser.add_argument("--shift_threshold", type=float, default=0.25)
    return parser


def main() -> None:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args()
    source_metadata = load_json(args.source_data_dir / "feature_metadata.json")
    annotator = ContrastiveRunAnnotator(
        source_metadata=source_metadata,
        amplified_ratio=args.amplified_ratio,
        emergent_ratio=args.emergent_ratio,
        rare_frequency=args.rare_frequency,
        min_frequency=args.min_frequency,
        stable_ratio_min=args.stable_ratio_min,
        stable_ratio_max=args.stable_ratio_max,
        shift_threshold=args.shift_threshold,
    )
    run_annotation(
        data_dir=args.target_data_dir,
        annotations_file=args.annotations_file,
        annotator=annotator,
        replace_all=args.replace_all,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
