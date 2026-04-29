"""
Annotate features that concentrate around the assistant response boundary.

This is a lightweight post-processing step for ``collect_feature_activations.py``
outputs. It reads ``feature_metadata.json`` only, scores existing feature stats,
and writes a persistent ``feature_annotations.json`` file consumable by the local
dashboard. By default, an existing annotations file is moved into an ``archive/``
subdirectory before fresh annotations are written. Pass ``--merge`` to merge new
tags into an existing annotations file instead.

Example:
    python -m analysis.features.annotate.annotate_assistant_response_features \
        --data_dir /path/to/feature_run
"""

from __future__ import annotations

import argparse
from typing import Any

from helpers.log import setup_logging

from analysis.features.annotate.annotation_framework import (
    FeatureAnnotationResult,
    FeatureAnnotator,
    add_common_annotation_args,
    annotate_features as run_feature_annotation,
    run_annotation,
)

ASSISTANT_MARKER_REGION = "assistant_marker"
ANSWER_REGION = "answer"


def _region_fraction(feature: dict[str, Any], region: str) -> float:
    return float((feature.get("region_fraction") or {}).get(region) or 0.0)


def _region_density(feature: dict[str, Any], region: str) -> float:
    return float((feature.get("region_density") or {}).get(region) or 0.0)


class AssistantResponseFeatureAnnotator(FeatureAnnotator):
    """Annotator for features localized to assistant marker or answer regions."""

    annotation_name = "assistant_response"

    def __init__(
        self,
        *,
        assistant_fraction_threshold: float = 0.35,
        marker_fraction_threshold: float = 0.20,
        min_marker_activations: int = 5,
        density_lift_threshold: float = 2.0,
    ) -> None:
        self.assistant_fraction_threshold = assistant_fraction_threshold
        self.marker_fraction_threshold = marker_fraction_threshold
        self.min_marker_activations = min_marker_activations
        self.density_lift_threshold = density_lift_threshold
        self.tokens_per_region: dict[str, int] = {}

    def setup(self, metadata: dict[str, Any]) -> None:
        self.tokens_per_region = metadata.get("tokens_per_region") or {}

    def score_feature(self, feature: dict[str, Any]) -> dict[str, float]:
        activation_count = int(feature.get("activation_count") or 0)
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

        marker_tokens = int(self.tokens_per_region.get(ASSISTANT_MARKER_REGION) or 0)
        has_marker_region = marker_tokens > 0
        marker_count = marker_fraction * activation_count
        answer_count = answer_fraction * activation_count

        return {
            "activation_count": float(activation_count),
            "assistant_fraction": assistant_fraction,
            "assistant_marker_fraction": marker_fraction,
            "answer_fraction": answer_fraction,
            "assistant_marker_count": marker_count,
            "answer_count": answer_count,
            "assistant_density": assistant_density,
            "assistant_marker_density": marker_density,
            "answer_density": answer_density,
            "density_lift": density_lift,
            "has_marker_region": 1.0 if has_marker_region else 0.0,
        }

    def classify_feature(
        self,
        feature: dict[str, Any],
        scores: dict[str, float],
    ) -> list[str]:
        tags: list[str] = []
        if (
            scores["assistant_fraction"] >= self.assistant_fraction_threshold
            and scores["density_lift"] >= self.density_lift_threshold
        ):
            tags.append("assistant_response")

        if (
            scores["assistant_marker_fraction"] >= self.marker_fraction_threshold
            and scores["assistant_marker_count"] >= self.min_marker_activations
            and scores["has_marker_region"] > 0
        ):
            tags.extend(["assistant_response", "assistant_token"])

        if (
            scores["answer_fraction"] >= self.assistant_fraction_threshold
            and "assistant_token" not in tags
        ):
            tags.extend(["assistant_response", "near_assistant"])

        return sorted(set(tags))

    def annotate_feature(
        self,
        feature: dict[str, Any],
        metadata: dict[str, Any],
    ) -> FeatureAnnotationResult:
        scores = self.score_feature(feature)
        tags = self.classify_feature(feature, scores)
        return FeatureAnnotationResult(tags=tags, scores=scores, hit_data=scores)

    def sort_key(self, hit: dict[str, Any]) -> tuple[float, float, float]:
        return (
            hit["assistant_marker_fraction"],
            hit["assistant_fraction"],
            hit["density_lift"],
        )

    def format_hit(self, hit: dict[str, Any]) -> str:
        tag_text = ",".join(hit["tags"])
        return (
            f"L{hit['layer']} F{hit['feature']} cantor={hit['cantor_id']} "
            f"tags={tag_text} assistant_frac={hit['assistant_fraction']:.3f} "
            f"marker_frac={hit['assistant_marker_fraction']:.3f} "
            f"marker_count={hit['assistant_marker_count']:.0f} "
            f"density_lift={hit['density_lift']:.2f}"
        )


def score_feature(
    feature: dict[str, Any],
    tokens_per_region: dict[str, int],
) -> dict[str, float]:
    """Return assistant-boundary scores from metadata for one feature."""
    annotator = AssistantResponseFeatureAnnotator()
    annotator.tokens_per_region = tokens_per_region
    return annotator.score_feature(feature)


def classify_feature(
    feature: dict[str, Any],
    scores: dict[str, float],
    *,
    assistant_fraction_threshold: float,
    marker_fraction_threshold: float,
    min_marker_activations: int,
    density_lift_threshold: float,
) -> list[str]:
    """Classify one feature into annotation tags."""
    annotator = AssistantResponseFeatureAnnotator(
        assistant_fraction_threshold=assistant_fraction_threshold,
        marker_fraction_threshold=marker_fraction_threshold,
        min_marker_activations=min_marker_activations,
        density_lift_threshold=density_lift_threshold,
    )
    return annotator.classify_feature(feature, scores)


def annotate_features(
    metadata: dict[str, Any],
    annotations: dict[str, Any],
    *,
    assistant_fraction_threshold: float,
    marker_fraction_threshold: float,
    min_marker_activations: int,
    density_lift_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    annotator = AssistantResponseFeatureAnnotator(
        assistant_fraction_threshold=assistant_fraction_threshold,
        marker_fraction_threshold=marker_fraction_threshold,
        min_marker_activations=min_marker_activations,
        density_lift_threshold=density_lift_threshold,
    )
    return run_feature_annotation(metadata, annotations, annotator)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find and persist annotations for assistant-response features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_annotation_args(parser)
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
        "--min_marker_activations",
        type=int,
        default=5,
        help=(
            "Minimum number of assistant-marker activations required for the "
            "assistant_token tag. This avoids tagging one-off marker hits in "
            "small feature collections."
        ),
    )
    parser.add_argument(
        "--density_lift_threshold",
        type=float,
        default=2.0,
        help="Minimum assistant-region density lift over non-assistant regions",
    )
    return parser


def main() -> None:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args()
    annotator = AssistantResponseFeatureAnnotator(
        assistant_fraction_threshold=args.assistant_fraction_threshold,
        marker_fraction_threshold=args.marker_fraction_threshold,
        min_marker_activations=args.min_marker_activations,
        density_lift_threshold=args.density_lift_threshold,
    )
    run_annotation(
        data_dir=args.data_dir,
        annotations_file=args.annotations_file,
        merge=args.merge,
        top_k=args.top_k,
        annotator=annotator,
    )


if __name__ == "__main__":
    main()
