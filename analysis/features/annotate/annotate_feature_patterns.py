"""
Annotate high-signal feature patterns from existing collection outputs.

This module reads ``feature_metadata.json`` plus optional
``features/{cantor_id}.json`` files and writes dashboard-ready annotations. The
annotators update only their own ``auto_tags`` / ``auto_scores`` entries.
"""

from __future__ import annotations

import argparse
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from helpers.log import setup_logging

from analysis.features.annotate.annotation_framework import (
    FeatureAnnotationResult,
    FeatureJsonAnnotator,
    add_common_annotation_args,
    concentration_entropy,
    concentration_scores,
    get_highlighted_token,
    get_logits,
    get_local_text_window,
    get_local_token_window,
    get_tokens_acts_list,
    get_top_activation_examples,
    run_annotation,
)

EPSILON = 1e-12

LOGIT_TAGS = frozenset(
    {
        "promotes_numbers",
        "promotes_newline",
        "promotes_code",
        "promotes_uncertainty",
        "promotes_refusal",
        "promotes_special_tokens",
        "promotes_answer_tokens",
        "suppresses_answer_tokens",
    }
)

ACTIVATION_SHAPE_TAGS = frozenset(
    {
        "single_token_spike",
        "sustained_context",
        "ramp_up",
        "ramp_down",
        "multi_token_phrase",
    }
)

FEATURE_SPECIFICITY_TAGS = frozenset(
    {
        "high_precision",
        "broad_context",
        "domain_specialist",
        "region_specialist",
        "token_specialist",
    }
)

TOKEN_SURFACE_TAGS = frozenset(
    {
        "digit_feature",
        "operator_feature",
        "punctuation_feature",
        "newline_feature",
        "capitalized_token",
        "whitespace_token",
        "quote_token",
    }
)

CROSS_EXAMPLE_TAGS = frozenset(
    {
        "consistent_token",
        "consistent_phrase",
        "consistent_context",
        "incoherent_feature",
    }
)

REASONING_MOVE_TAGS = frozenset(
    {
        "self_correction",
        "uncertainty",
        "planning",
        "verification",
        "answer_commitment",
        "backtracking",
    }
)


def _safe_fraction(count: float, total: float) -> float:
    return count / total if total > 0 else 0.0


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _nonzero_count(distribution: Any) -> int:
    values = distribution.values() if isinstance(distribution, dict) else distribution or []
    count = 0
    for value in values:
        try:
            if float(value) > 0:
                count += 1
        except (TypeError, ValueError):
            continue
    return count


def _counter_scores(counter: Counter[str], prefix: str) -> dict[str, float]:
    total = sum(counter.values())
    scores = concentration_scores(counter, prefix)
    scores[f"{prefix}_count"] = float(total)
    scores[f"{prefix}_unique"] = float(len(counter))
    scores[f"{prefix}_diversity"] = _safe_fraction(len(counter), total)
    return scores


def _is_special_token(token: str) -> bool:
    stripped = token.strip()
    if not stripped:
        return False
    return (
        bool(re.fullmatch(r"<[^>]+>", stripped))
        or "<|" in stripped
        or "|>" in stripped
        or bool(re.fullmatch(r"\[[A-Z_]+\]", stripped))
    )


def _matches_logit_family(token: str, family: str) -> bool:
    stripped = token.strip()
    lower = stripped.lower()

    if family == "numbers":
        return any(char.isdigit() for char in stripped)
    if family == "newline":
        return "\n" in token or "\\n" in token
    if family == "code":
        code_words = {
            "def",
            "return",
            "class",
            "import",
            "from",
            "for",
            "while",
            "if",
            "else",
            "elif",
            "print",
            "self",
        }
        code_symbols = {
            "{",
            "}",
            "(",
            ")",
            "[",
            "]",
            ";",
            "```",
            "->",
            "=>",
            "==",
            "!=",
            "<=",
            ">=",
            "::",
        }
        return lower in code_words or any(symbol in stripped for symbol in code_symbols)
    if family == "uncertainty":
        uncertainty_words = {
            "maybe",
            "probably",
            "perhaps",
            "possibly",
            "unsure",
            "uncertain",
            "might",
            "could",
        }
        return lower in uncertainty_words or "not sure" in lower
    if family == "refusal":
        refusal_parts = (
            "sorry",
            "cannot",
            "can't",
            "unable",
            "refuse",
            "policy",
            "not able",
            "as an ai",
        )
        return any(part in lower for part in refusal_parts)
    if family == "special_tokens":
        return _is_special_token(token)
    if family == "answer_tokens":
        answer_parts = (
            "answer",
            "therefore",
            "thus",
            "final",
            "result",
            "conclusion",
            "boxed",
        )
        return any(part in lower for part in answer_parts)
    return False


def _count_logit_families(tokens: list[str]) -> Counter[str]:
    families = Counter()
    for token in tokens:
        for family in (
            "numbers",
            "newline",
            "code",
            "uncertainty",
            "refusal",
            "special_tokens",
            "answer_tokens",
        ):
            if _matches_logit_family(token, family):
                families[family] += 1
    return families


def _family_hit(
    scores: dict[str, float],
    key_prefix: str,
    *,
    min_hits: int = 2,
    min_fraction: float = 0.15,
) -> bool:
    return (
        scores.get(f"{key_prefix}_count", 0.0) >= min_hits
        or scores.get(f"{key_prefix}_fraction", 0.0) >= min_fraction
    )


def _logit_family_scores(tokens: list[str], prefix: str) -> dict[str, float]:
    counts = _count_logit_families(tokens)
    scores: dict[str, float] = {f"{prefix}_logit_count": float(len(tokens))}
    for family in (
        "numbers",
        "newline",
        "code",
        "uncertainty",
        "refusal",
        "special_tokens",
        "answer_tokens",
    ):
        count = float(counts.get(family, 0))
        scores[f"{prefix}_{family}_count"] = count
        scores[f"{prefix}_{family}_fraction"] = _safe_fraction(count, len(tokens))
    return scores


FeatureJsonCache = dict[str, dict[str, Any] | None]


class LogitEffectFeatureAnnotator(FeatureJsonAnnotator):
    """Annotate promoted and suppressed token families from logit lens output."""

    annotation_name = "logit_effect"
    owned_tags = LOGIT_TAGS

    def __init__(
        self,
        *,
        data_dir: Path,
        top_k: int = 20,
        feature_json_cache: FeatureJsonCache | None = None,
    ) -> None:
        super().__init__(
            data_dir=data_dir,
            top_k=top_k,
            feature_json_cache=feature_json_cache,
        )

    def score_feature_json(self, feature_json: dict[str, Any]) -> dict[str, float]:
        top_logits = get_logits(feature_json, "top_logits", top_k=self.top_k)
        bottom_logits = get_logits(feature_json, "bottom_logits", top_k=self.top_k)
        return {
            **_logit_family_scores(top_logits, "top"),
            **_logit_family_scores(bottom_logits, "bottom"),
        }

    def classify_feature(self, scores: dict[str, float]) -> list[str]:
        tags: list[str] = []
        if _family_hit(scores, "top_numbers"):
            tags.append("promotes_numbers")
        if _family_hit(scores, "top_newline", min_hits=1, min_fraction=0.08):
            tags.append("promotes_newline")
        if _family_hit(scores, "top_code"):
            tags.append("promotes_code")
        if _family_hit(scores, "top_uncertainty", min_hits=1, min_fraction=0.08):
            tags.append("promotes_uncertainty")
        if _family_hit(scores, "top_refusal", min_hits=1, min_fraction=0.08):
            tags.append("promotes_refusal")
        if _family_hit(scores, "top_special_tokens", min_hits=1, min_fraction=0.08):
            tags.append("promotes_special_tokens")
        if _family_hit(scores, "top_answer_tokens", min_hits=1, min_fraction=0.08):
            tags.append("promotes_answer_tokens")
        if _family_hit(scores, "bottom_answer_tokens", min_hits=1, min_fraction=0.08):
            tags.append("suppresses_answer_tokens")
        return sorted(set(tags))

    def annotate_feature(
        self,
        feature: dict[str, Any],
        metadata: dict[str, Any],
    ) -> FeatureAnnotationResult:
        feature_json = self.get_feature_json(feature)
        if feature_json is None:
            return FeatureAnnotationResult(tags=[])
        scores = self.score_feature_json(feature_json)
        tags = self.classify_feature(scores)
        return FeatureAnnotationResult(tags=tags, scores=scores, hit_data=scores)

    def sort_key(self, hit: dict[str, Any]) -> tuple[float, float, float]:
        return (
            hit.get("top_answer_tokens_fraction", 0.0),
            hit.get("top_code_fraction", 0.0),
            hit.get("top_numbers_fraction", 0.0),
        )

    def format_hit(self, hit: dict[str, Any]) -> str:
        tag_text = ",".join(hit["tags"])
        return (
            f"L{hit['layer']} F{hit['feature']} cantor={hit['cantor_id']} "
            f"tags={tag_text} answer={hit.get('top_answer_tokens_fraction', 0.0):.2f} "
            f"code={hit.get('top_code_fraction', 0.0):.2f} "
            f"numbers={hit.get('top_numbers_fraction', 0.0):.2f}"
        )


def _linear_slope(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    x_mean = (len(values) - 1) / 2.0
    y_mean = _average(values)
    denom = sum((idx - x_mean) ** 2 for idx in range(len(values)))
    if denom <= 0:
        return 0.0
    numer = sum((idx - x_mean) * (value - y_mean) for idx, value in enumerate(values))
    return numer / denom


def _contiguous_width(values: list[float], idx: int, threshold: float) -> int:
    if idx < 0 or idx >= len(values):
        return 0
    left = idx
    while left > 0 and values[left - 1] >= threshold:
        left -= 1
    right = idx
    while right + 1 < len(values) and values[right + 1] >= threshold:
        right += 1
    return right - left + 1 if values[idx] >= threshold else 0


class ActivationShapeFeatureAnnotator(FeatureJsonAnnotator):
    """Annotate local activation traces around top examples."""

    annotation_name = "activation_shape"
    owned_tags = ACTIVATION_SHAPE_TAGS

    def __init__(
        self,
        *,
        data_dir: Path,
        top_k: int = 20,
        feature_json_cache: FeatureJsonCache | None = None,
    ) -> None:
        super().__init__(
            data_dir=data_dir,
            top_k=top_k,
            feature_json_cache=feature_json_cache,
        )

    def _score_example(self, example: dict[str, Any]) -> dict[str, float] | None:
        acts = [max(0.0, value) for value in get_tokens_acts_list(example)]
        try:
            idx = int(example["train_token_ind"])
        except (KeyError, TypeError, ValueError):
            return None
        if idx < 0 or idx >= len(acts) or not acts:
            return None

        peak = max(acts)
        center = acts[idx]
        total_mass = sum(acts)
        if peak <= 0 or total_mass <= 0:
            return None

        left_values = acts[max(0, idx - 5) : idx + 1]
        right_values = acts[idx : min(len(acts), idx + 6)]
        scale = max(peak, EPSILON)
        above_30 = sum(1 for value in acts if value >= 0.3 * peak)

        return {
            "peak_width_50": float(_contiguous_width(acts, idx, 0.5 * peak)),
            "peak_width_30": float(_contiguous_width(acts, idx, 0.3 * peak)),
            "highlighted_mass": center / total_mass,
            "left_slope": _linear_slope(left_values) / scale,
            "right_slope": _linear_slope(right_values) / scale,
            "activation_entropy": concentration_entropy(acts),
            "above_30_fraction": _safe_fraction(above_30, len(acts)),
        }

    def score_feature_json(self, feature_json: dict[str, Any]) -> dict[str, float]:
        examples = get_top_activation_examples(feature_json, top_k=self.top_k)
        scored = [
            score
            for example in examples
            if (score := self._score_example(example)) is not None
        ]
        if not scored:
            return {"examples_scored": 0.0}
        keys = scored[0].keys()
        return {
            "examples_scored": float(len(scored)),
            **{key: _average([score[key] for score in scored]) for key in keys},
        }

    def classify_feature(self, scores: dict[str, float]) -> list[str]:
        if scores.get("examples_scored", 0.0) <= 0:
            return []

        tags: list[str] = []
        if (
            scores["peak_width_50"] <= 1.5
            and scores["highlighted_mass"] >= 0.45
            and scores["activation_entropy"] <= 0.55
        ):
            tags.append("single_token_spike")
        if scores["above_30_fraction"] >= 0.35 and scores["activation_entropy"] >= 0.60:
            tags.append("sustained_context")
        if scores["peak_width_30"] >= 3.0 and scores["highlighted_mass"] < 0.55:
            tags.append("multi_token_phrase")
        if scores["left_slope"] >= 0.08 and scores["peak_width_30"] >= 3.0:
            tags.append("ramp_up")
        if scores["right_slope"] <= -0.08 and scores["peak_width_30"] >= 3.0:
            tags.append("ramp_down")
        return sorted(set(tags))

    def annotate_feature(
        self,
        feature: dict[str, Any],
        metadata: dict[str, Any],
    ) -> FeatureAnnotationResult:
        feature_json = self.get_feature_json(feature)
        if feature_json is None:
            return FeatureAnnotationResult(tags=[])
        scores = self.score_feature_json(feature_json)
        tags = self.classify_feature(scores)
        return FeatureAnnotationResult(tags=tags, scores=scores, hit_data=scores)

    def sort_key(self, hit: dict[str, Any]) -> tuple[float, float]:
        return (hit.get("highlighted_mass", 0.0), hit.get("activation_entropy", 0.0))


class FeatureSpecificityAnnotator(FeatureJsonAnnotator):
    """Annotate concentration across domains, regions, and highlighted tokens."""

    annotation_name = "feature_specificity"
    owned_tags = FEATURE_SPECIFICITY_TAGS

    def __init__(
        self,
        *,
        data_dir: Path,
        top_k: int = 20,
        feature_json_cache: FeatureJsonCache | None = None,
    ) -> None:
        super().__init__(
            data_dir=data_dir,
            top_k=top_k,
            feature_json_cache=feature_json_cache,
        )

    def score_feature(
        self,
        feature: dict[str, Any],
        feature_json: dict[str, Any] | None,
    ) -> dict[str, float]:
        domain_fraction = feature.get("domain_fraction") or {}
        region_fraction = feature.get("region_fraction") or {}
        scores: dict[str, float] = {
            "activation_count": float(feature.get("activation_count") or 0.0),
            "activation_freq": float(feature.get("activation_freq") or 0.0),
            "domain_nonzero": float(_nonzero_count(domain_fraction)),
            "region_nonzero": float(_nonzero_count(region_fraction)),
            **concentration_scores(domain_fraction, "domain"),
            **concentration_scores(region_fraction, "region"),
        }

        examples = get_top_activation_examples(feature_json, top_k=self.top_k)
        token_counter = Counter(
            token
            for example in examples
            if (token := get_highlighted_token(example)) is not None
        )
        scores.update(_counter_scores(token_counter, "highlighted_token"))
        return scores

    def classify_feature(self, scores: dict[str, float]) -> list[str]:
        tags: list[str] = []
        domain_signal = (
            scores["domain_nonzero"] >= 2
            and (
                scores["domain_max_fraction"] >= 0.70
                or scores["domain_lift_over_rest"] >= 3.0
            )
        )
        region_signal = (
            scores["region_nonzero"] >= 2
            and (
                scores["region_max_fraction"] >= 0.65
                or scores["region_lift_over_rest"] >= 3.0
            )
        )
        token_signal = (
            scores["highlighted_token_count"] >= 3
            and (
                scores["highlighted_token_max_fraction"] >= 0.50
                or scores["highlighted_token_herfindahl"] >= 0.35
            )
        )

        if domain_signal:
            tags.append("domain_specialist")
        if region_signal:
            tags.append("region_specialist")
        if token_signal:
            tags.append("token_specialist")
        if domain_signal or region_signal or token_signal:
            tags.append("high_precision")
        if (
            scores["region_nonzero"] >= 3
            and scores["region_entropy"] >= 0.80
            and scores["region_max_fraction"] <= 0.45
            and (
                scores["domain_nonzero"] <= 1
                or scores["domain_entropy"] >= 0.70
            )
        ):
            tags.append("broad_context")
        return sorted(set(tags))

    def annotate_feature(
        self,
        feature: dict[str, Any],
        metadata: dict[str, Any],
    ) -> FeatureAnnotationResult:
        scores = self.score_feature(feature, self.get_feature_json(feature))
        tags = self.classify_feature(scores)
        return FeatureAnnotationResult(tags=tags, scores=scores, hit_data=scores)

    def sort_key(self, hit: dict[str, Any]) -> tuple[float, float, float]:
        return (
            hit.get("highlighted_token_max_fraction", 0.0),
            hit.get("region_max_fraction", 0.0),
            hit.get("domain_max_fraction", 0.0),
        )


def _surface_classes(token: str) -> set[str]:
    classes: set[str] = set()
    stripped = token.strip()
    if any(char.isdigit() for char in stripped):
        classes.add("digit")
    if "\n" in token or "\\n" in token:
        classes.add("newline")
    if token and token.isspace():
        classes.add("whitespace")
    if any(char in {"'", '"'} for char in token):
        classes.add("quote")
    if stripped and stripped[0].isupper():
        classes.add("capitalized")

    operator_chars = set("+-*/=%<>^|&~")
    if stripped and any(char in operator_chars for char in stripped):
        classes.add("operator")

    punctuation_chars = set(string.punctuation)
    if stripped and any(char in punctuation_chars for char in stripped):
        classes.add("punctuation")
    return classes


class TokenSurfaceAnnotator(FeatureJsonAnnotator):
    """Annotate surface forms of highlighted top-activation tokens."""

    annotation_name = "token_surface"
    owned_tags = TOKEN_SURFACE_TAGS

    def __init__(
        self,
        *,
        data_dir: Path,
        top_k: int = 20,
        feature_json_cache: FeatureJsonCache | None = None,
    ) -> None:
        super().__init__(
            data_dir=data_dir,
            top_k=top_k,
            feature_json_cache=feature_json_cache,
        )

    def score_feature_json(self, feature_json: dict[str, Any]) -> dict[str, float]:
        examples = get_top_activation_examples(feature_json, top_k=self.top_k)
        tokens = [
            token
            for example in examples
            if (token := get_highlighted_token(example)) is not None
        ]
        class_counts = Counter()
        for token in tokens:
            for class_name in _surface_classes(token):
                class_counts[class_name] += 1

        scores = {
            "examples_scored": float(len(tokens)),
            **_counter_scores(Counter(tokens), "highlighted_token"),
        }
        for class_name in (
            "digit",
            "operator",
            "punctuation",
            "newline",
            "capitalized",
            "whitespace",
            "quote",
        ):
            count = float(class_counts.get(class_name, 0))
            scores[f"{class_name}_count"] = count
            scores[f"{class_name}_fraction"] = _safe_fraction(count, len(tokens))
        return scores

    def classify_feature(self, scores: dict[str, float]) -> list[str]:
        if scores.get("examples_scored", 0.0) <= 0:
            return []

        tag_specs = {
            "digit_feature": "digit",
            "operator_feature": "operator",
            "punctuation_feature": "punctuation",
            "newline_feature": "newline",
            "capitalized_token": "capitalized",
            "whitespace_token": "whitespace",
            "quote_token": "quote",
        }
        tags: list[str] = []
        for tag, class_name in tag_specs.items():
            min_hits = 1 if class_name in {"newline", "whitespace", "quote"} else 2
            if _family_hit(
                scores,
                class_name,
                min_hits=min_hits,
                min_fraction=0.40,
            ):
                tags.append(tag)
        return sorted(set(tags))

    def annotate_feature(
        self,
        feature: dict[str, Any],
        metadata: dict[str, Any],
    ) -> FeatureAnnotationResult:
        feature_json = self.get_feature_json(feature)
        if feature_json is None:
            return FeatureAnnotationResult(tags=[])
        scores = self.score_feature_json(feature_json)
        tags = self.classify_feature(scores)
        return FeatureAnnotationResult(tags=tags, scores=scores, hit_data=scores)

    def sort_key(self, hit: dict[str, Any]) -> tuple[float, float]:
        return (
            hit.get("highlighted_token_max_fraction", 0.0),
            hit.get("highlighted_token_count", 0.0),
        )


def _normalized_phrase(example: dict[str, Any]) -> str | None:
    tokens = get_local_token_window(example, before=1, after=1)
    text = re.sub(r"\s+", " ", "".join(tokens).strip().lower())
    return text or None


class CrossExampleConsistencyAnnotator(FeatureJsonAnnotator):
    """Annotate agreement across top activating examples."""

    annotation_name = "cross_example_consistency"
    owned_tags = CROSS_EXAMPLE_TAGS

    def __init__(
        self,
        *,
        data_dir: Path,
        top_k: int = 20,
        feature_json_cache: FeatureJsonCache | None = None,
    ) -> None:
        super().__init__(
            data_dir=data_dir,
            top_k=top_k,
            feature_json_cache=feature_json_cache,
        )

    def score_feature(
        self,
        feature: dict[str, Any],
        feature_json: dict[str, Any],
    ) -> dict[str, float]:
        examples = get_top_activation_examples(feature_json, top_k=self.top_k)
        token_counter = Counter(
            token
            for example in examples
            if (token := get_highlighted_token(example)) is not None
        )
        phrase_counter = Counter(
            phrase
            for example in examples
            if (phrase := _normalized_phrase(example)) is not None
        )
        region_fraction = feature.get("region_fraction") or {}
        top_logits = get_logits(feature_json, "top_logits", top_k=self.top_k)
        logit_family_counts = _count_logit_families(top_logits)
        dominant_logit_count = max(logit_family_counts.values()) if logit_family_counts else 0

        scores: dict[str, float] = {
            "examples_scored": float(len(examples)),
            "region_nonzero": float(_nonzero_count(region_fraction)),
            "dominant_logit_family_fraction": _safe_fraction(
                dominant_logit_count,
                len(top_logits),
            ),
            **_counter_scores(token_counter, "highlighted_token"),
            **_counter_scores(phrase_counter, "local_phrase"),
            **concentration_scores(region_fraction, "region"),
        }
        return scores

    def classify_feature(self, scores: dict[str, float]) -> list[str]:
        examples_scored = scores.get("examples_scored", 0.0)
        if examples_scored <= 0:
            return []

        tags: list[str] = []
        if examples_scored >= 3 and scores["highlighted_token_max_fraction"] >= 0.60:
            tags.append("consistent_token")
        if examples_scored >= 3 and scores["local_phrase_max_fraction"] >= 0.50:
            tags.append("consistent_phrase")
        if (
            scores["region_nonzero"] > 0
            and scores["region_max_fraction"] >= 0.65
        ) or scores["local_phrase_max_fraction"] >= 0.50:
            tags.append("consistent_context")
        if (
            examples_scored >= 4
            and scores["highlighted_token_max_fraction"] < 0.35
            and scores["local_phrase_max_fraction"] < 0.35
            and (
                scores["region_nonzero"] == 0
                or scores["region_max_fraction"] < 0.45
            )
            and scores["dominant_logit_family_fraction"] < 0.35
        ):
            tags.append("incoherent_feature")
        return sorted(set(tags))

    def annotate_feature(
        self,
        feature: dict[str, Any],
        metadata: dict[str, Any],
    ) -> FeatureAnnotationResult:
        feature_json = self.get_feature_json(feature)
        if feature_json is None:
            return FeatureAnnotationResult(tags=[])
        scores = self.score_feature(feature, feature_json)
        tags = self.classify_feature(scores)
        return FeatureAnnotationResult(tags=tags, scores=scores, hit_data=scores)

    def sort_key(self, hit: dict[str, Any]) -> tuple[float, float, float]:
        return (
            hit.get("highlighted_token_max_fraction", 0.0),
            hit.get("local_phrase_max_fraction", 0.0),
            hit.get("region_max_fraction", 0.0),
        )


REASONING_MOVE_PHRASES = {
    "self_correction": (
        "wait",
        "actually",
        "hold on",
        "mistake",
        "misread",
        "correction",
    ),
    "uncertainty": (
        "maybe",
        "probably",
        "not sure",
        "could be",
        "might",
        "perhaps",
        "uncertain",
    ),
    "planning": (
        "first",
        "then",
        "let's",
        "lets",
        "plan",
        "strategy",
        "step",
        "we need",
    ),
    "verification": (
        "check",
        "verify",
        "substitute",
        "test",
        "confirm",
    ),
    "answer_commitment": (
        "therefore",
        "so the answer",
        "final answer",
        "the answer is",
        "thus",
        "boxed",
    ),
    "backtracking": (
        "however",
        "instead",
        "reconsider",
        "on second thought",
        "but wait",
        "backtrack",
    ),
}


def _phrase_in_text(text: str, phrase: str) -> bool:
    if " " in phrase:
        return phrase in text
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


class ReasoningMoveAnnotator(FeatureJsonAnnotator):
    """Annotate local lexical windows that look like reasoning moves."""

    annotation_name = "reasoning_move"
    owned_tags = REASONING_MOVE_TAGS

    def __init__(
        self,
        *,
        data_dir: Path,
        top_k: int = 20,
        feature_json_cache: FeatureJsonCache | None = None,
    ) -> None:
        super().__init__(
            data_dir=data_dir,
            top_k=top_k,
            feature_json_cache=feature_json_cache,
        )

    def score_feature_json(self, feature_json: dict[str, Any]) -> dict[str, float]:
        examples = get_top_activation_examples(feature_json, top_k=self.top_k)
        example_counts = Counter()
        phrase_counts = Counter()

        for example in examples:
            text = get_local_text_window(example, before=12, after=12).lower()
            for move, phrases in REASONING_MOVE_PHRASES.items():
                hits = sum(1 for phrase in phrases if _phrase_in_text(text, phrase))
                if hits:
                    example_counts[move] += 1
                    phrase_counts[move] += hits

        scores: dict[str, float] = {"examples_scored": float(len(examples))}
        for move in REASONING_MOVE_PHRASES:
            move_examples = float(example_counts.get(move, 0))
            move_phrases = float(phrase_counts.get(move, 0))
            scores[f"{move}_example_count"] = move_examples
            scores[f"{move}_example_fraction"] = _safe_fraction(move_examples, len(examples))
            scores[f"{move}_phrase_count"] = move_phrases
        return scores

    def classify_feature(self, scores: dict[str, float]) -> list[str]:
        if scores.get("examples_scored", 0.0) <= 0:
            return []
        tags: list[str] = []
        for move in REASONING_MOVE_PHRASES:
            if (
                scores[f"{move}_example_count"] >= 2
                or (
                    scores[f"{move}_example_count"] >= 1
                    and scores[f"{move}_example_fraction"] >= 0.30
                )
            ):
                tags.append(move)
        return sorted(set(tags))

    def annotate_feature(
        self,
        feature: dict[str, Any],
        metadata: dict[str, Any],
    ) -> FeatureAnnotationResult:
        feature_json = self.get_feature_json(feature)
        if feature_json is None:
            return FeatureAnnotationResult(tags=[])
        scores = self.score_feature_json(feature_json)
        tags = self.classify_feature(scores)
        return FeatureAnnotationResult(tags=tags, scores=scores, hit_data=scores)

    def sort_key(self, hit: dict[str, Any]) -> tuple[float, float]:
        max_fraction = max(
            hit.get(f"{move}_example_fraction", 0.0)
            for move in REASONING_MOVE_PHRASES
        )
        max_count = max(
            hit.get(f"{move}_example_count", 0.0)
            for move in REASONING_MOVE_PHRASES
        )
        return (max_fraction, max_count)


ANNOTATOR_BUILDERS: dict[
    str,
    Callable[[Path, int, FeatureJsonCache], FeatureJsonAnnotator],
] = {
    "logit_effect": lambda data_dir, top_k, cache: LogitEffectFeatureAnnotator(
        data_dir=data_dir,
        top_k=top_k,
        feature_json_cache=cache,
    ),
    "activation_shape": lambda data_dir, top_k, cache: ActivationShapeFeatureAnnotator(
        data_dir=data_dir,
        top_k=top_k,
        feature_json_cache=cache,
    ),
    "feature_specificity": lambda data_dir, top_k, cache: FeatureSpecificityAnnotator(
        data_dir=data_dir,
        top_k=top_k,
        feature_json_cache=cache,
    ),
    "token_surface": lambda data_dir, top_k, cache: TokenSurfaceAnnotator(
        data_dir=data_dir,
        top_k=top_k,
        feature_json_cache=cache,
    ),
    "cross_example_consistency": lambda data_dir, top_k, cache: CrossExampleConsistencyAnnotator(
        data_dir=data_dir,
        top_k=top_k,
        feature_json_cache=cache,
    ),
    "reasoning_move": lambda data_dir, top_k, cache: ReasoningMoveAnnotator(
        data_dir=data_dir,
        top_k=top_k,
        feature_json_cache=cache,
    ),
}

DEFAULT_ANNOTATORS = tuple(ANNOTATOR_BUILDERS)


def parse_annotator_names(value: str) -> list[str]:
    raw_names = [name.strip() for name in value.split(",") if name.strip()]
    if not raw_names or raw_names == ["all"]:
        return list(DEFAULT_ANNOTATORS)
    unknown = [name for name in raw_names if name not in ANNOTATOR_BUILDERS]
    if unknown:
        known = ", ".join(ANNOTATOR_BUILDERS)
        raise ValueError(f"Unknown annotator(s): {', '.join(unknown)}. Known: {known}")
    return raw_names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Annotate feature patterns from feature collection outputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_annotation_args(parser)
    parser.add_argument(
        "--annotators",
        default=",".join(DEFAULT_ANNOTATORS),
        help=(
            "Comma-separated annotator names to run, or 'all'. Known: "
            + ", ".join(ANNOTATOR_BUILDERS)
        ),
    )
    return parser


def main() -> None:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args()
    try:
        annotator_names = parse_annotator_names(args.annotators)
    except ValueError as exc:
        parser.error(str(exc))

    feature_json_cache: FeatureJsonCache = {}
    annotators = [
        ANNOTATOR_BUILDERS[name](args.data_dir, args.top_k, feature_json_cache)
        for name in annotator_names
    ]
    run_annotation(
        data_dir=args.data_dir,
        annotations_file=args.annotations_file,
        annotator=annotators,
        replace_all=args.replace_all,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
