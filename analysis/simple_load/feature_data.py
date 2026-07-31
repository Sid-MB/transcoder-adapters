"""Feature-data snippets for simple_load steering targets."""

from __future__ import annotations

import json
from pathlib import Path

from models.steering import FeatureSteeringSpec, cantor_unpair


def _feature_path(feature_data_dir: Path, cantor_id: int) -> Path:
    if feature_data_dir.name == "features":
        return feature_data_dir / f"{cantor_id}.json"
    return feature_data_dir / "features" / f"{cantor_id}.json"


def _load_feature_json(feature_data_dir: Path, cantor_id: int) -> dict | None:
    path = _feature_path(feature_data_dir, cantor_id)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _top_examples(feature_json: dict, n_examples: int) -> tuple[str, list[dict]]:
    fallback_name = ""
    fallback_examples: list[dict] = []

    for quantile in feature_json.get("examples_quantiles", []):
        name = quantile.get("quantile_name", "")
        examples = quantile.get("examples", [])
        if name == "Top activations":
            return name, examples[:n_examples]
        if not fallback_examples and name.startswith("Top activations"):
            fallback_name = name
            fallback_examples = examples

    return fallback_name, fallback_examples[:n_examples]


def _activation(example: dict) -> float | None:
    peak = example.get("peak_activation")
    if peak is not None:
        return float(peak)

    idx = example.get("train_token_ind")
    acts = example.get("tokens_acts_list", [])
    if idx is None or idx < 0 or idx >= len(acts):
        return None
    return float(acts[idx])


def _format_example_with_marker(example: dict) -> str:
    """Format feature examples using the marker convention from feature analysis."""
    tokens = example.get("tokens", [])
    highlight_idx = example.get("train_token_ind")

    parts = []
    for idx, token in enumerate(tokens):
        if idx == highlight_idx:
            parts.append(f"<<<{token}>>>")
        else:
            parts.append(token)
    return "".join(parts)


def format_steered_feature_snippets(
    feature_data_dir: str,
    specs: list[FeatureSteeringSpec],
    n_examples: int = 3,
) -> str:
    """Return top-activation snippets for steered feature IDs."""
    data_dir = Path(feature_data_dir).expanduser()
    lines: list[str] = []

    for spec in specs:
        layer_idx, feature_idx = cantor_unpair(spec.cantor_id)
        lines.append(
            f"\n[FEATURE DATA] Cantor ID {spec.cantor_id} "
            f"(layer {layer_idx}, feature {feature_idx}, strength={spec.strength:g})"
        )

        feature_json = _load_feature_json(data_dir, spec.cantor_id)
        if feature_json is None:
            lines.append(f"  No feature JSON found at {_feature_path(data_dir, spec.cantor_id)}")
            continue

        quantile_name, examples = _top_examples(feature_json, n_examples)
        if not examples:
            lines.append("  No top activation examples found in feature JSON.")
            continue

        if quantile_name:
            lines.append(f"  {quantile_name}:")
        else:
            lines.append("  Top activation examples:")

        for idx, example in enumerate(examples, start=1):
            activation = _activation(example)
            activation_text = "unknown" if activation is None else f"{activation:.6g}"
            lines.append(f"  {idx}. act={activation_text}")
            lines.append(_format_example_with_marker(example))

    return "\n".join(lines)
