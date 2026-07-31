"""Helpers for passing feature steering through vLLM HF config overrides."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from models.steering import FeatureSteeringMode, FeatureSteeringSpec

STEERING_SPECS_CONFIG_KEY = "transcoder_feature_steering_specs"
STEERING_MODE_CONFIG_KEY = "transcoder_feature_steering_mode"
_VALID_MODES = frozenset(("min", "add", "set"))


def _validate_mode(mode: str) -> FeatureSteeringMode:
    if mode not in _VALID_MODES:
        available = ", ".join(sorted(_VALID_MODES))
        raise ValueError(f"Unknown feature steering mode {mode!r}. Available: {available}")
    return mode  # type: ignore[return-value]


def feature_steering_hf_overrides(
    specs: Iterable[FeatureSteeringSpec],
    mode: FeatureSteeringMode = "min",
) -> dict[str, Any]:
    """Build ``LLM(..., hf_overrides=...)`` values for global vLLM steering."""
    mode = _validate_mode(mode)
    specs_tuple = tuple(FeatureSteeringSpec(spec.cantor_id, spec.strength) for spec in specs)
    return {
        STEERING_SPECS_CONFIG_KEY: [
            {"cantor_id": spec.cantor_id, "strength": spec.strength}
            for spec in specs_tuple
        ],
        STEERING_MODE_CONFIG_KEY: mode,
    }


def get_feature_steering_from_config(config: Any) -> tuple[tuple[FeatureSteeringSpec, ...], FeatureSteeringMode]:
    """Read steering specs installed on a HuggingFace config object."""
    raw_specs = getattr(config, STEERING_SPECS_CONFIG_KEY, ()) or ()
    mode = _validate_mode(getattr(config, STEERING_MODE_CONFIG_KEY, "min"))
    specs: list[FeatureSteeringSpec] = []

    for raw_spec in raw_specs:
        if isinstance(raw_spec, FeatureSteeringSpec):
            specs.append(FeatureSteeringSpec(raw_spec.cantor_id, raw_spec.strength))
            continue
        if isinstance(raw_spec, dict):
            specs.append(FeatureSteeringSpec(raw_spec["cantor_id"], raw_spec["strength"]))
            continue
        try:
            cantor_id, strength = raw_spec
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "vLLM feature steering specs must be FeatureSteeringSpec, dict, or "
                "(cantor_id, strength) pairs"
            ) from exc
        specs.append(FeatureSteeringSpec(cantor_id, strength))

    return tuple(specs), mode
