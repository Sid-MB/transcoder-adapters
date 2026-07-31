"""Feature steering helpers for transcoder activations."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch

FeatureSteeringMode = Literal["min", "add", "set"]
LayerFeatureSteeringTarget = tuple[int, float]

_VALID_FEATURE_STEERING_MODES: frozenset[str] = frozenset(("min", "add", "set"))


def _validate_nonnegative_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def cantor_pair(layer: int, feature: int) -> int:
    """Map a nonnegative ``(layer, feature)`` pair to a unique integer."""
    layer = _validate_nonnegative_int(layer, "layer")
    feature = _validate_nonnegative_int(feature, "feature")
    summed = layer + feature
    return summed * (summed + 1) // 2 + feature


def cantor_unpair(cantor_id: int) -> tuple[int, int]:
    """Invert ``cantor_pair`` exactly for nonnegative integer IDs."""
    cantor_id = _validate_nonnegative_int(cantor_id, "cantor_id")
    diagonal = (math.isqrt(8 * cantor_id + 1) - 1) // 2
    diagonal_start = diagonal * (diagonal + 1) // 2
    feature = cantor_id - diagonal_start
    layer = diagonal - feature
    return layer, feature


@dataclass(frozen=True)
class FeatureSteeringSpec:
    """A global feature steering target encoded by Cantor ID."""

    cantor_id: int
    strength: float

    def __post_init__(self) -> None:
        cantor_id = _validate_nonnegative_int(self.cantor_id, "cantor_id")
        strength = float(self.strength)
        if not math.isfinite(strength) or strength < 0:
            raise ValueError("strength must be finite and nonnegative")
        object.__setattr__(self, "cantor_id", cantor_id)
        object.__setattr__(self, "strength", strength)


def _validate_mode(mode: str) -> FeatureSteeringMode:
    if mode not in _VALID_FEATURE_STEERING_MODES:
        available = ", ".join(sorted(_VALID_FEATURE_STEERING_MODES))
        raise ValueError(f"Unknown feature steering mode {mode!r}. Available: {available}")
    return mode  # type: ignore[return-value]


def apply_feature_steering(
    features: torch.Tensor,
    targets: Sequence[LayerFeatureSteeringTarget] | None,
    mode: FeatureSteeringMode,
) -> torch.Tensor:
    """Return ``features`` with layer-local feature steering applied.

    ``targets`` contains ``(feature_index, strength)`` pairs for the current
    layer. The input tensor is never mutated in-place.
    """
    mode = _validate_mode(mode)
    if not targets:
        return features

    n_features = features.shape[-1]
    seen: set[int] = set()
    target_indices: list[int] = []
    strength_values: list[float] = []

    for feature_idx, strength in targets:
        feature_idx = _validate_nonnegative_int(feature_idx, "feature")
        if feature_idx >= n_features:
            raise ValueError(f"feature index {feature_idx} out of range for {n_features} features")
        if feature_idx in seen:
            raise ValueError(f"Duplicate feature steering target for feature {feature_idx}")
        seen.add(feature_idx)

        strength_value = float(strength)
        if not math.isfinite(strength_value) or strength_value < 0:
            raise ValueError("strength must be finite and nonnegative")
        target_indices.append(feature_idx)
        strength_values.append(strength_value)

    index_tensor = torch.tensor(target_indices, dtype=torch.long, device=features.device)
    strength_tensor = features.new_tensor(strength_values)
    target_values = torch.zeros(n_features, dtype=features.dtype, device=features.device).scatter(
        0,
        index_tensor,
        strength_tensor,
    )
    target_mask = torch.zeros(n_features, dtype=torch.bool, device=features.device).scatter(
        0,
        index_tensor,
        torch.ones_like(index_tensor, dtype=torch.bool),
    )

    view_shape = (1,) * (features.ndim - 1) + (n_features,)
    target_values = target_values.view(view_shape)
    target_mask = target_mask.view(view_shape)

    if mode == "min":
        return torch.where(target_mask, torch.maximum(features, target_values), features)
    if mode == "add":
        return features + target_values
    return torch.where(target_mask, target_values, features)


def _coerce_feature_steering_spec(spec: FeatureSteeringSpec) -> FeatureSteeringSpec:
    if not isinstance(spec, FeatureSteeringSpec):
        raise TypeError("feature steering specs must be FeatureSteeringSpec instances")
    return FeatureSteeringSpec(spec.cantor_id, spec.strength)


def _mlp_sequence(mlps: Iterable[Any]) -> tuple[Any, ...]:
    mlp_tuple = tuple(mlps)
    if not mlp_tuple:
        raise ValueError("Model has no transcoder MLP layers")
    return mlp_tuple


def configure_feature_steering(
    mlps: Iterable[Any],
    specs: Iterable[FeatureSteeringSpec],
    mode: FeatureSteeringMode = "min",
) -> None:
    """Validate and install layer-local steering targets on transcoder MLPs."""
    mode = _validate_mode(mode)
    mlp_tuple = _mlp_sequence(mlps)
    specs_tuple = tuple(_coerce_feature_steering_spec(spec) for spec in specs)

    seen_cantor_ids: set[int] = set()
    layer_targets: list[list[LayerFeatureSteeringTarget]] = [[] for _ in mlp_tuple]

    for spec in specs_tuple:
        if spec.cantor_id in seen_cantor_ids:
            raise ValueError(f"Duplicate feature steering target for Cantor ID {spec.cantor_id}")
        seen_cantor_ids.add(spec.cantor_id)

        layer_idx, feature_idx = cantor_unpair(spec.cantor_id)
        if layer_idx >= len(mlp_tuple):
            raise ValueError(
                f"Feature steering layer {layer_idx} out of range for {len(mlp_tuple)} layers"
            )

        n_features = getattr(mlp_tuple[layer_idx], "n_features", None)
        if n_features is None:
            raise ValueError(f"Layer {layer_idx} MLP does not expose n_features")
        if feature_idx >= n_features:
            raise ValueError(
                f"Feature steering feature {feature_idx} out of range for layer {layer_idx} "
                f"with {n_features} features"
            )

        layer_targets[layer_idx].append((feature_idx, spec.strength))

    for mlp, targets in zip(mlp_tuple, layer_targets, strict=True):
        mlp.feature_steering_targets = tuple(targets)
        mlp.feature_steering_mode = mode


def clear_feature_steering(mlps: Iterable[Any]) -> None:
    """Remove all feature steering state from transcoder MLPs."""
    for mlp in mlps:
        mlp.feature_steering_targets = ()
        mlp.feature_steering_mode = "min"


def get_feature_steering(
    mlps: Iterable[Any],
) -> tuple[tuple[FeatureSteeringSpec, ...], FeatureSteeringMode]:
    """Read feature steering specs back from layer-local MLP state."""
    specs: list[FeatureSteeringSpec] = []
    active_mode: FeatureSteeringMode | None = None

    for layer_idx, mlp in enumerate(mlps):
        targets = getattr(mlp, "feature_steering_targets", ())
        mode = _validate_mode(getattr(mlp, "feature_steering_mode", "min"))
        if not targets:
            continue
        if active_mode is None:
            active_mode = mode
        elif mode != active_mode:
            raise RuntimeError("Inconsistent feature steering modes across transcoder layers")

        for feature_idx, strength in targets:
            specs.append(FeatureSteeringSpec(cantor_pair(layer_idx, feature_idx), strength))

    return tuple(specs), (active_mode or "min")


class FeatureSteeringMixin:
    """Shared model-level API for transcoder feature steering."""

    def _transcoder_mlps(self) -> Iterator[Any]:
        raise NotImplementedError

    def set_feature_steering(
        self,
        specs: Iterable[FeatureSteeringSpec],
        mode: FeatureSteeringMode = "min",
    ) -> None:
        configure_feature_steering(self._transcoder_mlps(), specs, mode)

    def clear_feature_steering(self) -> None:
        clear_feature_steering(self._transcoder_mlps())

    def get_feature_steering(self) -> tuple[tuple[FeatureSteeringSpec, ...], FeatureSteeringMode]:
        return get_feature_steering(self._transcoder_mlps())
