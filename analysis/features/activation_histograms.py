from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np


def build_default_histogram_bin_lower_bounds() -> np.ndarray:
    dense = np.round(np.arange(0.0, 8.0 + 0.1, 0.1), 1)
    tail = np.array([10.0, 12.0, 16.0, 24.0], dtype=np.float32)
    return np.concatenate([dense.astype(np.float32), tail]).astype(np.float32)


DEFAULT_HISTOGRAM_BIN_LOWER_BOUNDS = build_default_histogram_bin_lower_bounds()

DEFAULT_ACTIVATION_EXAMPLE_RANGES = (
    "0.5:1.0,1.0:2.0,2.0:2.5,2.5:3.0,3.0:4.0,4.0:8.0"
)


def parse_activation_example_ranges(value: str) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    if not value.strip():
        return ranges
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        pieces = part.split(":")
        if len(pieces) != 2:
            raise ValueError(f"Activation range {part!r} must be formatted as lo:hi")
        lo = float(pieces[0])
        hi = float(pieces[1])
        if hi <= lo:
            raise ValueError(f"Activation range {part!r} must have hi > lo")
        ranges.append((lo, hi))
    return ranges


def format_activation_range_label(value: tuple[float, float]) -> str:
    lo, hi = value
    return f"{lo}-{hi}"


def histogram_bin_indices(
    values: np.ndarray,
    bin_lower_bounds: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    bins = np.asarray(bin_lower_bounds, dtype=np.float32)
    if bins.ndim != 1 or len(bins) == 0:
        raise ValueError("bin_lower_bounds must be a non-empty 1D array")
    indices = np.searchsorted(bins, values, side="right") - 1
    return np.clip(indices, 0, len(bins) - 1).astype(np.int64)


def approximate_histogram_quantile(
    counts: np.ndarray,
    bin_lower_bounds: np.ndarray,
    quantile: float,
) -> float | None:
    counts = np.asarray(counts)
    bins = np.asarray(bin_lower_bounds, dtype=np.float32)
    if counts.ndim != 1:
        raise ValueError("counts must be a 1D array")
    if len(counts) != len(bins):
        raise ValueError("counts and bin_lower_bounds must have the same length")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")

    total = int(counts.sum())
    if total == 0:
        return None

    rank = max(1, math.ceil(total * quantile))
    cumulative = np.cumsum(counts, dtype=np.uint64)
    idx = int(np.searchsorted(cumulative, rank, side="left"))
    lower = float(bins[idx])
    if idx + 1 >= len(bins):
        return lower
    upper = float(bins[idx + 1])
    return (lower + upper) / 2.0


def max_nonzero_bin_lower_bound(
    counts: np.ndarray,
    bin_lower_bounds: np.ndarray,
) -> float | None:
    counts = np.asarray(counts)
    bins = np.asarray(bin_lower_bounds, dtype=np.float32)
    if counts.ndim != 1:
        raise ValueError("counts must be a 1D array")
    if len(counts) != len(bins):
        raise ValueError("counts and bin_lower_bounds must have the same length")
    nonzero = np.flatnonzero(counts)
    if len(nonzero) == 0:
        return None
    return float(bins[int(nonzero[-1])])


def _density(count: int, tokens: int) -> float:
    return count / tokens if tokens > 0 else 0.0


def compute_relative_domain_scores(
    *,
    domain_counts: dict[str, int],
    tokens_per_domain: dict[str, int],
    domain_names: list[str],
    feature_hist_by_domain: np.ndarray,
    bin_lower_bounds: np.ndarray,
    target_domain: str,
    baseline_domain: str,
) -> dict[str, Any] | None:
    if target_domain not in domain_names or baseline_domain not in domain_names:
        return None

    target_tokens = int(tokens_per_domain.get(target_domain) or 0)
    baseline_tokens = int(tokens_per_domain.get(baseline_domain) or 0)
    if target_tokens <= 0 or baseline_tokens <= 0:
        return None

    target_idx = domain_names.index(target_domain)
    baseline_idx = domain_names.index(baseline_domain)
    target_count = int(domain_counts.get(target_domain) or 0)
    baseline_count = int(domain_counts.get(baseline_domain) or 0)
    target_density = _density(target_count, target_tokens)
    baseline_density = _density(baseline_count, baseline_tokens)
    target_density_smoothed = max(target_density, 1.0 / target_tokens)
    baseline_density_smoothed = max(baseline_density, 1.0 / baseline_tokens)
    density_ratio = target_density_smoothed / baseline_density_smoothed

    target_hist = feature_hist_by_domain[target_idx]
    baseline_hist = feature_hist_by_domain[baseline_idx]
    target_p95 = approximate_histogram_quantile(target_hist, bin_lower_bounds, 0.95)
    baseline_p95 = approximate_histogram_quantile(baseline_hist, bin_lower_bounds, 0.95)
    target_p99 = approximate_histogram_quantile(target_hist, bin_lower_bounds, 0.99)
    baseline_p99 = approximate_histogram_quantile(baseline_hist, bin_lower_bounds, 0.99)

    p95_ratio = None
    if target_p95 is not None and baseline_p95 is not None and baseline_p95 > 0:
        p95_ratio = target_p95 / baseline_p95
    p99_ratio = None
    if target_p99 is not None and baseline_p99 is not None and baseline_p99 > 0:
        p99_ratio = target_p99 / baseline_p99

    target_max = max_nonzero_bin_lower_bound(target_hist, bin_lower_bounds)
    baseline_max = max_nonzero_bin_lower_bound(baseline_hist, bin_lower_bounds)
    max_nonzero_bin_ratio = None
    if target_max is not None and baseline_max is not None and baseline_max > 0:
        max_nonzero_bin_ratio = target_max / baseline_max

    return {
        "target_domain": target_domain,
        "baseline_domain": baseline_domain,
        "target_count": target_count,
        "baseline_count": baseline_count,
        "target_tokens": target_tokens,
        "baseline_tokens": baseline_tokens,
        "target_density": target_density,
        "baseline_density": baseline_density,
        "target_density_smoothed": target_density_smoothed,
        "baseline_density_smoothed": baseline_density_smoothed,
        "density_ratio": density_ratio,
        "log2_density_lift": math.log2(density_ratio),
        "target_p95": target_p95,
        "baseline_p95": baseline_p95,
        "p95_ratio": p95_ratio,
        "target_p99": target_p99,
        "baseline_p99": baseline_p99,
        "p99_ratio": p99_ratio,
        "target_max_nonzero_bin": target_max,
        "baseline_max_nonzero_bin": baseline_max,
        "max_nonzero_bin_ratio": max_nonzero_bin_ratio,
    }


def save_activation_histograms_npz(
    *,
    path: Path,
    bin_lower_bounds: np.ndarray,
    domain_names: list[str],
    run_hist_total: np.ndarray,
    run_hist_by_domain: np.ndarray,
    feature_hist_total_by_layer: list[np.ndarray],
    feature_hist_by_domain_by_layer: list[np.ndarray],
) -> None:
    payload: dict[str, np.ndarray] = {
        "bin_lower_bounds": np.asarray(bin_lower_bounds, dtype=np.float32),
        "domain_names": np.asarray(domain_names, dtype=str),
        "run_hist_total": np.asarray(run_hist_total, dtype=np.uint64),
        "run_hist_by_domain": np.asarray(run_hist_by_domain, dtype=np.uint64),
    }
    for layer_idx, counts in enumerate(feature_hist_total_by_layer):
        payload[f"feature_hist_total_layer_{layer_idx}"] = np.asarray(counts, dtype=np.uint32)
    for layer_idx, counts in enumerate(feature_hist_by_domain_by_layer):
        payload[f"feature_hist_by_domain_layer_{layer_idx}"] = np.asarray(
            counts,
            dtype=np.uint32,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        np.savez_compressed(f, **payload)
    tmp_path.replace(path)
