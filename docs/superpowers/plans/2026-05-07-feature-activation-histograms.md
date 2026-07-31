# Feature Activation Histograms Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collect exact all-nonzero activation magnitude histograms for new feature-data runs, expose target-vs-baseline domain frequency/strength scores, and make non-top activation ranges inspectable with text examples.

**Architecture:** Keep existing per-feature JSON files focused on examples, logits, activation-range sample tabs, and circuit-tracer compatibility. Store dense histogram arrays in a compact NumPy sidecar (`activation_histograms.npz`), while keeping small scalar summaries and table-sort fields in `feature_metadata.json`. The dashboard loads metadata at boot, shows range-sampled examples through the existing `examples_quantiles` tabs, and fetches one feature histogram at a time through a server endpoint.

**Tech Stack:** Python, NumPy, PyTorch tensors converted at aggregation boundaries, stdlib `unittest`, the existing `helpers.log` logger, existing browser dashboard served by `analysis.features.visualize.feature_dashboard`.

---

## File Structure

- Create `analysis/features/activation_histograms.py`: pure histogram utilities, relative-score formulas, and NPZ serialization helpers.
- Create `tests/test_activation_histograms.py`: unit tests for binning, quantile approximation, score smoothing, and NPZ roundtrip shape.
- Modify `pyproject.toml` and `uv.lock`: add NumPy as an explicit dependency because this feature imports NumPy directly from collection and dashboard code.
- Modify `analysis/features/collect_feature_activations.py`: initialize histogram arrays after domains are known, update arrays during nonzero activation processing, collect bounded activation-range examples, save `activation_histograms.npz`, add relative score metadata, and add CLI options.
- Modify `analysis/features/visualize/feature_dashboard.py`: add a small JSON endpoint for one feature histogram from the NPZ sidecar.
- Create `tests/test_feature_dashboard_histograms.py`: unit tests for dashboard histogram payload loading and the route-level `/api/feature_hist/{cantor_id}` behavior.
- Modify `analysis/features/visualize/dashboard.html`: add metadata-only relative-score sorting/table display, feature-frequency distributions, a joint chat-vs-fineweb scatter, and lazy per-feature histogram rendering.
- Modify `analysis/features/visualize/README.md` and `analysis/README.md`: document the new artifact and that exact histograms require a newly collected run.

## Artifact Contract

`feature_data/activation_histograms.npz` contains:

- `bin_lower_bounds`: `float32[n_bins]`
- `domain_names`: string array of length `n_domains`
- `run_hist_total`: `uint64[n_bins]`
- `run_hist_by_domain`: `uint64[n_domains, n_bins]`
- `feature_hist_total_layer_{L}`: `uint32[n_features, n_bins]`
- `feature_hist_by_domain_layer_{L}`: `uint32[n_domains, n_features, n_bins]`

`feature_metadata.json` adds:

- top-level `activation_histograms_file: "activation_histograms.npz"`
- top-level `activation_histogram_summary` with `bin_lower_bounds`, `domain_names`, `run_hist_total`, and `run_hist_by_domain`
- top-level `feature_frequency_summary` with per-feature firing-frequency histograms over all features, including zero-firing features
- top-level `relative_score_config` with `target_domain`, `baseline_domain`, and `smoothing`
- per-feature `relative_domain_scores` when both selected domains have tokens; include density lift, p95, p99, and strongest nonzero bin summaries

`feature_frequency_summary` contains:

- `frequency_bin_lower_bounds`: lower-bound bins for per-feature firing rates
- `domain_names`
- `all_features_total`: `n_layers * n_features`
- `active_features_total`: count of features with at least one nonzero activation
- `feature_frequency_hist_total`: histogram of `feature activation_count / total_tokens` across all features
- `feature_frequency_hist_by_domain`: per-domain histogram of `feature domain_count / tokens_per_domain[domain]` across all features
- `global_nonzero_density_by_domain`: per-domain `sum(feature domain_count) / (tokens_per_domain[domain] * all_features_total)`, the run-level sparsity/nonzero rate for token-feature opportunities
- `active_feature_count_by_domain`: per-domain count of features with at least one activation in that domain

`features/{cantor_id}.json` continues to contain top, per-domain top, and random examples. It also adds bounded activation-range example tabs named `Activation range {lo}-{hi} ({domain})` for configured ranges. Defaults are:

- ranges: `0.5:1.0,1.0:2.0,2.0:2.5,2.5:3.0,3.0:4.0,4.0:8.0`
- examples: one reservoir-sampled example per feature, per domain, per range
- disable with `--activation_range_examples_per_domain 0`

`GET /api/feature_hist/{cantor_id}` returns a small JSON payload for one feature:

- `bin_lower_bounds`
- `domain_names`
- `total_counts`
- `by_domain_counts`
- `by_domain_token_density`: per-bin `count / tokens_per_domain[domain]`
- `by_domain_activation_fraction`: per-bin `count / max(feature_domain_count, 1)`

Do not embed dense per-feature histograms in `features/{cantor_id}.json`.

## Coverage Against Original Notes

- **All nonzero activations:** covered by exact NPZ histograms from every positive activation, not only top examples.
- **Medium activations like 2.6-3.0:** covered by default `2.5:3.0` activation-range example tabs, so text around medium activations can be inspected directly.
- **Top/random text may be uninformative:** covered by range examples plus existing top/random examples; dashboard no longer forces interpretation from the global top tail only.
- **Dataset/domain distributions:** covered by per-domain run histograms, per-domain feature histograms, token-normalized density views, and arbitrary domain labels from `--val_data domain:path`.
- **Feature firing frequency distribution:** covered by `feature_frequency_summary`, which shows histograms of per-feature activation frequency by dataset/domain and includes zero-firing features.
- **Chat activations might always be higher:** covered by per-domain feature-frequency histograms and `global_nonzero_density_by_domain`, which directly compare run-level nonzero rates by domain.
- **Joint distribution:** covered by metadata-only target-vs-baseline feature-density scatter.
- **Relative web-vs-chat score:** covered by `relative_domain_scores` with default `target=chat`, `baseline=fineweb`, but CLI options support any two domain labels.
- **Compared to strongest fineweb activations:** covered by baseline-domain p95, p99, and strongest nonzero-bin summaries and their target/baseline ratios.

## Research Readouts

The dashboard and metadata should make these questions answerable without inspecting raw JSON:

- **Activation magnitude distribution:** how all positive activation values are distributed globally and by dataset/domain.
- **Feature firing frequency distribution:** how often features fire, including zero-frequency features, globally and by dataset/domain.
- **Run-level sparsity by domain:** whether chat has more nonzero token-feature opportunities than web/fineweb overall.
- **Joint target-vs-baseline distribution:** whether features cluster as chat-heavy, web-heavy, or balanced.
- **Relative strength:** for a chat-enriched feature, whether chat activations are also stronger than the strongest fineweb activation bins.

## Implementation Execution Checklist

Use this as the high-level execution order. The detailed code snippets and commands are in Tasks 0-7 below.

- [ ] **Preflight:** run `git status --short`; note unrelated dirty files and do not edit or revert them.
- [ ] **Dependency foundation:** complete Task 0 first so NumPy is a direct dependency in both `pyproject.toml` and `uv.lock`; verify with `uv run --frozen`.
- [ ] **Pure utilities:** complete Task 1 before touching collection code; this creates the stable histogram/scoring/NPZ API that later tasks import.
- [ ] **Collector data path:** complete Task 2 and Task 3 together before running full collection tests. Verify array shapes, exact histogram count updates, activation-range sampling/export, NPZ export, and JSON-safe relative-score metadata.
- [ ] **Dashboard serving path:** complete Task 4 after collector metadata/NPZ contracts exist; keep dense histogram loading lazy and per-feature.
- [ ] **Dashboard UI path:** complete Task 5 after the endpoint test passes; verify both Python server syntax and JavaScript syntax.
- [ ] **Docs:** complete Task 6 only after the final artifact names, CLI flags, and dashboard behavior match the implementation.
- [ ] **Final verification:** complete Task 7 before reporting done. If GPU smoke is unavailable, record that explicitly and still run all CPU tests and syntax checks.

## Task 0: Declare NumPy Dependency

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`

- [ ] **Step 1: Add NumPy to direct dependencies**

In `pyproject.toml`, add NumPy to `[project].dependencies`:

```toml
    "numpy>=2.0.0",
```

- [ ] **Step 2: Verify dependency metadata parses**

Run:

```sh
uv lock
```

Expected: `uv.lock` updates the local `transcoder-adapters` package metadata so NumPy appears as a direct project dependency.

- [ ] **Step 3: Verify dependency metadata parses with the frozen lockfile**

Run:

```sh
uv run --frozen python -c "import tomllib; tomllib.load(open('pyproject.toml','rb')); import numpy; assert numpy.__version__"
```

Expected: no output and exit code 0.

- [ ] **Step 4: Commit dependency declaration**

Run:

```sh
git add pyproject.toml uv.lock
git commit -m "chore: declare numpy dependency"
```

## Task 1: Add Histogram Utility Module

**Files:**
- Create: `analysis/features/activation_histograms.py`
- Create: `tests/test_activation_histograms.py`

- [ ] **Step 1: Write tests for binning, quantiles, scores, and NPZ shapes**

Add `tests/test_activation_histograms.py`:

```python
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.features.activation_histograms import (
    DEFAULT_HISTOGRAM_BIN_LOWER_BOUNDS,
    approximate_histogram_quantile,
    build_default_histogram_bin_lower_bounds,
    compute_relative_domain_scores,
    format_activation_range_label,
    histogram_bin_indices,
    max_nonzero_bin_lower_bound,
    parse_activation_example_ranges,
    save_activation_histograms_npz,
)


class ActivationHistogramTests(unittest.TestCase):
    def test_default_bins_cover_dense_interesting_range_and_tail(self):
        bins = build_default_histogram_bin_lower_bounds()
        self.assertEqual(bins.dtype, np.float32)
        self.assertAlmostEqual(float(bins[0]), 0.0)
        self.assertAlmostEqual(float(bins[1]), 0.1, places=6)
        self.assertIn(np.float32(8.0), bins)
        self.assertIn(np.float32(24.0), bins)
        self.assertTrue(np.all(np.diff(bins) > 0))

    def test_histogram_bin_indices_uses_lower_bound_bins(self):
        bins = np.array([0.0, 0.5, 1.0, 2.0], dtype=np.float32)
        values = np.array([0.0, 0.49, 0.5, 1.99, 2.0, 9.0], dtype=np.float32)
        got = histogram_bin_indices(values, bins)
        np.testing.assert_array_equal(got, np.array([0, 0, 1, 2, 3, 3], dtype=np.int64))

    def test_approximate_histogram_quantile_uses_midpoint_for_finite_bins(self):
        bins = np.array([0.0, 1.0, 2.0, 4.0], dtype=np.float32)
        counts = np.array([0, 2, 8, 0], dtype=np.uint64)
        self.assertEqual(approximate_histogram_quantile(counts, bins, 0.95), 3.0)

    def test_approximate_histogram_quantile_returns_open_bin_lower_bound(self):
        bins = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        counts = np.array([0, 0, 5], dtype=np.uint64)
        self.assertEqual(approximate_histogram_quantile(counts, bins, 0.5), 2.0)

    def test_max_nonzero_bin_lower_bound(self):
        bins = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        self.assertEqual(max_nonzero_bin_lower_bound(np.array([0, 3, 0], dtype=np.uint64), bins), 1.0)
        self.assertIsNone(max_nonzero_bin_lower_bound(np.array([0, 0, 0], dtype=np.uint64), bins))

    def test_approximate_histogram_quantile_returns_none_for_empty_counts(self):
        bins = DEFAULT_HISTOGRAM_BIN_LOWER_BOUNDS
        counts = np.zeros(len(bins), dtype=np.uint64)
        self.assertIsNone(approximate_histogram_quantile(counts, bins, 0.95))

    def test_compute_relative_domain_scores_is_finite_with_zero_baseline(self):
        bins = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        by_domain = np.array(
            [
                [0, 3, 1],
                [0, 0, 0],
            ],
            dtype=np.uint32,
        )
        scores = compute_relative_domain_scores(
            domain_counts={"chat": 4, "fineweb": 0},
            tokens_per_domain={"chat": 100, "fineweb": 200},
            domain_names=["chat", "fineweb"],
            feature_hist_by_domain=by_domain,
            bin_lower_bounds=bins,
            target_domain="chat",
            baseline_domain="fineweb",
        )
        self.assertEqual(scores["target_count"], 4)
        self.assertEqual(scores["baseline_count"], 0)
        self.assertGreater(scores["log2_density_lift"], 0)
        self.assertIsNone(scores["p95_ratio"])
        self.assertEqual(scores["target_p99"], 2.0)
        self.assertEqual(scores["target_max_nonzero_bin"], 2.0)
        self.assertIsNone(scores["baseline_max_nonzero_bin"])

    def test_compute_relative_domain_scores_returns_none_when_domains_are_absent(self):
        got = compute_relative_domain_scores(
            domain_counts={"chat": 4},
            tokens_per_domain={"chat": 100},
            domain_names=["chat"],
            feature_hist_by_domain=np.zeros((1, 3), dtype=np.uint32),
            bin_lower_bounds=np.array([0.0, 1.0, 2.0], dtype=np.float32),
            target_domain="chat",
            baseline_domain="fineweb",
        )
        self.assertIsNone(got)

    def test_parse_activation_example_ranges(self):
        ranges = parse_activation_example_ranges("2.5:3.0,3.0:4.0")
        self.assertEqual(ranges, [(2.5, 3.0), (3.0, 4.0)])
        self.assertEqual(format_activation_range_label((2.5, 3.0)), "2.5-3.0")

    def test_save_activation_histograms_npz_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "activation_histograms.npz"
            bins = np.array([0.0, 1.0], dtype=np.float32)
            run_total = np.array([5, 7], dtype=np.uint64)
            run_by_domain = np.array([[3, 4], [2, 3]], dtype=np.uint64)
            feature_total_by_layer = [np.array([[1, 2], [3, 4]], dtype=np.uint32)]
            feature_by_domain_by_layer = [
                np.array(
                    [
                        [[1, 0], [0, 1]],
                        [[0, 2], [3, 0]],
                    ],
                    dtype=np.uint32,
                )
            ]

            save_activation_histograms_npz(
                path=path,
                bin_lower_bounds=bins,
                domain_names=["chat", "fineweb"],
                run_hist_total=run_total,
                run_hist_by_domain=run_by_domain,
                feature_hist_total_by_layer=feature_total_by_layer,
                feature_hist_by_domain_by_layer=feature_by_domain_by_layer,
            )

            loaded = np.load(path, allow_pickle=False)
            np.testing.assert_array_equal(loaded["bin_lower_bounds"], bins)
            self.assertEqual(loaded["domain_names"].tolist(), ["chat", "fineweb"])
            np.testing.assert_array_equal(loaded["run_hist_total"], run_total)
            np.testing.assert_array_equal(loaded["feature_hist_total_layer_0"], feature_total_by_layer[0])
            np.testing.assert_array_equal(
                loaded["feature_hist_by_domain_layer_0"],
                feature_by_domain_by_layer[0],
            )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and confirm the module is missing**

Run:

```sh
uv run python -m unittest tests.test_activation_histograms
```

Expected: import failure for `analysis.features.activation_histograms`.

- [ ] **Step 3: Implement histogram helpers**

Create `analysis/features/activation_histograms.py`:

```python
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


DEFAULT_ACTIVATION_EXAMPLE_RANGES = "0.5:1.0,1.0:2.0,2.0:2.5,2.5:3.0,3.0:4.0,4.0:8.0"


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
    return f"{lo:g}-{hi:g}"


def histogram_bin_indices(values: np.ndarray, bin_lower_bounds: np.ndarray) -> np.ndarray:
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

    target_p95 = approximate_histogram_quantile(
        feature_hist_by_domain[target_idx],
        bin_lower_bounds,
        0.95,
    )
    baseline_p95 = approximate_histogram_quantile(
        feature_hist_by_domain[baseline_idx],
        bin_lower_bounds,
        0.95,
    )
    target_p99 = approximate_histogram_quantile(
        feature_hist_by_domain[target_idx],
        bin_lower_bounds,
        0.99,
    )
    baseline_p99 = approximate_histogram_quantile(
        feature_hist_by_domain[baseline_idx],
        bin_lower_bounds,
        0.99,
    )
    p95_ratio = None
    if target_p95 is not None and baseline_p95 is not None and baseline_p95 > 0:
        p95_ratio = target_p95 / baseline_p95
    p99_ratio = None
    if target_p99 is not None and baseline_p99 is not None and baseline_p99 > 0:
        p99_ratio = target_p99 / baseline_p99
    target_max = max_nonzero_bin_lower_bound(
        feature_hist_by_domain[target_idx],
        bin_lower_bounds,
    )
    baseline_max = max_nonzero_bin_lower_bound(
        feature_hist_by_domain[baseline_idx],
        bin_lower_bounds,
    )
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
        payload[f"feature_hist_by_domain_layer_{layer_idx}"] = np.asarray(counts, dtype=np.uint32)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        np.savez_compressed(f, **payload)
    tmp_path.replace(path)
```

- [ ] **Step 4: Run the histogram tests**

Run:

```sh
uv run python -m unittest tests.test_activation_histograms
```

Expected: all tests pass.

- [ ] **Step 5: Commit the helper module**

Run:

```sh
git add analysis/features/activation_histograms.py tests/test_activation_histograms.py
git commit -m "feat: add activation histogram helpers"
```

## Task 2: Wire Exact Histograms Into Feature Collection

**Files:**
- Modify: `analysis/features/collect_feature_activations.py`
- Test: `tests/test_activation_histograms.py`

- [ ] **Step 1: Add collector initialization tests for expected array shapes**

Extend `tests/test_activation_histograms.py` with:

```python
    def test_collector_histogram_arrays_have_expected_shapes(self):
        from analysis.features.collect_feature_activations import FeatureCollector

        collector = FeatureCollector(
            n_layers=2,
            n_features=3,
            top_k=1,
            n_random=1,
            domain_top_k=1,
            context_before=1,
            context_after=1,
            domain_names=["chat", "fineweb"],
        )
        self.assertEqual(collector.hist_bin_lower_bounds.ndim, 1)
        n_bins = len(collector.hist_bin_lower_bounds)
        self.assertEqual(collector.run_hist_total.shape, (n_bins,))
        self.assertEqual(collector.run_hist_by_domain.shape, (2, n_bins))
        self.assertEqual(collector.feature_hist_total_by_layer[0].shape, (3, n_bins))
        self.assertEqual(collector.feature_hist_by_domain_by_layer[0].shape, (2, 3, n_bins))

    def test_collector_histogram_update_counts_expected_bins(self):
        from analysis.features.collect_feature_activations import FeatureCollector

        collector = FeatureCollector(
            n_layers=1,
            n_features=2,
            top_k=1,
            n_random=1,
            domain_top_k=1,
            context_before=1,
            context_after=1,
            domain_names=["chat", "fineweb"],
            hist_bin_lower_bounds=np.array([0.0, 1.0, 2.0], dtype=np.float32),
        )
        collector._update_activation_histograms(
            layer_idx=0,
            domain="chat",
            active_features=np.array([0, 1, 1], dtype=np.int64),
            active_values=np.array([0.2, 1.2, 2.5], dtype=np.float32),
        )

        np.testing.assert_array_equal(collector.run_hist_total, np.array([1, 1, 1], dtype=np.uint64))
        np.testing.assert_array_equal(collector.run_hist_by_domain[0], np.array([1, 1, 1], dtype=np.uint64))
        np.testing.assert_array_equal(collector.run_hist_by_domain[1], np.array([0, 0, 0], dtype=np.uint64))
        np.testing.assert_array_equal(collector.feature_hist_total_by_layer[0][0], np.array([1, 0, 0], dtype=np.uint32))
        np.testing.assert_array_equal(collector.feature_hist_total_by_layer[0][1], np.array([0, 1, 1], dtype=np.uint32))
        np.testing.assert_array_equal(collector.feature_hist_by_domain_by_layer[0][0, 1], np.array([0, 1, 1], dtype=np.uint32))

    def test_collector_keeps_bounded_activation_range_examples(self):
        import torch
        from analysis.features.collect_feature_activations import FeatureCollector

        collector = FeatureCollector(
            n_layers=1,
            n_features=1,
            top_k=1,
            n_random=1,
            domain_top_k=1,
            context_before=1,
            context_after=1,
            domain_names=["chat"],
            activation_example_ranges=[(2.5, 3.0)],
            activation_range_examples_per_domain=1,
        )
        stats = collector.stats[0][0]
        collector._maybe_add_activation_range_example(
            stats,
            activation=2.7,
            tokens=[10, 11, 12],
            position=1,
            features_gpu=torch.tensor([[0.1], [2.7], [0.2]]),
            feature_idx=0,
            domain="chat",
            region="all",
            thinking_position=None,
            sequence_idx=0,
        )
        kept = stats.activation_range_examples["chat|2.5-3.0"]
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].activation, 2.7)

    def test_build_examples_quantiles_includes_activation_ranges_in_scale(self):
        import torch
        from analysis.features.collect_feature_activations import (
            FeatureCollector,
            _build_examples_quantiles,
        )

        class TinyTokenizer:
            def decode(self, token_ids):
                return f"tok{token_ids[0]}"

        collector = FeatureCollector(
            n_layers=1,
            n_features=1,
            top_k=1,
            n_random=0,
            domain_top_k=1,
            context_before=1,
            context_after=1,
            domain_names=["chat"],
            activation_example_ranges=[(2.5, 3.0)],
            activation_range_examples_per_domain=1,
        )
        stats = collector.stats[0][0]
        collector._maybe_add_activation_range_example(
            stats,
            activation=2.7,
            tokens=[10, 11, 12],
            position=1,
            features_gpu=torch.tensor([[0.1], [2.7], [0.2]]),
            feature_idx=0,
            domain="chat",
            region="all",
            thinking_position=None,
            sequence_idx=0,
        )

        quantiles, act_min, act_max = _build_examples_quantiles(stats, TinyTokenizer())
        names = [q["quantile_name"] for q in quantiles]
        self.assertIn("Activation range 2.5-3.0 (chat)", names)
        self.assertLessEqual(act_min, 2.7)
        self.assertGreaterEqual(act_max, 2.7)
```

- [ ] **Step 2: Run the targeted test and confirm it fails**

Run:

```sh
uv run python -m unittest tests.test_activation_histograms.ActivationHistogramTests.test_collector_histogram_arrays_have_expected_shapes
```

Expected: failure because `FeatureCollector.__init__` does not accept `domain_names`. If you run the full new collector-test set, the range-example helper is also missing.

- [ ] **Step 3: Add histogram imports and collector fields**

Modify `analysis/features/collect_feature_activations.py` imports:

```python
import numpy as np

from analysis.features.activation_histograms import (
    DEFAULT_HISTOGRAM_BIN_LOWER_BOUNDS,
    DEFAULT_ACTIVATION_EXAMPLE_RANGES,
    compute_relative_domain_scores,
    format_activation_range_label,
    histogram_bin_indices,
    parse_activation_example_ranges,
    save_activation_histograms_npz,
)
```

Extend `FeatureCollector.__init__` signature:

```python
        domain_names: list[str] | None = None,
        hist_bin_lower_bounds: np.ndarray | None = None,
        activation_example_ranges: list[tuple[float, float]] | None = None,
        activation_range_examples_per_domain: int = 1,
```

Add these fields after global token counters are initialized:

```python
        self.domain_names = list(domain_names or [])
        self.domain_to_index = {domain: i for i, domain in enumerate(self.domain_names)}
        self.hist_bin_lower_bounds = np.asarray(
            hist_bin_lower_bounds
            if hist_bin_lower_bounds is not None
            else DEFAULT_HISTOGRAM_BIN_LOWER_BOUNDS,
            dtype=np.float32,
        )
        n_bins = len(self.hist_bin_lower_bounds)
        n_domains = len(self.domain_names)
        self.run_hist_total = np.zeros(n_bins, dtype=np.uint64)
        self.run_hist_by_domain = np.zeros((n_domains, n_bins), dtype=np.uint64)
        self.feature_hist_total_by_layer = [
            np.zeros((n_features, n_bins), dtype=np.uint32)
            for _ in range(n_layers)
        ]
        self.feature_hist_by_domain_by_layer = [
            np.zeros((n_domains, n_features, n_bins), dtype=np.uint32)
            for _ in range(n_layers)
        ]
        self.activation_example_ranges = list(activation_example_ranges or [])
        self.activation_range_examples_per_domain = max(0, activation_range_examples_per_domain)
```

- [ ] **Step 4: Add a vectorized histogram update method**

Add this method to `FeatureCollector`:

```python
    def _update_activation_histograms(
        self,
        *,
        layer_idx: int,
        domain: str,
        active_features: np.ndarray,
        active_values: np.ndarray,
    ) -> None:
        domain_idx = self.domain_to_index.get(domain)
        if domain_idx is None:
            raise ValueError(f"Unknown activation domain {domain!r}; known domains={self.domain_names}")

        bin_indices = histogram_bin_indices(active_values, self.hist_bin_lower_bounds)
        np.add.at(self.run_hist_total, bin_indices, 1)
        np.add.at(self.run_hist_by_domain[domain_idx], bin_indices, 1)
        np.add.at(
            self.feature_hist_total_by_layer[layer_idx],
            (active_features, bin_indices),
            1,
        )
        np.add.at(
            self.feature_hist_by_domain_by_layer[layer_idx][domain_idx],
            (active_features, bin_indices),
            1,
        )
```

- [ ] **Step 5: Add bounded activation-range example sampling**

Add these fields to `FeatureStats`:

```python
    # Activation-range examples for inspecting non-top medium activations.
    # Key format: "{domain}|{lo}-{hi}".
    activation_range_examples: dict = field(default_factory=lambda: defaultdict(list))
    activation_range_seen_counts: dict = field(default_factory=lambda: defaultdict(int))
```

Add this method to `FeatureCollector`:

```python
    def _maybe_add_activation_range_example(
        self,
        stats: FeatureStats,
        activation: float,
        tokens: list[int],
        position: int,
        features_gpu: torch.Tensor,
        feature_idx: int,
        domain: str,
        region: str,
        thinking_position: float | None,
        sequence_idx: int,
    ) -> None:
        if self.activation_range_examples_per_domain <= 0:
            return

        matched_range = None
        for activation_range in self.activation_example_ranges:
            lo, hi = activation_range
            if lo <= activation < hi:
                matched_range = activation_range
                break
        if matched_range is None:
            return

        label = format_activation_range_label(matched_range)
        key = f"{domain}|{label}"
        stats.activation_range_seen_counts[key] += 1
        kept_examples = stats.activation_range_examples[key]
        add_example = False
        replace_idx = None
        if len(kept_examples) < self.activation_range_examples_per_domain:
            add_example = True
        else:
            j = random.randint(0, stats.activation_range_seen_counts[key] - 1)
            if j < self.activation_range_examples_per_domain:
                add_example = True
                replace_idx = j
        if not add_example:
            return

        context_tokens, pos_in_ctx = self._get_context(tokens, position)
        ctx_start = max(0, position - self.context_before)
        ctx_end = min(len(tokens), position + self.context_after + 1)
        context_activations = features_gpu[ctx_start:ctx_end, feature_idx].float().cpu().tolist()
        example = ActivatingExample(
            activation=activation,
            token_id=tokens[position],
            position=position,
            context_tokens=context_tokens,
            context_activations=context_activations,
            position_in_context=pos_in_ctx,
            domain=domain,
            region=region,
            thinking_position=thinking_position,
            sequence_idx=sequence_idx,
        )
        if replace_idx is None:
            kept_examples.append(example)
        else:
            kept_examples[replace_idx] = example
```

- [ ] **Step 6: Call histogram and activation-range updates during nonzero processing**

In `process_batch()`, immediately after `active_values` is computed and before the per-activation loop, add:

```python
                self._update_activation_histograms(
                    layer_idx=layer_idx,
                    domain=domain,
                    active_features=active_features,
                    active_values=active_values,
                )
```

Inside the per-activation loop, after `stats.region_counts[region] += 1` and the thinking-bin update, call:

```python
                    self._maybe_add_activation_range_example(
                        stats,
                        act,
                        tokens,
                        pos,
                        features_gpu,
                        feature_idx,
                        domain,
                        region,
                        think_pos,
                        sequence_idx,
                    )
```

- [ ] **Step 7: Move collector construction after domain discovery**

In `main()`, keep loading and preparing `items` before constructing `FeatureCollector`. After `items` is non-empty and before sorting by length, add:

```python
    domain_names = sorted({domain for _, domain, _ in items})
    logger.info(f"Activation histogram domains: {domain_names}")
```

Then construct `FeatureCollector` with `domain_names=domain_names`, `activation_example_ranges=parse_activation_example_ranges(args.activation_example_ranges)`, and `activation_range_examples_per_domain=args.activation_range_examples_per_domain`. Remove the earlier collector construction before item preparation.

- [ ] **Step 8: Add activation-range CLI arguments**

Add parser args:

```python
    parser.add_argument(
        "--activation_example_ranges",
        type=str,
        default=DEFAULT_ACTIVATION_EXAMPLE_RANGES,
        help=(
            "Comma-separated activation ranges formatted as lo:hi. "
            "The collector stores bounded per-feature/per-domain examples for these ranges "
            "so medium activations can be inspected in the dashboard."
        ),
    )
    parser.add_argument(
        "--activation_range_examples_per_domain",
        type=int,
        default=1,
        help=(
            "Reservoir-sampled examples per feature, per domain, per configured activation range. "
            "Set to 0 to disable activation-range example tabs."
        ),
    )
```

- [ ] **Step 9: Export activation-range examples as example tabs and include them in activation scaling**

Add this helper near `format_example_for_circuit_tracer()`:

```python
def _build_examples_quantiles(
    stats: FeatureStats,
    tokenizer,
) -> tuple[list[dict], float, float]:
    top_examples = sorted(stats.top_k_examples, key=lambda x: -x.activation)
    random_examples = stats.random_examples

    top_formatted = [
        format_example_for_circuit_tracer(ex, tokenizer)
        for ex in top_examples
    ]
    random_formatted = [
        format_example_for_circuit_tracer(ex, tokenizer)
        for ex in random_examples
    ]

    domain_quantiles = []
    for domain_name, domain_heap in sorted(stats.domain_top_k_examples.items()):
        domain_sorted = sorted(domain_heap, key=lambda x: -x.activation)
        domain_formatted = [
            format_example_for_circuit_tracer(ex, tokenizer)
            for ex in domain_sorted
        ]
        domain_quantiles.append({
            "quantile_name": f"Top activations ({domain_name})",
            "examples": domain_formatted,
        })

    range_quantiles = []
    range_examples_for_scale = []
    for range_key, range_examples in sorted(stats.activation_range_examples.items()):
        domain_name, range_label = range_key.split("|", 1)
        range_sorted = sorted(range_examples, key=lambda x: x.activation)
        range_examples_for_scale.extend(range_sorted)
        range_formatted = [
            format_example_for_circuit_tracer(ex, tokenizer)
            for ex in range_sorted
        ]
        range_quantiles.append({
            "quantile_name": f"Activation range {range_label} ({domain_name})",
            "examples": range_formatted,
        })

    all_acts = []
    for ex in top_examples + random_examples + range_examples_for_scale:
        all_acts.extend(ex.context_activations)
    act_min = min(all_acts) if all_acts else 0.0
    act_max = max(all_acts) if all_acts else 1.0

    return (
        [
            {"quantile_name": "Top activations", "examples": top_formatted},
            *domain_quantiles,
            *range_quantiles,
            {"quantile_name": "Random samples", "examples": random_formatted},
        ],
        act_min,
        act_max,
    )
```

In `export_circuit_tracer_json()`, replace the existing top/random/domain formatting and `act_min`/`act_max` block with:

```python
                    examples_quantiles, act_min, act_max = _build_examples_quantiles(
                        stats,
                        tokenizer,
                    )
```

Set the feature JSON field to the helper result:

```python
                        "examples_quantiles": examples_quantiles,
```

Update `_release_feature_examples()`:

```python
    stats.activation_range_examples.clear()
    stats.activation_range_seen_counts.clear()
```

- [ ] **Step 10: Add NPZ export**

Add this function near the existing export functions:

```python
def export_activation_histograms(collector: FeatureCollector, output_dir: Path) -> Path:
    hist_path = output_dir / "activation_histograms.npz"
    logger.info(f"Saving activation histograms to {hist_path}...")
    save_activation_histograms_npz(
        path=hist_path,
        bin_lower_bounds=collector.hist_bin_lower_bounds,
        domain_names=collector.domain_names,
        run_hist_total=collector.run_hist_total,
        run_hist_by_domain=collector.run_hist_by_domain,
        feature_hist_total_by_layer=collector.feature_hist_total_by_layer,
        feature_hist_by_domain_by_layer=collector.feature_hist_by_domain_by_layer,
    )
    logger.info(f"Saved activation histograms to {hist_path}")
    return hist_path
```

Call it after `export_circuit_tracer_json(...)` and before `export_metadata(...)`:

```python
    export_activation_histograms(collector, output_dir)
```

- [ ] **Step 11: Run targeted tests**

Run:

```sh
uv run python -m unittest tests.test_activation_histograms
```

Expected: all tests pass.

- [ ] **Step 12: Run py_compile for collector syntax**

Run:

```sh
uv run python -m py_compile analysis/features/collect_feature_activations.py analysis/features/activation_histograms.py
```

Expected: no output and exit code 0.

- [ ] **Step 13: Commit collector histogram and range-example wiring**

Run:

```sh
git add analysis/features/collect_feature_activations.py analysis/features/activation_histograms.py tests/test_activation_histograms.py
git commit -m "feat: collect activation histograms and range examples"
```

## Task 3: Add Feature-Frequency and Relative Domain Score Metadata

**Files:**
- Modify: `analysis/features/collect_feature_activations.py`
- Test: `tests/test_activation_histograms.py`

- [ ] **Step 1: Add metadata export tests**

Extend `tests/test_activation_histograms.py` with:

```python
    def test_metadata_summary_and_relative_scores_are_json_safe(self):
        import json
        from analysis.features.collect_feature_activations import (
            FeatureCollector,
            build_feature_metadata_entry,
            build_metadata_payload,
        )

        collector = FeatureCollector(
            n_layers=1,
            n_features=2,
            top_k=1,
            n_random=1,
            domain_top_k=1,
            context_before=1,
            context_after=1,
            domain_names=["chat", "fineweb"],
        )
        collector.total_tokens = 300
        collector.tokens_per_domain["chat"] = 100
        collector.tokens_per_domain["fineweb"] = 200

        stats = collector.stats[0][0]
        stats.activation_count = 4
        stats.domain_counts["chat"] = 4
        collector.feature_hist_by_domain_by_layer[0][0, 0, 10] = 4

        feature_meta = build_feature_metadata_entry(
            collector=collector,
            stats=stats,
            layer_idx=0,
            feature_idx=0,
            target_domain="chat",
            baseline_domain="fineweb",
        )
        self.assertIn("relative_domain_scores", feature_meta)
        self.assertGreater(feature_meta["relative_domain_scores"]["log2_density_lift"], 0)
        self.assertIn("target_p99", feature_meta["relative_domain_scores"])
        self.assertIn("target_max_nonzero_bin", feature_meta["relative_domain_scores"])

        payload = build_metadata_payload(
            collector=collector,
            target_domain="chat",
            baseline_domain="fineweb",
        )
        encoded = json.dumps(payload)
        self.assertIn("activation_histograms.npz", encoded)
        self.assertIn("activation_histogram_summary", payload)
        self.assertIn("feature_frequency_summary", payload)
        frequency_summary = payload["feature_frequency_summary"]
        self.assertEqual(frequency_summary["all_features_total"], 2)
        self.assertEqual(frequency_summary["active_features_total"], 1)
        self.assertIn("chat", frequency_summary["feature_frequency_hist_by_domain"])
        self.assertIn("fineweb", frequency_summary["feature_frequency_hist_by_domain"])
        self.assertGreater(frequency_summary["global_nonzero_density_by_domain"]["chat"], 0)
```

- [ ] **Step 2: Run the metadata test and confirm helper functions are missing**

Run:

```sh
uv run python -m unittest tests.test_activation_histograms.ActivationHistogramTests.test_metadata_summary_and_relative_scores_are_json_safe
```

Expected: import failure for `build_feature_metadata_entry` or `build_metadata_payload`.

- [ ] **Step 3: Extract metadata entry creation into a helper**

In `analysis/features/collect_feature_activations.py`, add:

```python
def build_feature_metadata_entry(
    *,
    collector: FeatureCollector,
    stats: FeatureStats,
    layer_idx: int,
    feature_idx: int,
    target_domain: str,
    baseline_domain: str,
) -> dict[str, Any]:
    total_acts = stats.activation_count

    domain_density = {}
    domain_fraction = {}
    for domain, count in stats.domain_counts.items():
        domain_tokens = collector.tokens_per_domain.get(domain, 0)
        domain_density[domain] = count / domain_tokens if domain_tokens > 0 else 0
        domain_fraction[domain] = count / total_acts

    region_density = {}
    region_fraction = {}
    for region, count in stats.region_counts.items():
        region_tokens = collector.tokens_per_region.get(region, 0)
        region_density[region] = count / region_tokens if region_tokens > 0 else 0
        region_fraction[region] = count / total_acts

    thinking_density = []
    thinking_fraction = []
    thinking_total = sum(stats.thinking_position_counts)
    for bin_idx in range(10):
        bin_acts = stats.thinking_position_counts[bin_idx]
        bin_tokens = collector.tokens_per_thinking_bin[bin_idx]
        thinking_density.append(bin_acts / bin_tokens if bin_tokens > 0 else 0)
        thinking_fraction.append(bin_acts / thinking_total if thinking_total > 0 else 0)

    feature_meta = {
        "layer": layer_idx,
        "feature": feature_idx,
        "cantor_id": cantor_pair(layer_idx, feature_idx),
        "activation_count": total_acts,
        "activation_freq": total_acts / collector.total_tokens,
        "domain_density": domain_density,
        "domain_fraction": domain_fraction,
        "region_density": region_density,
        "region_fraction": region_fraction,
        "thinking_position_density": thinking_density,
        "thinking_position_fraction": thinking_fraction,
    }
    relative_scores = compute_relative_domain_scores(
        domain_counts=dict(stats.domain_counts),
        tokens_per_domain=dict(collector.tokens_per_domain),
        domain_names=collector.domain_names,
        feature_hist_by_domain=collector.feature_hist_by_domain_by_layer[layer_idx][:, feature_idx, :],
        bin_lower_bounds=collector.hist_bin_lower_bounds,
        target_domain=target_domain,
        baseline_domain=baseline_domain,
    )
    if relative_scores is not None:
        feature_meta["relative_domain_scores"] = relative_scores
    return feature_meta
```

- [ ] **Step 4: Extract full metadata payload creation**

Add this helper before `build_metadata_payload()`:

```python
def build_feature_frequency_summary(collector: FeatureCollector) -> dict[str, Any]:
    frequency_bin_lower_bounds = np.array(
        [
            0.0,
            1e-9,
            3e-9,
            1e-8,
            3e-8,
            1e-7,
            3e-7,
            1e-6,
            3e-6,
            1e-5,
            3e-5,
            1e-4,
            3e-4,
            1e-3,
            3e-3,
            1e-2,
            3e-2,
            1e-1,
            3e-1,
            1.0,
        ],
        dtype=np.float32,
    )
    n_total_features = collector.n_layers * collector.n_features
    total_counts = np.zeros(n_total_features, dtype=np.uint64)
    counts_by_domain = {
        domain: np.zeros(n_total_features, dtype=np.uint64)
        for domain in collector.domain_names
    }

    flat_idx = 0
    for layer_idx in range(collector.n_layers):
        for feature_idx in range(collector.n_features):
            stats = collector.stats[layer_idx][feature_idx]
            total_counts[flat_idx] = int(stats.activation_count)
            for domain in collector.domain_names:
                counts_by_domain[domain][flat_idx] = int(stats.domain_counts.get(domain) or 0)
            flat_idx += 1

    total_tokens = max(int(collector.total_tokens), 1)
    total_freqs = total_counts.astype(np.float64) / total_tokens
    total_bins = histogram_bin_indices(total_freqs, frequency_bin_lower_bounds)
    feature_frequency_hist_total = np.bincount(
        total_bins,
        minlength=len(frequency_bin_lower_bounds),
    ).astype(int).tolist()

    feature_frequency_hist_by_domain: dict[str, list[int]] = {}
    global_nonzero_density_by_domain: dict[str, float] = {}
    active_feature_count_by_domain: dict[str, int] = {}
    for domain, counts in counts_by_domain.items():
        domain_tokens = int(collector.tokens_per_domain.get(domain) or 0)
        if domain_tokens > 0:
            domain_freqs = counts.astype(np.float64) / domain_tokens
            domain_bins = histogram_bin_indices(domain_freqs, frequency_bin_lower_bounds)
            feature_frequency_hist_by_domain[domain] = np.bincount(
                domain_bins,
                minlength=len(frequency_bin_lower_bounds),
            ).astype(int).tolist()
            global_nonzero_density_by_domain[domain] = (
                float(counts.sum()) / (domain_tokens * max(n_total_features, 1))
            )
        else:
            feature_frequency_hist_by_domain[domain] = [0] * len(frequency_bin_lower_bounds)
            global_nonzero_density_by_domain[domain] = 0.0
        active_feature_count_by_domain[domain] = int(np.count_nonzero(counts))

    return {
        "frequency_bin_lower_bounds": frequency_bin_lower_bounds.tolist(),
        "domain_names": collector.domain_names,
        "all_features_total": n_total_features,
        "active_features_total": int(np.count_nonzero(total_counts)),
        "feature_frequency_hist_total": feature_frequency_hist_total,
        "feature_frequency_hist_by_domain": feature_frequency_hist_by_domain,
        "global_nonzero_density_by_domain": global_nonzero_density_by_domain,
        "active_feature_count_by_domain": active_feature_count_by_domain,
    }
```

Then add:

```python
def build_metadata_payload(
    *,
    collector: FeatureCollector,
    target_domain: str,
    baseline_domain: str,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "total_tokens": collector.total_tokens,
        "tokens_per_domain": dict(collector.tokens_per_domain),
        "tokens_per_region": dict(collector.tokens_per_region),
        "tokens_per_thinking_bin": collector.tokens_per_thinking_bin,
        "activation_histograms_file": "activation_histograms.npz",
        "activation_histogram_summary": {
            "bin_lower_bounds": collector.hist_bin_lower_bounds.tolist(),
            "domain_names": collector.domain_names,
            "run_hist_total": collector.run_hist_total.astype(int).tolist(),
            "run_hist_by_domain": collector.run_hist_by_domain.astype(int).tolist(),
        },
        "feature_frequency_summary": build_feature_frequency_summary(collector),
        "relative_score_config": {
            "target_domain": target_domain,
            "baseline_domain": baseline_domain,
            "smoothing": "max(raw_density, 1 / domain_token_count) for target and baseline",
            "strength_quantiles": [0.95, 0.99],
        },
        "features": [],
    }

    for layer_idx in range(collector.n_layers):
        for feature_idx in range(collector.n_features):
            stats = collector.stats[layer_idx][feature_idx]
            if stats.activation_count == 0:
                continue
            metadata["features"].append(
                build_feature_metadata_entry(
                    collector=collector,
                    stats=stats,
                    layer_idx=layer_idx,
                    feature_idx=feature_idx,
                    target_domain=target_domain,
                    baseline_domain=baseline_domain,
                )
            )
    return metadata
```

- [ ] **Step 5: Simplify `export_metadata()` to use the helper**

Change `export_metadata` signature:

```python
def export_metadata(
    collector: FeatureCollector,
    output_dir: Path,
    target_domain: str,
    baseline_domain: str,
):
```

Replace its body with:

```python
    logger.info("Exporting metadata...")
    if target_domain not in collector.domain_names or baseline_domain not in collector.domain_names:
        logger.warning(
            "Relative scores skipped for missing domains: target=%r baseline=%r available=%s",
            target_domain,
            baseline_domain,
            collector.domain_names,
        )
    metadata = build_metadata_payload(
        collector=collector,
        target_domain=target_domain,
        baseline_domain=baseline_domain,
    )
    with open(output_dir / "feature_metadata.json", "w") as f:
        json.dump(metadata, f)
    logger.info(f"Saved metadata for {len(metadata['features'])} features")
```

- [ ] **Step 6: Add CLI arguments**

Add parser args:

```python
    parser.add_argument("--relative_target_domain", type=str, default="chat",
                        help="Target domain for relative feature scores")
    parser.add_argument("--relative_baseline_domain", type=str, default="fineweb",
                        help="Baseline domain for relative feature scores")
```

Update the export call:

```python
    export_metadata(
        collector,
        output_dir,
        target_domain=args.relative_target_domain,
        baseline_domain=args.relative_baseline_domain,
    )
```

- [ ] **Step 7: Run metadata tests**

Run:

```sh
uv run python -m unittest tests.test_activation_histograms
```

Expected: all tests pass.

- [ ] **Step 8: Run collector syntax check**

Run:

```sh
uv run python -m py_compile analysis/features/collect_feature_activations.py
```

Expected: no output and exit code 0.

- [ ] **Step 9: Commit metadata scoring**

Run:

```sh
git add analysis/features/collect_feature_activations.py tests/test_activation_histograms.py
git commit -m "feat: add relative domain feature scores"
```

## Task 4: Add Dashboard Histogram Endpoint

**Files:**
- Modify: `analysis/features/visualize/feature_dashboard.py`
- Create: `tests/test_feature_dashboard_histograms.py`

- [ ] **Step 1: Write payload loader tests**

Create `tests/test_feature_dashboard_histograms.py`:

```python
import json
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np

from analysis.features.activation_histograms import save_activation_histograms_npz
from analysis.features.visualize.feature_dashboard import (
    _load_feature_histogram_payload,
    make_handler_class,
)


class FeatureDashboardHistogramTests(unittest.TestCase):
    def test_load_feature_histogram_payload_returns_small_json_safe_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            bins = np.array([0.0, 1.0, 2.0], dtype=np.float32)
            save_activation_histograms_npz(
                path=data_dir / "activation_histograms.npz",
                bin_lower_bounds=bins,
                domain_names=["chat", "fineweb"],
                run_hist_total=np.array([10, 20, 30], dtype=np.uint64),
                run_hist_by_domain=np.array([[8, 9, 10], [2, 11, 20]], dtype=np.uint64),
                feature_hist_total_by_layer=[
                    np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint32)
                ],
                feature_hist_by_domain_by_layer=[
                    np.array(
                        [
                            [[1, 0, 0], [2, 0, 0]],
                            [[0, 2, 3], [0, 5, 6]],
                        ],
                        dtype=np.uint32,
                    )
                ],
            )

            payload = _load_feature_histogram_payload(
                data_dir=data_dir,
                histograms_file="activation_histograms.npz",
                feature_meta={"layer": 0, "feature": 1},
                tokens_per_domain={"chat": 100, "fineweb": 200},
                layer_cache={},
            )
            self.assertEqual(payload["bin_lower_bounds"], [0.0, 1.0, 2.0])
            self.assertEqual(payload["domain_names"], ["chat", "fineweb"])
            self.assertEqual(payload["total_counts"], [4, 5, 6])
            self.assertEqual(payload["by_domain_counts"]["chat"], [2, 0, 0])
            self.assertEqual(payload["by_domain_counts"]["fineweb"], [0, 5, 6])
            self.assertEqual(payload["by_domain_token_density"]["chat"], [0.02, 0.0, 0.0])
            self.assertAlmostEqual(payload["by_domain_activation_fraction"]["fineweb"][1], 5 / 11)
            json.dumps(payload)

    def test_feature_histogram_route_returns_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            (data_dir / "features").mkdir()
            (data_dir / "feature_metadata.json").write_text(json.dumps({
                "activation_histograms_file": "activation_histograms.npz",
                "tokens_per_domain": {"chat": 100, "fineweb": 200},
                "features": [{"cantor_id": 2, "layer": 0, "feature": 1}],
            }))
            save_activation_histograms_npz(
                path=data_dir / "activation_histograms.npz",
                bin_lower_bounds=np.array([0.0, 1.0, 2.0], dtype=np.float32),
                domain_names=["chat", "fineweb"],
                run_hist_total=np.array([10, 20, 30], dtype=np.uint64),
                run_hist_by_domain=np.array([[8, 9, 10], [2, 11, 20]], dtype=np.uint64),
                feature_hist_total_by_layer=[
                    np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint32)
                ],
                feature_hist_by_domain_by_layer=[
                    np.array(
                        [
                            [[1, 0, 0], [2, 0, 0]],
                            [[0, 2, 3], [0, 5, 6]],
                        ],
                        dtype=np.uint32,
                    )
                ],
            )

            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler_class(data_dir))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{server.server_port}/api/feature_hist/2",
                    timeout=5,
                ) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                self.assertEqual(payload["total_counts"], [4, 5, 6])
                self.assertIn("by_domain_token_density", payload)
            finally:
                server.shutdown()
                thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the dashboard test and confirm helper is missing**

Run:

```sh
uv run python -m unittest tests.test_feature_dashboard_histograms
```

Expected: import failure for `_load_feature_histogram_payload`.

- [ ] **Step 3: Add NumPy import and payload helper**

Modify `analysis/features/visualize/feature_dashboard.py`:

```python
import numpy as np
```

Add helper near `_load_annotations`:

```python
def _load_feature_histogram_payload(
    *,
    data_dir: Path,
    histograms_file: str,
    feature_meta: dict,
    tokens_per_domain: dict[str, int],
    layer_cache: dict[int, dict],
) -> dict:
    hist_path = data_dir / histograms_file
    if not hist_path.is_file():
        raise FileNotFoundError(f"Missing activation histogram sidecar: {hist_path}")

    layer_idx = int(feature_meta["layer"])
    feature_idx = int(feature_meta["feature"])
    if layer_idx not in layer_cache:
        with np.load(hist_path, allow_pickle=False) as data:
            layer_cache[layer_idx] = {
                "bin_lower_bounds": data["bin_lower_bounds"].astype(float),
                "domain_names": data["domain_names"].astype(str).tolist(),
                "feature_hist_total": data[f"feature_hist_total_layer_{layer_idx}"],
                "feature_hist_by_domain": data[f"feature_hist_by_domain_layer_{layer_idx}"],
            }
    layer_data = layer_cache[layer_idx]
    bins = layer_data["bin_lower_bounds"].tolist()
    domain_names = layer_data["domain_names"]
    total_arr = layer_data["feature_hist_total"][feature_idx]
    by_domain_arr = layer_data["feature_hist_by_domain"][:, feature_idx, :]
    by_domain_counts = {
        domain: by_domain_arr[i].astype(int).tolist()
        for i, domain in enumerate(domain_names)
    }
    by_domain_token_density = {}
    by_domain_activation_fraction = {}
    for i, domain in enumerate(domain_names):
        counts = by_domain_arr[i].astype(float)
        domain_tokens = int(tokens_per_domain.get(domain) or 0)
        domain_total = float(counts.sum())
        by_domain_token_density[domain] = (
            (counts / domain_tokens).tolist() if domain_tokens > 0 else [0.0] * len(counts)
        )
        by_domain_activation_fraction[domain] = (
            (counts / domain_total).tolist() if domain_total > 0 else [0.0] * len(counts)
        )
    return {
        "bin_lower_bounds": bins,
        "domain_names": domain_names,
        "total_counts": total_arr.astype(int).tolist(),
        "by_domain_counts": by_domain_counts,
        "by_domain_token_density": by_domain_token_density,
        "by_domain_activation_fraction": by_domain_activation_fraction,
    }
```

- [ ] **Step 4: Build feature metadata index in handler closure**

Inside `make_handler_class()`, before `class DashboardHandler`, add:

```python
    feature_index_by_cantor: dict[str, dict] = {}
    tokens_per_domain: dict[str, int] = {}
    histogram_layer_cache: dict[int, dict] = {}
    metadata_path = data_dir / "feature_metadata.json"
    histograms_file = "activation_histograms.npz"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text())
            histograms_file = metadata.get("activation_histograms_file") or histograms_file
            tokens_per_domain = metadata.get("tokens_per_domain") or {}
            feature_index_by_cantor = {
                str(feature["cantor_id"]): feature
                for feature in metadata.get("features") or []
            }
        except Exception as exc:
            logger.warning("Could not pre-load feature metadata index: %s", exc)
```

- [ ] **Step 5: Add GET endpoint**

In `do_GET()`, after `/api/feature/{cantor_id}`, add:

```python
            m = re.match(r"^/api/feature_hist/(\d+)$", path)
            if m:
                cantor_id = m.group(1)
                feature_meta = feature_index_by_cantor.get(cantor_id)
                if feature_meta is None:
                    _json_response(
                        self,
                        json.dumps({"error": f"No metadata for feature {cantor_id}"}).encode(),
                        status=404,
                    )
                    return
                try:
                    payload = _load_feature_histogram_payload(
                        data_dir=data_dir,
                        histograms_file=histograms_file,
                        feature_meta=feature_meta,
                        tokens_per_domain=tokens_per_domain,
                        layer_cache=histogram_layer_cache,
                    )
                    _json_response(self, json.dumps(payload).encode())
                except Exception as exc:
                    _json_response(
                        self,
                        json.dumps({"error": str(exc)}).encode(),
                        status=404,
                    )
                return
```

- [ ] **Step 6: Run dashboard endpoint tests**

Run:

```sh
uv run python -m unittest tests.test_feature_dashboard_histograms
```

Expected: all tests pass.

- [ ] **Step 7: Run dashboard syntax check**

Run:

```sh
uv run python -m py_compile analysis/features/visualize/feature_dashboard.py
```

Expected: no output and exit code 0.

- [ ] **Step 8: Commit dashboard endpoint**

Run:

```sh
git add analysis/features/visualize/feature_dashboard.py tests/test_feature_dashboard_histograms.py
git commit -m "feat: serve feature activation histograms"
```

## Task 5: Update Dashboard UI

**Files:**
- Modify: `analysis/features/visualize/dashboard.html`

- [ ] **Step 1: Add metadata-derived relative score fields to rows**

In `buildRows(meta)`, add `_relative`:

```javascript
      _relative: f.relative_domain_scores || null,
```

The returned row object should include both `_skew` and `_relative`.

- [ ] **Step 2: Add relative score table columns**

In `thead()`, change the base header:

```javascript
    let h = '<th>L</th><th>F</th><th>Cantor</th><th>Tags</th><th>Freq</th><th>Rel</th><th>P95x</th><th>P99x</th><th>Maxx</th><th>Skew</th>';
```

In `renderTable()`, add four cells before skew:

```javascript
        <td class="num">${f._relative && f._relative.log2_density_lift != null ? f._relative.log2_density_lift.toFixed(2) : '–'}</td>
        <td class="num">${f._relative && f._relative.p95_ratio != null ? f._relative.p95_ratio.toFixed(2) : '–'}</td>
        <td class="num">${f._relative && f._relative.p99_ratio != null ? f._relative.p99_ratio.toFixed(2) : '–'}</td>
        <td class="num">${f._relative && f._relative.max_nonzero_bin_ratio != null ? f._relative.max_nonzero_bin_ratio.toFixed(2) : '–'}</td>
```

- [ ] **Step 3: Add relative sorting options**

In the sort dropdown setup, add options before domain density options:

```javascript
      '<option value="rel">Relative density lift</option>',
      '<option value="p95_ratio">P95 activation ratio</option>',
      '<option value="p99_ratio">P99 activation ratio</option>',
      '<option value="max_ratio">Max nonzero-bin ratio</option>',
```

In `cmp(a, b)`, add:

```javascript
    else if (sortKey === 'rel') {
      va = a._relative && a._relative.log2_density_lift != null ? a._relative.log2_density_lift : -Infinity;
      vb = b._relative && b._relative.log2_density_lift != null ? b._relative.log2_density_lift : -Infinity;
    }
    else if (sortKey === 'p95_ratio') {
      va = a._relative && a._relative.p95_ratio != null ? a._relative.p95_ratio : -Infinity;
      vb = b._relative && b._relative.p95_ratio != null ? b._relative.p95_ratio : -Infinity;
    }
    else if (sortKey === 'p99_ratio') {
      va = a._relative && a._relative.p99_ratio != null ? a._relative.p99_ratio : -Infinity;
      vb = b._relative && b._relative.p99_ratio != null ? b._relative.p99_ratio : -Infinity;
    }
    else if (sortKey === 'max_ratio') {
      va = a._relative && a._relative.max_nonzero_bin_ratio != null ? a._relative.max_nonzero_bin_ratio : -Infinity;
      vb = b._relative && b._relative.max_nonzero_bin_ratio != null ? b._relative.max_nonzero_bin_ratio : -Infinity;
    }
```

- [ ] **Step 4: Show run-level histogram and feature-frequency overview from metadata**

Add helper functions near `formatScalar()`:

```javascript
  function renderHistogramBars(binLowerBounds, counts, labelPrefix) {
    const maxCount = counts.length ? Math.max(...counts, 1) : 1;
    return counts.map((count, i) => {
      const lo = binLowerBounds[i];
      const hi = i + 1 < binLowerBounds.length ? binLowerBounds[i + 1] : null;
      const label = hi == null ? `${labelPrefix}${formatScalar(lo)}+` : `${labelPrefix}${formatScalar(lo)}-${formatScalar(hi)}`;
      const width = Math.max(1, 100 * count / maxCount);
      return `<div class="bar-row"><span class="bar-label" title="${escapeHtml(label)}">${escapeHtml(label)}</span><div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div><span>${escapeHtml(formatScalar(count))}</span></div>`;
    }).join('');
  }

  function renderRelativeScatter(featureRows) {
    const pts = featureRows.filter((f) => f._relative && f._relative.target_density != null && f._relative.baseline_density != null);
    if (!pts.length) return '';
    const sample = pts.slice(0, 2500);
    const maxX = Math.max(...sample.map((f) => f._relative.baseline_density), 1e-12);
    const maxY = Math.max(...sample.map((f) => f._relative.target_density), 1e-12);
    const circles = sample.map((f) => {
      const x = 36 + 250 * Math.sqrt(Math.max(0, f._relative.baseline_density) / maxX);
      const y = 286 - 250 * Math.sqrt(Math.max(0, f._relative.target_density) / maxY);
      const lift = f._relative.log2_density_lift || 0;
      const fill = lift >= 0 ? '#58a6ff' : '#f85149';
      return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="2.2" fill="${fill}" opacity="0.55"><title>L${f.layer} F${f.feature} log2_lift=${formatScalar(lift)}</title></circle>`;
    }).join('');
    return `<h2>Joint domain distribution</h2>
      <div class="meta-mini">Each point is one feature. X = baseline-domain density, Y = target-domain density; blue means target-enriched.</div>
      <svg viewBox="0 0 320 310" width="100%" height="240" role="img">
        <line x1="36" y1="286" x2="302" y2="286" stroke="var(--border)" />
        <line x1="36" y1="286" x2="36" y2="20" stroke="var(--border)" />
        <line x1="36" y1="286" x2="286" y2="36" stroke="var(--muted)" opacity="0.45" />
        ${circles}
      </svg>`;
  }

  function renderFeatureFrequencySummary(meta) {
    const fs = meta.feature_frequency_summary;
    if (!fs || !fs.frequency_bin_lower_bounds) return '';
    let h = '<h2>Feature firing frequency distribution</h2>';
    h += '<div class="meta-mini">Histograms are over all features, including zero-firing features. Per-domain frequency is feature fires divided by tokens in that domain.</div>';
    h += renderHistogramBars(
      fs.frequency_bin_lower_bounds,
      fs.feature_frequency_hist_total || [],
      ''
    );
    const domains = fs.domain_names || [];
    const byDomain = fs.feature_frequency_hist_by_domain || {};
    const globalDensity = fs.global_nonzero_density_by_domain || {};
    const activeCounts = fs.active_feature_count_by_domain || {};
    for (const domain of domains) {
      const hist = byDomain[domain];
      if (!hist) continue;
      const density = globalDensity[domain] != null ? formatScalar(globalDensity[domain]) : '0';
      const active = activeCounts[domain] != null ? activeCounts[domain] : 0;
      h += `<h2>Feature firing frequency · ${escapeHtml(domain)}</h2>`;
      h += `<div class="meta-mini">Run-level nonzero token-feature density ${escapeHtml(density)}; active features ${escapeHtml(active)} / ${escapeHtml(fs.all_features_total || 0)}.</div>`;
      h += renderHistogramBars(fs.frequency_bin_lower_bounds, hist, '');
    }
    return h;
  }
```

In `renderOverview(meta)`, after validation mix, add:

```javascript
    const hs = meta.activation_histogram_summary;
    if (hs && hs.bin_lower_bounds && hs.run_hist_total) {
      h += '<h2>Nonzero activation histogram</h2>';
      h += renderHistogramBars(hs.bin_lower_bounds, hs.run_hist_total, '');
      const domains = hs.domain_names || [];
      const byDomain = hs.run_hist_by_domain || [];
      const tokens = meta.tokens_per_domain || {};
      for (let i = 0; i < domains.length; i++) {
        const domain = domains[i];
        const denom = tokens[domain] || 0;
        if (!denom || !byDomain[i]) continue;
        h += `<h2>Nonzero activation density · ${escapeHtml(domain)}</h2>`;
        h += renderHistogramBars(hs.bin_lower_bounds, byDomain[i].map((c) => c / denom), '');
      }
    }
    h += renderFeatureFrequencySummary(meta);
    h += renderRelativeScatter(rows);
```

- [ ] **Step 5: Add per-feature relative score and lazy histogram detail**

In `loadDetail(cantorId)`, after the feature pills, add:

```javascript
    if (metaRow && metaRow.relative_domain_scores) {
      const r = metaRow.relative_domain_scores;
      h += `<h2>Relative domain score</h2>
        <div class="logit-box">
          target ${escapeHtml(r.target_domain)} density ${escapeHtml(formatScalar(r.target_density))}<br/>
          baseline ${escapeHtml(r.baseline_domain)} density ${escapeHtml(formatScalar(r.baseline_density))}<br/>
          log2 lift ${escapeHtml(formatScalar(r.log2_density_lift))}<br/>
          target p95 ${escapeHtml(formatScalar(r.target_p95))}<br/>
          baseline p95 ${escapeHtml(formatScalar(r.baseline_p95))}<br/>
          p95 ratio ${escapeHtml(formatScalar(r.p95_ratio))}<br/>
          target p99 ${escapeHtml(formatScalar(r.target_p99))}<br/>
          baseline p99 ${escapeHtml(formatScalar(r.baseline_p99))}<br/>
          p99 ratio ${escapeHtml(formatScalar(r.p99_ratio))}<br/>
          target max bin ${escapeHtml(formatScalar(r.target_max_nonzero_bin))}<br/>
          baseline max bin ${escapeHtml(formatScalar(r.baseline_max_nonzero_bin))}<br/>
          max-bin ratio ${escapeHtml(formatScalar(r.max_nonzero_bin_ratio))}
        </div>`;
    }
```

Before `detail.innerHTML = h;`, add a placeholder if histograms exist:

```javascript
    const hasHistogramSidecar = rawMeta && rawMeta.activation_histograms_file;
    if (hasHistogramSidecar) {
      h += '<h2>Activation histograms</h2><div id="feature-histograms" class="meta-mini">Loading histograms...</div>';
    }
```

After `detail.innerHTML = h;`, add:

```javascript
    if (rawMeta && rawMeta.activation_histograms_file) {
      loadFeatureHistogram(cantorId);
    }
```

Add this function near `loadDetail`:

```javascript
  async function loadFeatureHistogram(cantorId) {
    const target = el('feature-histograms');
    if (!target) return;
    try {
      const res = await fetch('/api/feature_hist/' + cantorId);
      if (!res.ok) throw new Error(await res.text());
      const payload = await res.json();
      let html = '<h2>Total</h2>' + renderHistogramBars(payload.bin_lower_bounds, payload.total_counts, '');
      for (const domain of payload.domain_names || []) {
        const counts = payload.by_domain_counts ? payload.by_domain_counts[domain] : null;
        if (!counts) continue;
        html += `<h2>${escapeHtml(domain)} counts</h2>` + renderHistogramBars(payload.bin_lower_bounds, counts, '');
        const density = payload.by_domain_token_density ? payload.by_domain_token_density[domain] : null;
        if (density) html += `<h2>${escapeHtml(domain)} token density</h2>` + renderHistogramBars(payload.bin_lower_bounds, density, '');
        const fraction = payload.by_domain_activation_fraction ? payload.by_domain_activation_fraction[domain] : null;
        if (fraction) html += `<h2>${escapeHtml(domain)} conditional magnitude distribution</h2>` + renderHistogramBars(payload.bin_lower_bounds, fraction, '');
      }
      target.innerHTML = html;
    } catch (e) {
      target.innerHTML = `<span class="err">Histogram load failed: ${escapeHtml(e.message)}</span>`;
    }
  }
```

- [ ] **Step 6: Run dashboard static syntax smoke**

Run:

```sh
uv run python -m py_compile analysis/features/visualize/feature_dashboard.py
```

Expected: no output and exit code 0.

Extract the inline dashboard script and run a JavaScript syntax check:

```sh
uv run python -c "from pathlib import Path; import re; html = Path('analysis/features/visualize/dashboard.html').read_text(); match = re.search(r'<script>\\s*(.*?)\\s*</script>', html, re.S); assert match is not None; Path('/tmp/feature_dashboard_script.js').write_text(match.group(1))"
```

```sh
node --check /tmp/feature_dashboard_script.js
```

Expected: `node --check` reports no syntax errors and exits 0. Validate by opening the dashboard during Task 7 smoke if a small run is available.

- [ ] **Step 7: Commit dashboard UI**

Run:

```sh
git add analysis/features/visualize/dashboard.html
git commit -m "feat: show activation histograms in dashboard"
```

## Task 6: Update Docs

**Files:**
- Modify: `analysis/features/visualize/README.md`
- Modify: `analysis/README.md`
- Modify: `analysis/features/collect_feature_activations.py`

- [ ] **Step 1: Update collector docstring output tree**

In `analysis/features/collect_feature_activations.py`, change the output section to include:

```text
    ├── activation_histograms.npz  # Exact all-nonzero activation histograms
```

Add a sentence after the domain top-K paragraph:

```text
All positive activations are also aggregated into fixed activation-magnitude
histograms, globally and per domain. Dense per-feature histogram arrays are
stored in activation_histograms.npz; feature_metadata.json stores scalar
summaries for sorting and dashboard display. Dashboard histogram views show raw
counts, token-normalized density by domain, and conditional magnitude
distributions by domain. Metadata also includes per-feature firing-frequency
histograms and global nonzero token-feature density by domain, so run-level
chat-vs-web sparsity differences are visible. Configured activation bands are
reservoir-sampled into feature JSON example tabs such as Activation range
2.5-3.0 (chat), so medium non-top activations can be inspected directly.
```

- [ ] **Step 2: Update visualization README prerequisites**

In `analysis/features/visualize/README.md`, update the artifact list:

```markdown
- `activation_histograms.npz` — exact all-nonzero activation magnitude histograms for new runs
```

Update the detail description to include:

```markdown
- **Histograms and activation-range examples:** new collection runs show global and per-domain activation magnitude distributions, feature firing-frequency distributions across all features, normalized per-domain activation densities, conditional per-feature magnitude distributions, run-level nonzero token-feature density by domain, and a joint target-vs-baseline feature-density scatter. Feature detail pages also expose bounded `Activation range ...` example tabs for configured bands such as `2.5:3.0`. Older runs without `activation_histograms.npz` still load, but histogram sections are hidden.
```

- [ ] **Step 3: Update analysis README outputs**

In `analysis/README.md`, update Step 1 outputs:

```markdown
- `activation_histograms.npz` — exact nonzero activation histograms, globally and per feature/domain; the dashboard derives count, token-density, and conditional-distribution views from this sidecar
- `feature_metadata.json` — includes `feature_frequency_summary` with per-feature firing-frequency histograms, including zero-firing features, plus run-level nonzero token-feature density by domain
- `features/{cantor_id}.json` — feature examples, including top activations, per-domain top activations, random samples, and bounded activation-range tabs for medium/non-top activation inspection
```

- [ ] **Step 4: Run docs grep sanity**

Run:

```sh
rg -n "activation_histograms|feature_frequency_summary|relative_target_domain|relative_baseline_domain|activation_example_ranges|activation_range_examples_per_domain" analysis/README.md analysis/features/visualize/README.md analysis/features/collect_feature_activations.py
```

Expected: all three files mention the new histogram artifact and feature-frequency summary; the collector mentions both relative-domain CLI flags and both activation-range CLI flags.

- [ ] **Step 5: Commit docs**

Run:

```sh
git add analysis/README.md analysis/features/visualize/README.md analysis/features/collect_feature_activations.py
git commit -m "docs: document activation histogram artifacts"
```

## Task 7: Final Verification

**Files:**
- Verify: all changed files

- [ ] **Step 1: Run focused unit tests**

Run:

```sh
uv run python -m unittest tests.test_activation_histograms tests.test_feature_dashboard_histograms
```

Expected: all tests pass.

- [ ] **Step 2: Verify frozen dependency resolution**

Run:

```sh
uv run --frozen python -c "import numpy; assert numpy.__version__"
```

Expected: no output and exit code 0.

- [ ] **Step 3: Run full repo tests**

Run:

```sh
uv run python -m unittest discover -s tests
```

Expected: all tests pass. If unrelated tests fail, record the failing test names and the failure text before deciding whether the implementation is responsible.

- [ ] **Step 4: Run syntax checks**

Run:

```sh
uv run python -m py_compile \
  analysis/features/activation_histograms.py \
  analysis/features/collect_feature_activations.py \
  analysis/features/visualize/feature_dashboard.py
```

Expected: no output and exit code 0.

- [ ] **Step 5: Run dashboard JavaScript syntax check**

Run:

```sh
uv run python -c "from pathlib import Path; import re; html = Path('analysis/features/visualize/dashboard.html').read_text(); match = re.search(r'<script>\\s*(.*?)\\s*</script>', html, re.S); assert match is not None; Path('/tmp/feature_dashboard_script.js').write_text(match.group(1))"
```

```sh
node --check /tmp/feature_dashboard_script.js
```

Expected: no syntax errors and exit code 0.

- [ ] **Step 6: Optional GPU smoke for exact collector output**

Run this only on a GPU-capable environment with model access:

```sh
uv run python -m analysis.features.collect_feature_activations \
  --model_path siddharthmb/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860 \
  --val_data chat:siddharthmb/2026.transcoder-adapters.lmsys-chat-1m-splits fineweb:science-of-finetuning/fineweb-1m-sample \
  --max_samples 1 \
  --batch_size 1 \
  --top_k 1 \
  --domain_top_k 1 \
  --n_random 1 \
  --activation_example_ranges 0.0:1000000 \
  --activation_range_examples_per_domain 1 \
  --output_dir /tmp/feature_hist_smoke
```

Expected:

- `/tmp/feature_hist_smoke/feature_metadata.json` exists.
- `/tmp/feature_hist_smoke/activation_histograms.npz` exists.
- `/tmp/feature_hist_smoke/feature_metadata.json` contains `feature_frequency_summary`.
- At least one feature metadata entry contains `relative_domain_scores` when both domains produced activations.
- At least one feature JSON contains an `examples_quantiles` entry whose `quantile_name` starts with `Activation range`. The wide `0.0:1000000` smoke range makes this deterministic for any positive activation.

- [ ] **Step 7: Optional dashboard smoke**

If the optional smoke run succeeded, run:

```sh
uv run python -m analysis.features.visualize.feature_dashboard \
  --data_dir /tmp/feature_hist_smoke \
  --no-open \
  --port 8765
```

Expected:

- `GET /api/metadata` returns metadata with `activation_histograms_file`.
- `GET /api/feature_hist/{cantor_id}` returns `bin_lower_bounds`, `total_counts`, `by_domain_counts`, `by_domain_token_density`, and `by_domain_activation_fraction`.
- The dashboard table renders relative columns, the overview renders feature firing-frequency histograms, run-level nonzero token-feature density by domain, and the joint target-vs-baseline scatter, and selecting a feature renders histogram bars plus any `Activation range ...` example tabs already present in that feature JSON.

- [ ] **Step 8: Inspect final diff**

Run:

```sh
git diff --stat HEAD
git diff HEAD -- pyproject.toml uv.lock analysis/features/activation_histograms.py analysis/features/collect_feature_activations.py analysis/features/visualize/feature_dashboard.py analysis/features/visualize/dashboard.html analysis/README.md analysis/features/visualize/README.md tests/test_activation_histograms.py tests/test_feature_dashboard_histograms.py
```

Expected: diff is limited to histogram collection, bounded activation-range example sampling, metadata scoring, dashboard serving/display, docs, and tests.

## Implementation Notes

- Do not use `print()`; use `logger.info`, `logger.warning`, or `logger.error`.
- Use `uv run python` for Python commands.
- Store dense per-feature histogram arrays only in `activation_histograms.npz`.
- Keep dashboard boot metadata-only; do not fetch every feature JSON or every per-feature histogram at startup.
- Existing feature-data runs without `activation_histograms.npz` must continue to load with current dashboard behavior.
- Keep `feature_annotations.json` behavior unchanged.
- Keep `features/{cantor_id}.json` compatible with `pack_features.py`; no dense histogram arrays go into those JSON files.
- Keep activation-range examples bounded by `activation_range_examples_per_domain`; do not store every medium activation as text examples.
