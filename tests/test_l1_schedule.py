"""Unit tests for L1-weight warmup scheduling (training.train.scheduled_l1_weight).

The L1 sparsity coefficient can be linearly warmed up from 0 to its target over the first
``warmup_ratio`` fraction of training (transcoder.schedule_l1_weight). This lets features learn
to reconstruct before the sparsity penalty bites, reducing dead features. These tests pin the
ramp shape and the off/edge cases without needing a real model or training loop.
"""

from types import SimpleNamespace

import pytest

from training.train import scheduled_l1_weight


def _cfg(l1, schedule, warmup_ratio=0.05):
    return SimpleNamespace(
        transcoder=SimpleNamespace(l1_weight=l1, schedule_l1_weight=schedule),
        warmup_ratio=warmup_ratio,
    )


def test_constant_when_scheduling_off():
    cfg = _cfg(0.0003, schedule=False)
    assert scheduled_l1_weight(cfg, 0, 100_000) == 0.0003
    assert scheduled_l1_weight(cfg, 99_999, 100_000) == 0.0003


def test_linear_ramp_to_target():
    # warmup_ratio 0.05 over 100k steps -> warmup_steps = 5000.
    cfg = _cfg(0.0003, schedule=True)
    assert scheduled_l1_weight(cfg, 0, 100_000) == 0.0  # no penalty at the start
    assert scheduled_l1_weight(cfg, 2_500, 100_000) == pytest.approx(0.00015)  # halfway
    assert scheduled_l1_weight(cfg, 5_000, 100_000) == pytest.approx(0.0003)  # reaches target
    assert scheduled_l1_weight(cfg, 50_000, 100_000) == pytest.approx(0.0003)  # holds after warmup


def test_no_penalty_when_l1_none():
    cfg = _cfg(None, schedule=True)
    assert scheduled_l1_weight(cfg, 100, 100_000) == 0.0


def test_short_run_warmup_at_least_one_step():
    # total_steps small enough that warmup_ratio rounds to 0; max(1, ...) keeps it well-defined.
    cfg = _cfg(0.0003, schedule=True)
    assert scheduled_l1_weight(cfg, 0, 5) == 0.0
    assert scheduled_l1_weight(cfg, 1, 5) == pytest.approx(0.0003)  # warmup_steps clamped to 1
