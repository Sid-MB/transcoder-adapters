#!/usr/bin/env bash

# ── Usage ────────────────────────────────────────────────────────────
#   ./sh/slurm_batch_collect_base_features.sh [args]
#   Or: sbatch --export=ALL,HF_TOKEN=... ./run_on_gpu/run_collect_base_features.sh [args]
#
# Collects BASE-model (GemmaScope) transcoder feature activations on our data
# (see analysis/features/collect_base_feature_activations.py).
#
# Logs: logs/collect_base_features/<timestamp>_<job_id>.{out,err}
# ─────────────────────────────────────────────────────────────────────

SLURM_LOG_DIR="logs/collect_base_features"
source run_on_gpu/common.sh

# Ensure the viz extra (circuit_tracer) is present without pruning anything else, then run
# with --no-sync so we never mutate the shared venv at runtime (avoids the concurrent-job
# prune race; see common.sh).
uv sync --extra viz --inexact
run uv run --no-sync python -m analysis.features.collect_base_feature_activations "$@"
