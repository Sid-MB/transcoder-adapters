#!/usr/bin/env bash

# ── Usage ────────────────────────────────────────────────────────────
#   ./sh/slurm_batch_collect_features.sh
#   Or: sbatch --export=ALL,HF_TOKEN=... ./run_on_gpu/run_collect_features.sh [args]
#
# Logs: logs/collect_features/<timestamp>_<job_id>.{out,err}
# ─────────────────────────────────────────────────────────────────────

SLURM_LOG_DIR="logs/collect_features"
source run_on_gpu/common.sh

# --no-sync: use the venv as set up by common.sh (uv sync --inexact) without re-syncing,
# so concurrent jobs sharing the venv don't prune each other's packages mid-run.
run uv run --no-sync python -m analysis.features.collect_feature_activations "$@"
