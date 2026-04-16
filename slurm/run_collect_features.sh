#!/usr/bin/env bash

# ── Usage ────────────────────────────────────────────────────────────
#   ./sh/slurm_batch_collect_features.sh
#   Or: sbatch --export=ALL,HF_TOKEN=... ./slurm/run_collect_features.sh [args]
#
# Logs: logs/collect_features/<job_id>_<timestamp>.{out,err}
# ─────────────────────────────────────────────────────────────────────

SLURM_LOG_DIR="logs/collect_features"
source slurm/common.sh

run uv run python -m analysis.features.collect_feature_activations "$@"
