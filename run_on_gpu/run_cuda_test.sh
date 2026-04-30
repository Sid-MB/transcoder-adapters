#!/usr/bin/env bash

# ── Usage ────────────────────────────────────────────────────────────
#   Run ./sh/slurm_batch_cuda_test.sh, or sbatch with env vars set manually.
#
# Logs: logs/cuda_test/<timestamp>_<job_id>.{out,err}
# ─────────────────────────────────────────────────────────────────────

SLURM_LOG_DIR="logs/cuda_test"
source run_on_gpu/common.sh

run uv run python misc_scripts/cuda_test.py "$@"
