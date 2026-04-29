#!/usr/bin/env bash

# ── Usage ────────────────────────────────────────────────────────────
#   Run ./sh/slurm_batch_sweep.sh, or sbatch with env vars set manually.
#
# Logs: logs/sweep/<timestamp>_<job_id>.{out,err}
# ─────────────────────────────────────────────────────────────────────

SLURM_LOG_DIR="logs/sweep"
source run_on_gpu/common.sh

run run_on_gpu/run_sweep.sh \
    --config training/configs/gemma2_2b.yaml \
    --sweep training/configs/sweeps/lr.yaml \
    --sweep_count 4 \
    "$@"
