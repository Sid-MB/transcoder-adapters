#!/usr/bin/env bash

# ── Usage ────────────────────────────────────────────────────────────
#   Run ./sh/slurm_batch_sweep.sh, or sbatch with env vars set manually.
#
# Logs: logs/sweep/<job_id>_<timestamp>.{out,err}
# ─────────────────────────────────────────────────────────────────────

SLURM_LOG_DIR="logs/sweep"
source slurm/common.sh

run slurm/run_sweep.sh \
    --config training/configs/gemma2_2b.yaml \
    --sweep training/configs/sweeps/lr.yaml \
    --sweep_count 4 \
    "$@"
