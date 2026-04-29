#!/usr/bin/env bash

# ── Usage ────────────────────────────────────────────────────────────
#   Run ./sh/slurm_batch_train.sh, or sbatch with env vars set manually.
#
# Logs: logs/train/<job_id>_<timestamp>.{out,err}
# ─────────────────────────────────────────────────────────────────────

SLURM_LOG_DIR="logs/train"
source run_on_gpu/common.sh

# uv run python -m training.train --config training/configs/gemma2_2b.yaml "$@"
run uv run python -m training.train "$@"
