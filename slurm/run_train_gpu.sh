#!/usr/bin/env bash

# ── Usage ────────────────────────────────────────────────────────────
#   Run ./slurm_batch.sh, or sbatch with env vars set manually.
#
# Logs: logs/train/<job_id>_<timestamp>.{out,err}
# ─────────────────────────────────────────────────────────────────────

SLURM_LOG_DIR="logs/train"
source slurm/common.sh

# uv run python -m training.train --config training/configs/gemma2_9b.yaml "$@"
run uv run python -m training.train "$@"
