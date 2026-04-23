#!/usr/bin/env bash

# ── Usage ────────────────────────────────────────────────────────────
#   ./sh/slurm_batch_response_length_eval.sh --model <model_path> [--transcoder] [...]
#   Or: sbatch --export=ALL,HF_TOKEN=... ./slurm/run_response_length_eval.sh [args]
#
# Logs: logs/response_length_eval/<job_id>_<timestamp>.{out,err}
# ─────────────────────────────────────────────────────────────────────

SLURM_LOG_DIR="logs/response_length_eval"
source slurm/common.sh

run uv run python -m analysis.evals.response_length_eval "$@"
