#!/usr/bin/env bash

# ── Usage ────────────────────────────────────────────────────────────
#   ./sh/slurm_batch_token_metrics.sh --model <model_path> [--transcoder] [...]
#   Or: sbatch --export=ALL,HF_TOKEN=... ./run_on_gpu/run_token_metrics.sh [args]
#
# Logs: logs/token_metrics/<job_id>_<timestamp>.{out,err}
# ─────────────────────────────────────────────────────────────────────

SLURM_LOG_DIR="logs/token_metrics"
source run_on_gpu/common.sh

run uv run python -m analysis.evals.compute_token_metrics_onpolicy "$@"
