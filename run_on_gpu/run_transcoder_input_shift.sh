#!/usr/bin/env bash

# ── Usage ────────────────────────────────────────────────────────────
#   ./sh/slurm_batch_transcoder_input_shift.sh [args]
#   Or: sbatch --export=ALL,HF_TOKEN=... ./run_on_gpu/run_transcoder_input_shift.sh [args]
#
# Measures GemmaScope transcoder L0 / reconstruction error when the input hidden states
# come from the base model vs. the instruction-tuned model (Experiment 1; see
# analysis/features/transcoder_input_shift.py and plan cached-doodling-horizon.md).
#
# Logs: logs/transcoder_input_shift/<timestamp>_<job_id>.{out,err}
# ─────────────────────────────────────────────────────────────────────

SLURM_LOG_DIR="logs/transcoder_input_shift"
source run_on_gpu/common.sh

# Ensure the viz extra (circuit_tracer) is present without pruning anything else, then run
# with --no-sync so we never mutate the shared venv at runtime (see common.sh).
uv sync --extra viz --inexact
run uv run --no-sync python -m analysis.features.transcoder_input_shift "$@"
