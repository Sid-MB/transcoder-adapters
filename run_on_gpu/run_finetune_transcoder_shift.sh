#!/usr/bin/env bash

# ── Usage ────────────────────────────────────────────────────────────
#   ./sh/slurm_batch_finetune_transcoder_shift.sh [args]
#
# Re-fine-tunes GemmaScope transcoders to the instruct input distribution and reports
# FVU/L0 before vs after (see analysis/features/finetune_transcoder_shift.py).
# Logs: logs/finetune_transcoder_shift/<timestamp>_<job_id>.{out,err}
# ─────────────────────────────────────────────────────────────────────

SLURM_LOG_DIR="logs/finetune_transcoder_shift"
source run_on_gpu/common.sh
uv sync --extra viz --inexact
run uv run --no-sync python -m analysis.features.finetune_transcoder_shift "$@"
