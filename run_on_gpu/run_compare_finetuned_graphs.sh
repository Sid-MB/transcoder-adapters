#!/usr/bin/env bash
# Circuit-tracer graphs: original vs re-fine-tuned GemmaScope transcoders (side-by-side).
# See analysis/attribution/compare_finetuned_transcoder_graphs.py.
# Logs: logs/compare_finetuned_graphs/<timestamp>_<job_id>.{out,err}
SLURM_LOG_DIR="logs/compare_finetuned_graphs"
source run_on_gpu/common.sh
uv sync --extra viz --inexact
run uv run --no-sync python -m analysis.attribution.compare_finetuned_transcoder_graphs "$@"
