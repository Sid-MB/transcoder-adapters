#!/usr/bin/env bash

# ── Usage ────────────────────────────────────────────────────────────
#   ./sh/slurm_batch_attribution.sh \
#     --checkpoint <gemma2-or-qwen2-transcoder-checkpoint> \
#     --run_name gemma2_relp_smoke \
#     --prompts <prompt.txt-or-directory> \
#     --output_dir <output-dir> \
#     --max_feature_nodes 64 \
#     --batch_size 4
#
#   Or: sbatch --export=ALL,HF_TOKEN=... ./run_on_gpu/run_attribution.sh [args]
#
# Logs: logs/attribution/<job_id>_<timestamp>.{out,err}
#
# Attribution uses circuit-tracer, which is declared in the project's "viz"
# optional dependency group, so this runner uses `uv run --extra viz`.
# ─────────────────────────────────────────────────────────────────────

SLURM_LOG_DIR="logs/attribution"
source run_on_gpu/common.sh

run uv run --extra viz python -m analysis.attribution.run_attribution "$@"
