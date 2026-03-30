#!/usr/bin/env bash

# ── Usage ────────────────────────────────────────────────────────────
#   ./sh/slurm_batch_collect_features.sh
#   Or: sbatch --export=ALL,HF_TOKEN=... ./slurm/run_collect_features.sh [args]
#
# Logs: logs/collect_features/<job_id>_<timestamp>.{out,err}
# ─────────────────────────────────────────────────────────────────────

SLURM_LOG_DIR="logs/collect_features"
source slurm/common.sh

run uv run python -m analysis.features.collect_feature_activations \
    --model_path siddharthmb/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr10000_lr8e-04_bs4_sl14754432 \
    --val_data siddharthmb/2026.transcoder-adapters.lmsys-chat-1m-splits \
    "$@"
