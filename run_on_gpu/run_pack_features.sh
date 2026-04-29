#!/usr/bin/env bash

# ── Usage ────────────────────────────────────────────────────────────
#   ./sh/slurm_batch_pack_features.sh
#   Or: sbatch --export=ALL,HF_TOKEN=... ./run_on_gpu/run_pack_features.sh [args]
#
# Logs: logs/pack_features/<job_id>_<timestamp>.{out,err}
# ─────────────────────────────────────────────────────────────────────

SLURM_LOG_DIR="logs/pack_features"
source run_on_gpu/common.sh

# Output directory from a collect_feature_activations.py run
FEATURE_RUN="/nlp/scr/siddharth/sparse-adaptation/feature_data/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr10000_lr8e-04_bs4_sl147_20260309_191634_14778816"

run uv run python -m analysis.features.pack_features \
    --feature_dir "${FEATURE_RUN}/features" \
    --output_dir "${FEATURE_RUN}/packed_features" \
    --n_layers 26 \
    --n_features 8192 \
    "$@"
