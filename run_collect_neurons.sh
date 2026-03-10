#!/usr/bin/env bash

# ── Usage ────────────────────────────────────────────────────────────
#   ./slurm_batch_collect_neurons.sh
#   Or: sbatch --export=ALL,HF_TOKEN=... ./run_collect_neurons.sh [args]
#
# Logs: logs/collect_neurons/<job_id>_<timestamp>.{out,err}
# ─────────────────────────────────────────────────────────────────────

SLURM_LOG_DIR="logs/collect_neurons"
source slurm/common.sh

run uv run python -m analysis.features.collect_neuron_activations \
    --model_path siddharthmb/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr10000_lr8e-04_bs4_sl14754432 \
    --val_data siddharthmb/2026.transcoder-adapters.lmsys-chat-1m-splits \
    "$@"
