#!/bin/bash
#SBATCH --account=nlp
#SBATCH --gres=gpu:4
#SBATCH --constraint=48G
#SBATCH --mem=128G
#SBATCH --partition=jag-hi
#SBATCH --job-name=gemma2_2b
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# ── Usage ────────────────────────────────────────────────────────────
#   Run ./slurm_batch. Or you can do `sbatch slurm_run_debug.sh`, but
#   will need to set up env variables yourself in that case.
# ─────────────────────────────────────────────────────────────────────

# ── Logs & Outputs ──────────────────────────────────────────────────
#
# SLURM stdout/stderr:
#   logs/<job_id>.out   and   logs/<job_id>.err
#   (location from repo root)
#
#   To find your job ID after submitting:
#     squeue --me
#
#   To tail logs of a running job:
#     tail -f logs/<job_id>.out
#
# Training checkpoints & wandb artifacts:
#   Written to the output directory configured in the YAML config
#   (defaults are printed at the start of training in stdout).
# ─────────────────────────────────────────────────────────────────────

mkdir -p logs

echo "[Slurm] Setting up (uv sync)..."

uv sync

echo "[Slurm] Running Python..."

# Detect GPU count: SLURM env var, nvidia-smi fallback, or default to 1
GPUS=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}
GPUS=${GPUS:-1}
# Trim whitespace (wc -l may produce leading spaces)
GPUS=$(echo "$GPUS" | tr -d '[:space:]')

CONFIG="training/configs/gemma2_2b.yaml"

if [ "$GPUS" -gt 1 ]; then
    echo "[Slurm] Using torchrun with $GPUS GPUs (data parallel)"
    uv run torchrun --nproc_per_node="$GPUS" -m training.train --config "$CONFIG"
else
    echo "[Slurm] Single GPU mode"
    uv run python -m training.train --config "$CONFIG"
fi

echo "[Slurm] Job finished!"
