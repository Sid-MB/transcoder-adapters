#!/bin/bash
#SBATCH --account=nlp
#SBATCH --gres=gpu:1
#SBATCH --constraint=48G
#SBATCH --mem=128G
#SBATCH --partition=jag-standard
#SBATCH --job-name=gemma2_2b
#SBATCH --time=21-00:00:00

# ── Usage ────────────────────────────────────────────────────────────
#   Run ./slurm_batch. Or you can do `sbatch slurm_run_debug.sh`, but
#   will need to set up env variables yourself in that case.
# ─────────────────────────────────────────────────────────────────────

# ── Logs & Outputs ──────────────────────────────────────────────────
#
# SLURM stdout/stderr:
#   logs/<job_id>_<timestamp>.out   and   logs/<job_id>_<timestamp>.err
#   (location from repo root)
#
#   To find your job ID after submitting:
#     squeue --me
#
#   To tail logs of a running job:
#     tail -f logs/<job_id>_*.out
#
# Training checkpoints & wandb artifacts:
#   Written to the output directory configured in the YAML config
#   (defaults are printed at the start of training in stdout).
# ─────────────────────────────────────────────────────────────────────

mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="logs/${SLURM_JOB_ID}_${TIMESTAMP}"
exec > "${LOGFILE}.out" 2> "${LOGFILE}.err"

echo "[Slurm] Setting up (uv sync)..."

uv sync

echo "[Slurm] Running Python..."

set -x
# uv run python -m training.train --config training/configs/gemma2_9b.yaml "$@"
uv run python -m training.train --config training/configs/gemma2_2b.yaml "$@"
set +x

echo "[Slurm] Job finished!"
