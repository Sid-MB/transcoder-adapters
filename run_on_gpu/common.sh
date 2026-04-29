#!/usr/bin/env bash
#
# Shared SLURM job boilerplate. Source this from job scripts.
#
# Usage (in a job script):
#   SLURM_LOG_DIR="logs/train"   # subfolder within logs/
#   source run_on_gpu/common.sh
#   run uv run python -m training.train --config training/configs/gemma2_2b.yaml "$@"
#
# Provides:
#   - Log directory setup (stdout/stderr redirected to SLURM_LOG_DIR/)
#   - uv sync
#   - run() function that executes a command with set -xe

# Assert that we're in the root of the repo (where run_on_gpu/ is) for consistent log paths and uv sync
if [ ! -d "run_on_gpu" ]; then
    echo "ERROR: common.sh must be sourced from the root of the repository (the parent folder of run_on_gpu/), so that log paths, Python calls and other run commands are consistent" >&2
    exit 1
fi

# ── Validate ──────────────────────────────────────────────────────────
if [ -z "$SLURM_LOG_DIR" ]; then
    echo "ERROR: SLURM_LOG_DIR must be set before sourcing common.sh" >&2
    exit 1
fi

# ── Logging ───────────────────────────────────────────────────────────
mkdir -p "$SLURM_LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="${SLURM_LOG_DIR}/${SLURM_JOB_ID:-local}_${TIMESTAMP}"
exec > >(tee "${LOGFILE}.out") 2> >(tee "${LOGFILE}.err" >&2)

# ── Setup ─────────────────────────────────────────────────────────────
echo "[Slurm] Setting up (uv sync)..."
uv sync

# ── Run helper ────────────────────────────────────────────────────────
run() {
    echo "[Slurm] Running: $*"
    set -xe
    "$@"
    set +xe
    echo "[Slurm] Job finished!"
}
