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

# ── Robust LARGE_ARTIFACTS_DIR ─────────────────────────────────────────
# helpers.paths requires LARGE_ARTIFACTS_DIR. It is normally set by ~/.shell/set-vars.sh,
# but that only runs in interactive bash (sourced from ~/.bashrc) -- NOT in Slurm batch
# scripts (non-interactive, non-login shells), NOT in non-interactive submitting shells,
# and it was empirically lost on requeue. Relying on --export=ALL to copy it from the
# submitter is therefore fragile. Set the same default set-vars.sh would (/nlp/scr/$USER)
# when it is missing, so every job -- including requeued ones -- has it.
export LARGE_ARTIFACTS_DIR="${LARGE_ARTIFACTS_DIR:-/nlp/scr/$USER}"
echo "[Slurm] LARGE_ARTIFACTS_DIR=$LARGE_ARTIFACTS_DIR"

# ── Writable UV_CACHE_DIR ──────────────────────────────────────────────
# `uv` needs a writable cache dir just to START. It's usually inherited (via set-vars.sh) as a
# node-local /scr/$USER path, but sc-loprio spans clusters where /scr/$USER may not exist or be
# writable -> `uv run` dies with "Failed to initialize cache ... Permission denied" before any
# work begins (this failed ~10/64 shards). If the inherited path isn't creatable, fall back to
# an always-writable node-local TMPDIR/tmp path (cheap: `uv run --no-sync` barely uses it).
if [ -z "${UV_CACHE_DIR:-}" ] || ! mkdir -p "$UV_CACHE_DIR" 2>/dev/null; then
    export UV_CACHE_DIR="${TMPDIR:-/tmp}/uv-cache-$USER"
    mkdir -p "$UV_CACHE_DIR" 2>/dev/null || true
fi
echo "[Slurm] UV_CACHE_DIR=$UV_CACHE_DIR"

# ── Logging ───────────────────────────────────────────────────────────
LOG_DIR="$SLURM_LOG_DIR"
LOG_PREFIX="${SLURM_JOB_ID:-local}"
source sh/common_logging.sh

# ── Setup ─────────────────────────────────────────────────────────────
echo "[Slurm] Setting up (uv sync)..."
# --inexact: do NOT remove extraneous packages. Jobs share one .venv on the network
# filesystem; a plain `uv sync` prunes packages outside the default set (e.g. the `viz`
# extra / circuit_tracer), which rips modules out from under other jobs running on the
# same venv concurrently. --inexact installs what's missing without pruning, so concurrent
# jobs don't break each other. Run scripts additionally pass `uv run --no-sync`.
uv sync --inexact

# ── Run helper ────────────────────────────────────────────────────────
run() {
    echo "[Slurm] Running: $*"
    set -xe
    "$@"
    set +xe
    echo "[Slurm] Job finished!"
}
