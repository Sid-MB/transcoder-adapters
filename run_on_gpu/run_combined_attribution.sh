#!/usr/bin/env bash

# ── Usage ────────────────────────────────────────────────────────────
#   sbatch --export=ALL,HF_TOKEN=... ./run_on_gpu/run_combined_attribution.sh [args]
#   (or via ./sh/sbatch ... ./run_on_gpu/run_combined_attribution.sh [args])
#
# Combined full-replacement base+adapter attribution (real error nodes in one graph;
# see analysis/attribution/run_combined_attribution.py). Runs on jagupard (jag-standard).
#
# Logs: logs/combined_attribution/<timestamp>_<job_id>.{out,err}
# ─────────────────────────────────────────────────────────────────────

SLURM_LOG_DIR="logs/combined_attribution"
source run_on_gpu/common.sh

# Ensure the viz extra (circuit_tracer) is present without pruning anything else, then run
# with --no-sync so concurrent jobs sharing the venv don't prune each other's packages.
uv sync --extra viz --inexact
run uv run --no-sync python -m analysis.attribution.run_combined_attribution "$@"
