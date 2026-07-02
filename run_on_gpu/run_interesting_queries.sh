#!/usr/bin/env bash
# Created by Claude Code session "explore: queries to analyze".
#
# Run one of the interesting-query finders on GPU.
# Usage (via sbatch): ./run_on_gpu/run_interesting_queries.sh <script.py> [args...]
#   e.g. sbatch ... ./run_on_gpu/run_interesting_queries.sh experiments/interesting_queries/scripts/find_harmful.py --n_select 18
#
# Logs: logs/interesting_queries/<timestamp>_<job_id>.{out,err}

SLURM_LOG_DIR="logs/interesting_queries"
source run_on_gpu/common.sh

export HF_TOKEN="${HF_TOKEN:-$(cat ~/.shell/secrets/hf_token_write 2>/dev/null)}"

SCRIPT="$1"; shift
# find_adv_suffix needs nanoGCG (additive install into the shared venv; --no-sync run below won't prune it).
if [[ "$SCRIPT" == *find_adv_suffix.py ]]; then
  echo "[Slurm] Installing nanogcg (additive)"
  uv pip install nanogcg || echo "[Slurm] nanogcg install failed; finder will fall back to published suffixes"
fi

run uv run --no-sync python "$SCRIPT" "$@"
