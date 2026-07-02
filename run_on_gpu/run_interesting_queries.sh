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
# find_adv_suffix needs nanoGCG. Install it into an EPHEMERAL uv overlay (`--with`) rather than the
# shared .venv: `uv pip install nanogcg` downgrades transformers/tokenizers in the network venv and
# breaks concurrent jobs. The overlay layers on top without mutating the base venv. common.load_models
# is dtype-robust so it works whether nanoGCG's pins pull an older transformers into the overlay.
if [[ "$SCRIPT" == *find_adv_suffix.py ]]; then
  run uv run --no-sync --with nanogcg python "$SCRIPT" "$@"
else
  run uv run --no-sync python "$SCRIPT" "$@"
fi
