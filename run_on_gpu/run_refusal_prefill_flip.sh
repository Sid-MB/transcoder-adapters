#!/usr/bin/env bash
#
# [refusal-token-tracing] (session fd1f0d19-f5d1-48cc-9002-3f06f2abe2fd), 2026-08-10
#
# Exp 1 of the "which token loads the refusal?" study: transplant the instruct model's refusal opening
# ("I", "I cannot", ...) onto the BASE model and see whether base then refuses. Cheap (no attribution
# graphs) -- base + instruct (2b each) + Qwen3-8B judge all fit on one GPU.
#
# Run on sphinx (user-approved) or jagupard:
#   ./sh/sbatch --gres=gpu:1 --constraint=48G --mem=128G --cpus-per-task=8 --partition=sphinx \
#     --job-name='[refusal-token-tracing] prefill_flip' ./run_on_gpu/run_refusal_prefill_flip.sh
# Any extra args after the script name are appended to the python call (e.g. --prompt_template plain).

SLURM_LOG_DIR="logs/refusal_prefill_flip"
source run_on_gpu/common.sh

run uv run --no-sync --extra viz python experiments/interesting_queries/scripts/refusal_prefill_flip.py \
  --prompt_ids harm_125 harm_139 harm_116 \
  --continue_with base instruct \
  --prompt_template chat \
  --max_prefill 10 \
  "$@"
