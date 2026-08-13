#!/usr/bin/env bash
#
# [refusal-token-tracing] (session fd1f0d19), 2026-08-13
#
# Causal test of the one refusal-specific adapter feature (L6/F4241) found by the compliance contrast:
# clamp it to 0 and measure whether the tc8192 adapter still refuses. Cheap (one adapter model, a few
# forwards + short greedy gens). Compares ablating L6/F4241 vs a generic control feature (L0/F5489).
#
#   TOK=... ./sh/sbatch --gres=gpu:1 --mem=128G --cpus-per-task=8 --partition=sphinx,jag-standard,jag-lo \
#     --job-name='[refusal-token-tracing] ablate_L6F4241' --export=ALL,HF_TOKEN=$TOK \
#     ./run_on_gpu/run_ablate_refusal_feature.sh

SLURM_LOG_DIR="logs/refusal_prefill_flip"
source run_on_gpu/common.sh

run uv run --no-sync --extra viz python experiments/interesting_queries/scripts/ablate_refusal_feature.py \
  --ablate_feature 6:4241 --control_feature 0:5489 \
  --refusal_ids harm_125 harm_139 harm_116 \
  --output_dir experiments/interesting_queries/results/ablate_refusal_feature \
  "$@"
