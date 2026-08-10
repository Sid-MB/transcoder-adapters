#!/usr/bin/env bash
#
# [refusal-token-tracing] (session fd1f0d19-f5d1-48cc-9002-3f06f2abe2fd), 2026-08-10
#
# Continuation-only re-judge of the refusal_prefill_flip runs (removes the prefill confound). Re-scores the
# saved generations only -- no regeneration. Judges every results.json passed as an argument.
#
#   ./sh/sbatch --gres=gpu:1 --mem=128G --cpus-per-task=8 --partition=sphinx,jag-standard,jag-lo \
#     --job-name='[refusal-token-tracing] rejudge' ./run_on_gpu/run_rejudge_prefill.sh \
#     experiments/interesting_queries/results/refusal_prefill_flip_chat/results.json \
#     experiments/interesting_queries/results/refusal_prefill_flip_plain/results.json

SLURM_LOG_DIR="logs/refusal_prefill_flip"
source run_on_gpu/common.sh

for results in "$@"; do
  run uv run --no-sync --extra viz python experiments/interesting_queries/scripts/rejudge_prefill_continuation.py --results "$results"
done
