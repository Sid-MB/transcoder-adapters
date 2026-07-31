#!/usr/bin/env bash

# Measure GemmaScope transcoder input-distribution shift: L0 (sparsity) and reconstruction
# error (FVU) when the transcoder input hidden states come from the base model vs. the
# instruction-tuned model. Target is always MLP_base(x). See
# analysis/features/transcoder_input_shift.py (Experiment 1).
#
# Runs on jagupard (jag-standard, 48G GPUs) -- fits comfortably in 48G (gemma-2-2b x2 +
# GemmaScope transcoders). Do NOT use sphinx unless specifically directed.
#
# Smoke (single layer, ~2k tokens, no wandb):
#  ./sh/slurm_batch_transcoder_input_shift.sh \
#    --gemmascope_width width_16k --gemmascope_l0 average_l0_76 \
#    --sources base instruct --layers 0 --max_tokens 2000 --no-wandb
#
# First real read (5 layers across depth, ~1M tokens, wandb on):
#  ./sh/slurm_batch_transcoder_input_shift.sh \
#    --gemmascope_width width_16k --gemmascope_l0 average_l0_76 \
#    --sources base instruct --layers 0 6 12 18 25 --max_tokens 1000000 --wandb
#
# The output directory (results.json, per_feature_fire_freq.npz, summary.md) is printed in
# the logs.
#
# Hugging Face auth: run `huggingface-cli login` once (token is cached) or pre-set HF_TOKEN.

./sh/sbatch \
  --gres=gpu:1 \
  --constraint=48G \
  --mem=64G \
  --partition=jag-standard \
  --job-name=transcoder_input_shift \
  ./run_on_gpu/run_transcoder_input_shift.sh "$@"
