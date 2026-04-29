#!/usr/bin/env bash

# Ex
#  ./sh/slurm_batch_collect_features.sh  --model_path siddharthmb/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860 --val_data chat:siddharthmb/2026.transcoder-adapters.lmsys-chat-1m-splits fineweb:science-of-finetuning/fineweb-1m-sample --batch_size 16
#
# Small chat-only run for testing assistant-response annotations:
# ./sh/slurm_batch_collect_features.sh --model_path siddharthmb/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860 --val_data chat:siddharthmb/2026.transcoder-adapters.lmsys-chat-1m-splits --max_samples 100 --batch_size 16
#
# Smaller mixed-domain run, roughly 30 min:
# ./sh/slurm_batch_collect_features.sh --model_path siddharthmb/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860 --val_data chat:siddharthmb/2026.transcoder-adapters.lmsys-chat-1m-splits fineweb:science-of-finetuning/fineweb-1m-sample --max_samples 1000 --batch_size 16
#
# Old example (don't use)
# ./sh/slurm_batch_collect_features.sh --model_path siddharthmb/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr10000_lr8e-04_bs4_sl14754432 --val_data siddharthmb/2026.transcoder-adapters.lmsys-chat-1m-splits
#

export HF_TOKEN=$(cat ~/.shell/secrets/hf_token_write)

./sh/sbatch \
  --gres=gpu:1 \
  --constraint=48G \
  --mem=128G \
  --partition=jag-standard \
  --job-name=collect_features \
  ./run_on_gpu/run_collect_features.sh "$@"
