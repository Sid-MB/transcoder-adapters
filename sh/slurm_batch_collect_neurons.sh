#!/usr/bin/env bash

# Ex:
# ./sh/slurm_batch_collect_neurons.sh --model_path siddharthmb/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860 --val_data siddharthmb/2026.transcoder-adapters.lmsys-chat-1m-splits

export HF_TOKEN=$(cat ~/.shell/secrets/hf_token_write)

sbatch \
  --account=nlp \
  --gres=gpu:1 \
  --constraint=48G \
  --mem=128G \
  --partition=jag-hi \
  --job-name=collect_neurons \
  --time=2-00:00:00 \
  --mail-user="$USER@cs.stanford.edu" \
  --mail-type=END,FAIL \
  ./slurm/run_collect_neurons.sh "$@"
