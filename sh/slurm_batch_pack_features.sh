#!/usr/bin/env bash

# Ex:
# ./sh/slurm_batch_pack_features.sh --feature_dir /nlp/scr/siddharth/sparse-adaptation/feature_data/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl147_20260416_152706_15175469/features --n_layers 26 --n_features 8192

export HF_TOKEN=$(cat ~/.shell/secrets/hf_token_write)

./sh/sbatch \
  --gres=gpu:0 \
  --mem=64G \
  --partition=jag-standard \
  --job-name=pack_features \
  --time=0-04:00:00 \
  ./run_on_gpu/run_pack_features.sh "$@"
