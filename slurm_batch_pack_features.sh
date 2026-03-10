#!/usr/bin/env bash

export HF_TOKEN=$(cat ~/.shell/secrets/hf_token_write)

sbatch \
  --account=nlp \
  --gres=gpu:0 \
  --mem=64G \
  --partition=jag-standard \
  --job-name=pack_features \
  --time=0-04:00:00 \
  --mail-user="$USER@cs.stanford.edu" \
  --mail-type=END,FAIL \
  ./slurm/run_pack_features.sh "$@"
