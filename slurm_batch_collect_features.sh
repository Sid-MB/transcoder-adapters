#!/usr/bin/env bash

export HF_TOKEN=$(cat ~/.shell/secrets/hf_token_write)

sbatch \
  --account=nlp \
  --gres=gpu:1 \
  --constraint=48G \
  --mem=128G \
  --partition=jag-standard \
  --job-name=collect_features \
  --time=1-00:00:00 \
  --mail-user="$USER@cs.stanford.edu" \
  --mail-type=END,FAIL \
  ./run_collect_features.sh "$@"
