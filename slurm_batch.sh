#!/usr/bin/env bash

# Setup environment variables. sbatch passes all current env variables to the job.
# This runs `export HF_TOKEN=...`
export HF_TOKEN=$(cat ~/.shell/secrets/hf_token_write)

# Submit batch
sbatch \
  --account=nlp \
  --gres=gpu:1 \
  --constraint=48G \
  --mem=128G \
  --partition=jag-standard \
  --job-name=gemma2_2b \
  --time=21-00:00:00 \
  --mail-user="$USER@cs.stanford.edu" \
  --mail-type=END,FAIL \
  ./run_train_gpu.sh "$@"
