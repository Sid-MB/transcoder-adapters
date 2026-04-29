#!/usr/bin/env bash

# Setup environment variables. sbatch passes all current env variables to the job.
export HF_TOKEN=$(cat ~/.shell/secrets/hf_token_write)

# Submit batch
./sh/sbatch \
  --gres=gpu:4 \
  --constraint=48G \
  --mem=128G \
  --partition=jag-standard \
  --job-name=sweep \
  --time=21-00:00:00 \
  ./run_on_gpu/run_sweep_gpu.sh "$@"
