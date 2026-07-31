#!/usr/bin/env bash

# Setup environment variables. sbatch passes all current env variables to the job.
# Hugging Face auth: run `huggingface-cli login` once (token is cached) or pre-set HF_TOKEN. See check-env.py.

# Submit batch
./sh/sbatch \
  --gres=gpu:4 \
  --constraint=48G \
  --mem=128G \
  --partition=jag-standard \
  --job-name=sweep \
  --time=21-00:00:00 \
  ./run_on_gpu/run_sweep_gpu.sh "$@"
