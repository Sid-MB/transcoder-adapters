#!/usr/bin/env bash

# Submit a short CUDA test that imports torch, checks CUDA availability,
# allocates a tensor on the GPU, and logs the result.

./sh/sbatch \
  --gres=gpu:1 \
  --constraint=48G \
  --mem=16G \
  --partition=jag-standard \
  --job-name=cuda_test \
  --time=0-00:10:00 \
  ./run_on_gpu/run_cuda_test.sh "$@"
