#!/usr/bin/env bash
set -euo pipefail

BASE_CONFIG="training/configs/gemma2_2b.yaml"

for override in training/configs/gemma-matrix/*.yaml; do
  echo "Submitting: $override"
  ./sh/slurm_batch_train.sh --config "$BASE_CONFIG" "$override"
done
