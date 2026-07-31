#!/usr/bin/env bash

# You can override certain config options via command line args.
# Example:
#  ./sh/slurm_batch_train.sh --config training/configs/gemma2_2b.yaml
#
#   ./sh/slurm_batch_train.sh \
#     --config training/configs/gemma2_2b.yaml training/configs/gemma2-matrix/chat_filter.yaml \
#     --learning_rate 1e-3 \
#     --batch_size 8 \
#     --run_name_prefix gemma2_chat_filter
#
# ./sh/slurm_batch_train.sh --config training/configs/gemma4_2b.yaml training/configs/gemma4_2b_full_500k.yaml


# sbatch passes all current env variables to the job, so the cached Hugging Face login
# (from `huggingface-cli login`) or a pre-set HF_TOKEN carries through. See check-env.py.

# Submit batch
./sh/sbatch \
  --gres=gpu:1 \
  --constraint=48G \
  --mem=128G \
  --partition=jag-standard \
  --job-name=gemma2_2b_huge \
  --time=21-00:00:00 \
  ./run_on_gpu/run_train_gpu.sh "$@"
