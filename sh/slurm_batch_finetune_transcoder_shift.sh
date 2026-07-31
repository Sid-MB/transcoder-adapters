#!/usr/bin/env bash

# Re-fine-tune GemmaScope transcoders to the instruct input distribution (jagupard, 1x48G).
# See analysis/features/finetune_transcoder_shift.py.
#
# Smoke (layer 25, tiny):
#  ./sh/slurm_batch_finetune_transcoder_shift.sh --gemmascope_width width_16k --gemmascope_l0 average_l0_76 \
#    --layers 25 --train_tokens 20000 --eval_tokens 8000 --no-wandb
# Full (endpoint layers, 2M train tokens):
#  ./sh/slurm_batch_finetune_transcoder_shift.sh --gemmascope_width width_16k --gemmascope_l0 average_l0_76 \
#    --layers 0 24 25 --train_tokens 2000000 --eval_tokens 200000 --lr 1e-4 --wandb

./sh/sbatch \
  --gres=gpu:1 \
  --constraint=48G \
  --mem=64G \
  --partition=jag-standard \
  --time=0-04:00:00 \
  --job-name=ft_transcoder_shift \
  ./run_on_gpu/run_finetune_transcoder_shift.sh "$@"
