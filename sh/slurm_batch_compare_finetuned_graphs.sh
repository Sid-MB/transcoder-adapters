#!/usr/bin/env bash
# Build circuit-tracer attribution graphs with original vs fine-tuned GemmaScope transcoders
# and compare error-node fraction (jagupard, 1x48G). See
# analysis/attribution/compare_finetuned_transcoder_graphs.py.
./sh/sbatch \
  --gres=gpu:1 --constraint=48G --mem=64G --partition=jag-standard \
  --job-name=cmp_ft_graphs \
  ./run_on_gpu/run_compare_finetuned_graphs.sh "$@"
