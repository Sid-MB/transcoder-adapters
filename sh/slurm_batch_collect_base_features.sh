#!/usr/bin/env bash

# Collect BASE-model (GemmaScope) transcoder feature activations on chat + web data.
# Runs on jagupard (jag-standard, 48G GPUs). The GemmaScope collection fits comfortably
# in 48G. Do not use the sphinx partition unless specifically directed (sphinx has
# preempted/killed long jobs here).
#
# Reduced ~30-min run (packed-only for the attribution overlay, capped samples):
#  ./sh/slurm_batch_collect_base_features.sh \
#    --gemmascope_width width_16k --gemmascope_l0 average_l0_76 \
#    --val_data chat:siddharthmb/2026.transcoder-adapters.lmsys-chat-1m-splits fineweb:science-of-finetuning/fineweb-1m-sample \
#    --max_samples 75 --max_length 1024 --no_per_feature_json --no-upload_circuit_tracer_features_to_hub \
#    --top_k 8 --domain_top_k 5 --n_random 3 --activation_range_examples_per_domain 0
#
# Notes:
#  - GemmaScope base features fire very densely on this data (~68% of all ~425k
#    features fire), so --no_per_feature_json keeps only the packed cache (what the
#    overlay uses) to finish quickly. Drop it (and raise --max_samples) for a full,
#    dashboard-browsable collection.
#  - The output directory is printed in the logs; pass it to
#    run_base_adapter_comparison --base_feature_data_path to serve it as /base_features.

export HF_TOKEN=$(cat ~/.shell/secrets/hf_token_write)

./sh/sbatch \
  --gres=gpu:1 \
  --constraint=48G \
  --mem=128G \
  --partition=jag-standard \
  --job-name=collect_base_features \
  ./run_on_gpu/run_collect_base_features.sh "$@"
