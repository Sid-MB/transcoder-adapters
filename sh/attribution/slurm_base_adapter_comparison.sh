#!/usr/bin/env bash
# Submit a GPU job for matched base-vs-adapter circuit graph comparison.
#
# Example:
#   SBATCH_WAIT=1 ATTRIBUTION_GPUS=1 ./sh/attribution/slurm_base_adapter_comparison.sh \
#     --adapter_checkpoint siddharthmb/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860 \
#     --base_model google/gemma-2-2b \
#     --feature_data_path /nlp/scr/siddharth/sparse-adaptation/feature_data/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860_20260521_011754_15515871 \
#     --prompts analysis/attribution/prompts/interesting_small/capital_paris.txt \
#     --run_name capital_paris_base_adapter_overlay \
#     --prompt_format chat \
#     --gemmascope_width width_16k \
#     --gemmascope_l0 average_l0_76 \
#     --max_feature_nodes 64 \
#     --batch_size 4 \
#     --max_n_logits 5

set -euo pipefail

ATTRIBUTION_GPUS="${ATTRIBUTION_GPUS:-1}"
GPU_PARTITION="${CT_GPU_PARTITION:-jag-standard}"
GPU_CPUS="${CT_GPU_CPUS:-8}"
GPU_MEM="${CT_GPU_MEM:-128G}"
GPU_TIME="${CT_GPU_TIME:-04:00:00}"

if ! [[ "$ATTRIBUTION_GPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: ATTRIBUTION_GPUS must be a positive integer, got '$ATTRIBUTION_GPUS'" >&2
  exit 1
fi

if [[ -f ~/.shell/secrets/hf_token_write ]]; then
  export HF_TOKEN
  HF_TOKEN="$(cat ~/.shell/secrets/hf_token_write)"
fi

./sh/sbatch \
  --gres="gpu:${ATTRIBUTION_GPUS}" \
  --constraint=48G \
  --cpus-per-task="$GPU_CPUS" \
  --mem="$GPU_MEM" \
  --time="$GPU_TIME" \
  --partition="$GPU_PARTITION" \
  --job-name=base_adapter_cmp \
  --output=logs/attribution/%x_%j.out \
  --error=logs/attribution/%x_%j.err \
  ./run_on_gpu/run_base_adapter_comparison.sh "$@"
