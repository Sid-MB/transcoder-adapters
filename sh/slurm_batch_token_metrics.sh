#!/usr/bin/env bash
# Example (gemma-2-2b):
# ./sh/slurm_batch_token_metrics.sh --model siddharthmb/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860 --transcoder --reference_model google/gemma-2-2b-it --data_source lmsys_chat
#
# Hybrid Example (adapters disabled):
# ./sh/slurm_batch_token_metrics.sh --model siddharthmb/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860 --hybrid --reference_model google/gemma-2-2b-it --data_source lmsys_chat

export HF_TOKEN=$(cat ~/.shell/secrets/hf_token_write)

JOB_NAME=token_metrics
OTHER_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --job-name)
      JOB_NAME="$2"
      shift 2
      ;;
    --job-name=*)
      JOB_NAME="${1#*=}"
      shift
      ;;
    *)
      OTHER_ARGS+=("$1")
      shift
      ;;
  esac
done

./sh/sbatch \
  --gres=gpu:1 \
  --constraint=48G \
  --mem=128G \
  --partition=jag-hi \
  --job-name="$JOB_NAME" \
  --time=1-00:00:00 \
  ./run_on_gpu/run_token_metrics.sh "${OTHER_ARGS[@]}"
