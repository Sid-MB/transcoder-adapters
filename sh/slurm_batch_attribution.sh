#!/usr/bin/env bash
# Submit a RelP attribution job.
#
# The files in analysis/attribution/prompts_l0_1p4 are attribution prompts:
# each file includes the target continuation at the end, and run_attribution
# drops that final token before attributing the prediction.
#
# Gemma2 smoke-test example:
# ./sh/slurm_batch_attribution.sh --checkpoint siddharthmb/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860 --run_name gemma2_relp_smoke --prompts analysis/attribution/prompts_l0_1p4/wait.txt --output_dir products/attribution/gemma2_relp_smoke --max_feature_nodes 64 --batch_size 4 --max_n_logits 5
#
# Fuller Gemma2 run over all curated prompts:
# ./sh/slurm_batch_attribution.sh --job-name gemma2_relp_full --time 1-00:00:00 --checkpoint siddharthmb/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860 --run_name gemma2_relp_l0_1p4 --prompts analysis/attribution/prompts_l0_1p4 --output_dir products/attribution/gemma2_relp_l0_1p4 --max_feature_nodes 10000 --batch_size 16 --max_n_logits 10 --node_threshold 0.8 --edge_threshold 0.98

export HF_TOKEN=$(cat ~/.shell/secrets/hf_token_write)

JOB_NAME=attribution
TIME_LIMIT=04:00:00
PARTITION=jag-standard
GPU_CONSTRAINT=48G
MEMORY=128G
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
    --time)
      TIME_LIMIT="$2"
      shift 2
      ;;
    --time=*)
      TIME_LIMIT="${1#*=}"
      shift
      ;;
    --partition)
      PARTITION="$2"
      shift 2
      ;;
    --partition=*)
      PARTITION="${1#*=}"
      shift
      ;;
    --constraint)
      GPU_CONSTRAINT="$2"
      shift 2
      ;;
    --constraint=*)
      GPU_CONSTRAINT="${1#*=}"
      shift
      ;;
    --mem)
      MEMORY="$2"
      shift 2
      ;;
    --mem=*)
      MEMORY="${1#*=}"
      shift
      ;;
    *)
      OTHER_ARGS+=("$1")
      shift
      ;;
  esac
done

sbatch \
  --account=nlp \
  --gres=gpu:1 \
  --constraint="$GPU_CONSTRAINT" \
  --mem="$MEMORY" \
  --partition="$PARTITION" \
  --job-name="$JOB_NAME" \
  --time="$TIME_LIMIT" \
  --mail-user="$USER@cs.stanford.edu" \
  --mail-type=END,FAIL \
  ./slurm/run_attribution.sh "${OTHER_ARGS[@]}"
