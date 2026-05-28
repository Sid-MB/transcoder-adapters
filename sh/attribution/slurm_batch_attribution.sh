#!/usr/bin/env bash
# Submit a RelP attribution job.
# Produces circuit graph files that can be visualized with ./visualize_circuits.sh
#
# The files in analysis/attribution/prompts/interesting_small are attribution prompts:
# each file includes the target continuation at the end, and run_attribution
# drops that final token before attributing the prediction.
#
# Gemma2 smoke-test example:
# ./sh/slurm_batch_attribution.sh --checkpoint siddharthmb/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860 --run_name gemma2_relp_smoke --prompts analysis/attribution/prompts/interesting_small/capital_paris.txt --max_feature_nodes 64 --batch_size 4 --max_n_logits 5
#
# Fuller Gemma2 run over all curated prompts:
# ./sh/slurm_batch_attribution.sh --checkpoint siddharthmb/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860 --run_name gemma2_relp_interesting_small --prompts analysis/attribution/prompts/interesting_small --max_feature_nodes 10000 --batch_size 16 --max_n_logits 10 --node_threshold 0.8 --edge_threshold 0.98
#
# Multi-GPU auto-sharded run:
# ATTRIBUTION_GPUS=4 ./sh/slurm_batch_attribution.sh --checkpoint siddharthmb/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860 --run_name gemma2_relp_interesting_small --prompts analysis/attribution/prompts/interesting_small --max_feature_nodes 10000 --batch_size 16 --max_n_logits 10 --node_threshold 0.8 --edge_threshold 0.98 --auto_shard_gpus

ATTRIBUTION_GPUS="${ATTRIBUTION_GPUS:-1}"

if ! [[ "$ATTRIBUTION_GPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: ATTRIBUTION_GPUS must be a positive integer, got '$ATTRIBUTION_GPUS'" >&2
  exit 1
fi

export HF_TOKEN=$(cat ~/.shell/secrets/hf_token_write)

./sh/sbatch \
  --gres="gpu:${ATTRIBUTION_GPUS}" \
  --constraint=48G \
  --mem=128G \
  --partition=jag-standard \
  --job-name=attribution \
  ./run_on_gpu/run_attribution.sh "$@"

# ./sh/sbatch \
#   --gres="gpu:${ATTRIBUTION_GPUS}" \
#   --constraint=80G \
#   --mem=128G \
#   --partition=sphinx \
#   --job-name=attribution \
#   ./run_on_gpu/run_attribution.sh "$@"
