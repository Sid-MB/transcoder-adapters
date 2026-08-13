#!/usr/bin/env bash
#
# [refusal-token-tracing] (session fd1f0d19), 2026-08-12
#
# COMPLIANCE-graph contrast for the first-ten-token study: are the recurring "scaffolding" adapter
# features (L0/F5489, L5/F6977, ...) that drive the refusal opening refusal-SPECIFIC, or generic
# response-start features active whenever the adapter answers? We trace the ADAPTER's OWN completion on
# prompts it COMPLIES with (benign controls, guaranteed; + huge-adapter jailbreak candidates, kept only
# if tc8192 also complies -- --skip_refusals drops any that open "I cannot"), at the first 10 positions,
# then diff the top features against the refusal set (graph_analysis).
#
# Run on sphinx/jagupard:
#   TOK=... ./sh/sbatch --gres=gpu:1 --mem=128G --cpus-per-task=8 --partition=sphinx,jag-standard,jag-lo \
#     --job-name='[refusal-token-tracing] compliance_contrast' --export=ALL,HF_TOKEN=$TOK \
#     ./run_on_gpu/run_refusal_compliance_contrast.sh

SLURM_LOG_DIR="logs/attribution"
source run_on_gpu/common.sh

OUT="${CONTRAST_OUT:-/nlp/scr/siddharth/transcoder-adapters/base_adapter_comparisons/tc8192_compliance_first10}"
PROMPTS="$OUT/prompts"
REQUESTS="${CONTRAST_REQUESTS:-my_notes/08-10-26/refusal_token_tracing/contrast_requests.json}"

ADAPTER=siddharthmb/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860
ADAPTER_SCAN=siddharthmb/2026.TA.features_2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr_he94e9602bafa
BASE_SCAN=siddharthmb/2026.TA.features_gemma-2-2b_gemmascope_width_16k_average_l0_76_ms100000_ml1024_tk2_h12ad59325ffd

# Step 1: harvest the ADAPTER's own greedy completion per candidate, keep only the COMPLIANT ones,
# and write <id>__pos{k}.txt for their first 10 tokens.
run uv run --no-sync --extra viz python experiments/interesting_queries/scripts/make_first_n_token_prompts.py \
  --requests_json "$REQUESTS" \
  --source_model "$ADAPTER" --source_is_adapter \
  --n_tokens 10 --skip_refusals \
  --output_dir "$PROMPTS"

# Step 2: matched base-vs-adapter graphs at each kept position. Same config as the refusal run.
run uv run --no-sync --extra viz python -m analysis.attribution.run_base_adapter_comparison \
  --adapter_checkpoint "$ADAPTER" \
  --base_model google/gemma-2-2b \
  --prompts "$PROMPTS" \
  --prompt_format chat \
  --run_name tc8192_compliance_first10 \
  --output_dir "$OUT" \
  --gemmascope_width width_16k \
  --gemmascope_l0 average_l0_76 \
  --gemmascope_l0_match nearest \
  --feature_data_path "$ADAPTER_SCAN" \
  --base_feature_data_path "$BASE_SCAN" \
  --continuation_max_new_tokens 0 \
  "$@"
