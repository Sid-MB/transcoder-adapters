#!/usr/bin/env bash
#
# [refusal-token-tracing] (session fd1f0d19-f5d1-48cc-9002-3f06f2abe2fd), 2026-08-10
#
# Exp 2 of the "which token loads the refusal?" study: build base-vs-adapter attribution overlays at each
# of the FIRST 10 positions of the instruct refusal (I, cannot, provide, ...) for 3 prompts -> 30 graphs.
# Same tc8192 config as the *_strict_refusal overlay (which traces only position 0, "I"); this extends it
# across the opening so we can see where refusal / harmful circuitry appears.
#
# Step 1 generates growing-prefix prompt files (assistant = first k+1 refusal tokens) from the instruct
# model's greedy refusal; Step 2 traces the last token of each with run_base_adapter_comparison.
#
# Run on sphinx (user-approved) or jagupard:
#   ./sh/sbatch --gres=gpu:1 --mem=128G --cpus-per-task=8 --partition=sphinx \
#     --job-name='[refusal-token-tracing] first10_graphs' ./run_on_gpu/run_refusal_first10_graphs.sh
# Serve afterward (then open http://localhost:8052):
#   uv run --extra viz python -m analysis.attribution.serve_comparison_graphs --graph_file_dir "$OUT/overlay" --port 8052

SLURM_LOG_DIR="logs/attribution"
source run_on_gpu/common.sh

OUT="${REFUSAL_FIRST10_OUT:-/nlp/scr/siddharth/transcoder-adapters/base_adapter_comparisons/tc8192_refusal_first10}"
PROMPTS="$OUT/prompts"
N_TOKENS="${REFUSAL_FIRST10_N:-10}"

ADAPTER=siddharthmb/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860
ADAPTER_SCAN=siddharthmb/2026.TA.features_2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr_he94e9602bafa
BASE_SCAN=siddharthmb/2026.TA.features_gemma-2-2b_gemmascope_width_16k_average_l0_76_ms100000_ml1024_tk2_h12ad59325ffd

# Step 1: harvest the instruct refusal openings and write <id>__pos{k}.txt prompt files.
run uv run --no-sync --extra viz python experiments/interesting_queries/scripts/make_first_n_token_prompts.py \
  --prompt_ids harm_125 harm_139 harm_116 \
  --n_tokens "$N_TOKENS" \
  --output_dir "$PROMPTS"

# Step 2: matched base-vs-adapter graphs + overlay for every position. Config mirrors tc8192_strict_refusal.
run uv run --no-sync --extra viz python -m analysis.attribution.run_base_adapter_comparison \
  --adapter_checkpoint "$ADAPTER" \
  --base_model google/gemma-2-2b \
  --prompts "$PROMPTS" \
  --prompt_format chat \
  --run_name tc8192_refusal_first10 \
  --output_dir "$OUT" \
  --gemmascope_width width_16k \
  --gemmascope_l0 average_l0_76 \
  --gemmascope_l0_match nearest \
  --feature_data_path "$ADAPTER_SCAN" \
  --base_feature_data_path "$BASE_SCAN" \
  --continuation_max_new_tokens 0 \
  "$@"
