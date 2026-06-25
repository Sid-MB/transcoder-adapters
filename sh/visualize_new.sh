#!/usr/bin/env bash
#
# Visualize the base-vs-adapter FULL-REPLACEMENT comparison graph.
#
# What you get in the graph viz:
#   - REAL reconstruction-error nodes, rendered as triangles (triangle glyph), ranked by influence.
#     Full replacement MLP(x) = T_base(x) + T_adapter(x) + Err means circuit-tracer's
#     standard linear attribution produces real error nodes for free (no RelP).
#   - PER-FEATURE ACTIVATION PROPORTIONS in the feature-detail panel:
#       * "Fires on XX% of tokens"            (activation_frequency)
#       * top tokens with "NN%"               (token_specificity = % of this feature's
#                                              activations that land on each token)
#   - BASE GemmaScope activation examples, collected on OUR chat + web data
#     (so you can see how base features behave on chat, e.g. on <end_of_turn>).
#
# Base feature index < 16384 = GemmaScope (hexagon, /base_features); >= 16384 = adapter (circle, /adapter_features).
#
# NOTE: GPU Slurm jobs run on jagupard (--partition=jag-standard --constraint=48G), not sphinx.

set -euo pipefail

ADAPTER=siddharthmb/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860
BASE=google/gemma-2-2b
PROMPTS=analysis/attribution/prompts/interesting_small

# ===========================================================================
# READY NOW: a graph with ALL elements is already generated and saved to durable
# scratch at /nlp/scr/siddharth/sparse-adaptation/comparison_demo -- error
# triangles + per-feature proportions + activation examples for BOTH base
# GemmaScope (/base_features) AND adapter (/adapter_features) feature nodes, all
# collected on our chat+web data. Just serve it and open http://localhost:8044
# (forward the port if remote):
#
#   cd /juice2/u/siddharth/transcoder-adapters && \
#   uv run --extra viz python -m analysis.attribution.serve_comparison_graphs \
#     --graph_file_dir /nlp/scr/siddharth/sparse-adaptation/comparison_demo/graph \
#     --base_features_dir /nlp/scr/siddharth/sparse-adaptation/comparison_demo/base_features \
#     --adapter_features_dir /nlp/scr/siddharth/sparse-adaptation/comparison_demo/adapter_features \
#     --port 8044
#
# (Regenerate equivalents from scratch with demo_local below.)
# ===========================================================================
DEMO=/nlp/scr/siddharth/sparse-adaptation/comparison_demo
serve_demo() {
  uv run --extra viz python -m analysis.attribution.serve_comparison_graphs \
    --graph_file_dir "$DEMO/graph" \
    --base_features_dir "$DEMO/base_features" \
    --adapter_features_dir "$DEMO/adapter_features" \
    --port 8044
}

# ===========================================================================
# QUICK SMOKE (verified working): one prompt, base examples from HF (mntss),
# no separate collection needed. Run on a GPU box (outside the sandbox).
# ===========================================================================
quick_smoke() {
  uv run --extra viz python -m analysis.attribution.run_combined_attribution \
    --adapter_checkpoint "$ADAPTER" --base_model "$BASE" \
    --prompts "$PROMPTS/capital_colesseum.txt" \
    --prompt_format chat \
    --gemmascope_width width_16k --gemmascope_l0 average_l0_76 \
    --base_feature_data_path mntss/gemma-scope-transcoders \
    --max_feature_nodes 256 --batch_size 4 --max_n_logits 5 --max_error_nodes 32 \
    --run_name combined_smoke --output_dir ./combined_graphs_smoke

  uv run --extra viz python -m analysis.attribution.serve_comparison_graphs \
    --graph_file_dir ./combined_graphs_smoke --port 8044
  # Open http://localhost:8044  (base examples load from HF; proportions need step 1 below)
}

# ===========================================================================
# FULL PIPELINE (proportions + base examples collected on OUR chat+web data)
# ===========================================================================

# --- 1) Collect BASE (GemmaScope) features on our chat+web data (jagupard) -----
#     Produces packed circuit_tracer_features/ with per-feature activation examples,
#     activation_frequency (% tokens active), and token_specificity (% activations/token).
#     The output dir is printed in logs/collect_base_features/<...>.out.
collect_base_features() {
  ./sh/slurm_batch_collect_base_features.sh \
    --gemmascope_width width_16k --gemmascope_l0 average_l0_76 \
    --val_data chat:siddharthmb/2026.transcoder-adapters.lmsys-chat-1m-splits \
               fineweb:science-of-finetuning/fineweb-1m-sample \
    --max_samples 50 --max_length 1024 \
    --no_per_feature_json --no-upload_circuit_tracer_features_to_hub \
    --top_k 5 --domain_top_k 3 --n_random 0 --activation_range_examples_per_domain 0 \
    --context_before 30 --context_after 8
  # -> note the printed output dir; BASE_FEATURES=<that dir>/circuit_tracer_features
}

# --- 2) Combined full-replacement attribution over all prompts ----------------
#     Point --base_feature_data_path at the collection from step 1 (local dir served as
#     /base_features) so base nodes show OUR chat examples + proportions. Add
#     --adapter_feature_data_path <adapter feature repo/dir> for adapter-side examples.
run_combined() {
  local BASE_FEATURES="$1"      # base .../circuit_tracer_features (or an HF feature repo id)
  local ADAPTER_FEATURES="${2:-}"  # optional adapter .../circuit_tracer_features (or HF repo id)
  local ADAPTER_ARG=()
  [ -n "$ADAPTER_FEATURES" ] && ADAPTER_ARG=(--adapter_feature_data_path "$ADAPTER_FEATURES")
  uv run --extra viz python -m analysis.attribution.run_combined_attribution \
    --adapter_checkpoint "$ADAPTER" --base_model "$BASE" \
    --prompts "$PROMPTS" --prompt_format chat \
    --gemmascope_width width_16k --gemmascope_l0 average_l0_76 \
    --base_feature_data_path "$BASE_FEATURES" "${ADAPTER_ARG[@]}" \
    --max_feature_nodes 4096 --batch_size 4 --max_n_logits 5 --max_error_nodes 32 \
    --run_name combined --output_dir ./combined_graphs
}

# Collect ADAPTER feature activations (sparse -> fast) so adapter (●) nodes also
# show examples + proportions. Output dir is printed in logs / stdout.
collect_adapter_features() {
  uv run --extra viz python -m analysis.features.collect_feature_activations \
    --model_path "$ADAPTER" \
    --val_data chat:siddharthmb/2026.transcoder-adapters.lmsys-chat-1m-splits \
               fineweb:science-of-finetuning/fineweb-1m-sample \
    --max_samples 12 --max_length 512 --no-upload_circuit_tracer_features_to_hub \
    --top_k 3 --domain_top_k 2 --n_random 0 --activation_range_examples_per_domain 0 \
    --context_before 20 --context_after 5 --output_dir ./adapter_features_demo
}

# --- 3) Serve the comparison graph (patched circuit-tracer frontend) ----------
#     --base_features_dir / --adapter_features_dir serve local packed features as
#     /base_features and /adapter_features. Open http://localhost:8044, click nodes for
#     examples + proportions; reconstruction-error nodes are triangles.
serve_combined() {
  local BASE_FEATURES="$1"         # local base .../circuit_tracer_features dir (omit if base is an HF repo)
  local ADAPTER_FEATURES="${2:-}"  # optional adapter .../circuit_tracer_features dir
  local ADAPTER_ARG=()
  [ -n "$ADAPTER_FEATURES" ] && ADAPTER_ARG=(--adapter_features_dir "$ADAPTER_FEATURES")
  uv run --extra viz python -m analysis.attribution.serve_comparison_graphs \
    --graph_file_dir ./combined_graphs \
    --base_features_dir "$BASE_FEATURES" "${ADAPTER_ARG[@]}" \
    --port 8044
}

# ===========================================================================
# VERIFIED LOCAL DEMO (single GPU box, no Slurm). The exact flow validated
# end-to-end: error triangles + activation proportions + chat/web examples on
# BOTH base and adapter feature nodes, all in one served graph. ~12 min total --
# the per-token-id decode cache in export makes packing the dense base features
# fast (~3s/layer).
# ===========================================================================
demo_local() {
  local BASE_OUT=./base_features_demo
  local ADAPTER_OUT=./adapter_features_demo
  # 1a) base GemmaScope features on our chat+web data (with proportions)
  uv run --extra viz python -m analysis.features.collect_base_feature_activations \
    --base_model "$BASE" --gemmascope_width width_16k --gemmascope_l0 average_l0_76 \
    --val_data chat:siddharthmb/2026.transcoder-adapters.lmsys-chat-1m-splits \
               fineweb:science-of-finetuning/fineweb-1m-sample \
    --max_samples 12 --max_length 512 --no_per_feature_json --no-upload_circuit_tracer_features_to_hub \
    --top_k 3 --domain_top_k 2 --n_random 0 --activation_range_examples_per_domain 0 \
    --context_before 20 --context_after 5 --output_dir "$BASE_OUT"
  # 1b) adapter features (sparse) so adapter nodes also show examples + proportions
  collect_adapter_features   # -> ./adapter_features_demo
  # 2) combined full-replacement attribution with BOTH feature sets
  run_combined "$BASE_OUT" "$ADAPTER_OUT/circuit_tracer_features"
  # 3) serve both; open http://localhost:8044
  serve_combined "$BASE_OUT/circuit_tracer_features" "$ADAPTER_OUT/circuit_tracer_features"
}

# Usage:
#   ./sh/visualize_new.sh serve_demo            # serve THIS session's already-generated graph (instant)
#   ./sh/visualize_new.sh demo_local            # verified end-to-end (collect + attribute + serve)
#   ./sh/visualize_new.sh quick_smoke           # one prompt, base examples from HF (no proportions)
#   ./sh/visualize_new.sh collect_base_features # base collection on jagupard
#   ./sh/visualize_new.sh run_combined   /path/to/circuit_tracer_features
#   ./sh/visualize_new.sh serve_combined /path/to/circuit_tracer_features
"${@:-demo_local}"
