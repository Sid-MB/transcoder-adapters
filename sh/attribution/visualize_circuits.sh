#!/usr/bin/env bash

# Ex:
#
# 1. make graphs with ./slurm_batch_attribution.sh
#
# 2. Then:
# 
# uv run circuit-tracer start-server --graph_file_dir="$LARGE_ARTIFACTS_DIR/transcoder-adapters/attribution_graphs/dashboard_tiny_prompt_fixed_siddharthmb/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860_20260520_231545_15506755"
#
# uv run --extra viz circuit-tracer start-server --graph_file_dir $LARGE_ARTIFACTS_DIR/transcoder-adapters/attribution_graphs/interesting_small_sl14793860_features15515871_2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860 --features_dir $LARGE_ARTIFACTS_DIR/transcoder-adapters/feature_data/2026.TA.gemma2_2b_sparse_probe_l1_0p016_tc8192_decb_l1w0.016_tarbb_lb2.0_ln1_dr10000_sl15512630_20260521_014635_15516077/circuit_tracer_features
#
# or, for the l1=0.016 run: uv run --extra viz circuit-tracer start-server --graph_file_dir="$LARGE_ARTIFACTS_DIR/transcoder-adapters/attribution_graphs/interesting_small_sl15516187_features15529706_2026.TA.gemma2_2b_full_100k_l1_0p016_gemma2_2b_full_100k_tc8192_decb_l1w0.016_tarbb_sl15516187" --features_dir="$LARGE_ARTIFACTS_DIR/transcoder-adapters/feature_data/2026.TA.gemma2_2b_full_100k_l1_0p016_gemma2_2b_full_100k_tc8192_decb_l1w0.016_tarbb_sl15516187_20260522_115513_15529706/circuit_tracer_features" --port=8042
#
# uv run --extra viz circuit-tracer start-server --graph_file_dir="$LARGE_ARTIFACTS_DIR/transcoder-adapters/attribution_graphs/interesting_small_nodes1024_logits10_sl15516187_features15529706_fixed_2026.TA.gemma2_2b_full_100k_l1_0p016_gemma2_2b_full_100k_tc8192_decb_l1w0.016_tarbb_sl15516187" --features_dir="$LARGE_ARTIFACTS_DIR/transcoder-adapters/feature_data/2026.TA.gemma2_2b_full_100k_l1_0p016_gemma2_2b_full_100k_tc8192_decb_l1w0.016_tarbb_sl15516187_20260522_115513_15529706/circuit_tracer_features"

uv run --extra viz circuit-tracer start-server "$@"
