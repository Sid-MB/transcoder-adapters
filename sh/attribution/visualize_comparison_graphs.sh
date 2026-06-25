#!/usr/bin/env bash

# To produce this graph dir from an existing overlay: 
# First, run the preparation script 
# Ex: uv run python -m analysis.attribution.prepare_comparison_overlay_view --overlay_dir "$LARGE_ARTIFACTS_DIR/transcoder-adapters/base_adapter_comparisons/calibrated_2h_actual_base_adapter_jag_nodes2048_x28_20260528/overlay" --max_base_feature_nodes 64 --max_base_error_nodes 0
# Visualizes compact overlays comparing GemmaScope base-model circuit nodes with adapter circuit nodes for the same interesting_small prompts.
uv run --extra viz python -m analysis.attribution.serve_comparison_graphs \
  --graph_file_dir "$LARGE_ARTIFACTS_DIR/transcoder-adapters/base_adapter_comparisons/calibrated_2h_actual_base_adapter_jag_nodes2048_x28_20260528/overlay_compact_base64_error0" \
  --features_dir "$LARGE_ARTIFACTS_DIR/transcoder-adapters/feature_data/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860_20260521_011754_15515871/circuit_tracer_features" \
  --port 8046
