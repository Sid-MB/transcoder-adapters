#!/usr/bin/env bash

# To produce this graph dir from an existing overlay: 
# First, run the preparation script 
# Ex: uv run python -m analysis.attribution.prepare_comparison_overlay_view --overlay_dir /nlp/scr/siddharth/sparse-adaptation/base_adapter_comparisons/interesting_small_base_adapter_sphinx_nodes2048_logits10_20260528/overlay --output_dir /nlp/scr/siddharth/sparse-adaptation/base_adapter_comparisons/interesting_small_base_adapter_sphinx_nodes2048_logits10_20260528/overlay_compact_base64_error32 --max_base_feature_nodes 64 --max_base_error_nodes 32
# Visualizes compact overlays comparing GemmaScope base-model circuit nodes with adapter circuit nodes for the same interesting_small prompts.
uv run --extra viz python -m analysis.attribution.serve_comparison_graphs \
  --graph_file_dir /nlp/scr/siddharth/sparse-adaptation/base_adapter_comparisons/interesting_small_base_adapter_sphinx_nodes2048_logits10_20260528/overlay_compact_base64_error32 \
  --features_dir /nlp/scr/siddharth/sparse-adaptation/feature_data/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860_20260521_011754_15515871/circuit_tracer_features \
  --port 8044
