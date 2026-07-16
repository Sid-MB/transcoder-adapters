#!/usr/bin/env bash


## New version!! (older below)
# - fine-tuned GemmaScope base transcoders (siddharthmb/2026.TA.gemma2_2b_gemmascope_transcoders_instruct_ft_L0-24-25, layers 0/24/25) on the base side,
# - the real adapter model (siddharthmb/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860) on the adapter side → instruct completions,
# - best 100K-corpus feature examples (ms100000_dtk20) on both sides.

# LARGE_ARTIFACTS_DIR=${LARGE_ARTIFACTS_DIR:-/nlp/scr/siddharth}

# Comprehensive set (40 prompts: agree__ / diverge__ / interesting_small__). This is the current best one to serve.
uv run --extra viz python -m analysis.attribution.serve_comparison_graphs --graph_file_dir /nlp/scr/siddharth/transcoder-adapters/base_adapter_comparisons/hybrid_ft_overlay_ms100k_comprehensive/overlay --port 8046
# From HF (compact overlays, no local copy needed):
# uv run --extra viz python -m analysis.attribution.serve_comparison_graphs --graph_file_dir siddharthmb/2026.TA.hybrid_ft_overlay_ms100k_graphs:comprehensive/overlay_compact --port 8046
# 12-token base/hybrid/instruct completions per prompt: hybrid_ft_overlay_ms100k_comprehensive/completions.{json,md}

# Original smaller set (interesting_small, 12 prompts):
# uv run --extra viz python -m analysis.attribution.serve_comparison_graphs --graph_file_dir /nlp/scr/siddharth/transcoder-adapters/base_adapter_comparisons/hybrid_ft_overlay_ms100k/overlay --port 8046
# uv run --extra viz python -m analysis.attribution.serve_comparison_graphs --graph_file_dir siddharthmb/2026.TA.hybrid_ft_overlay_ms100k_graphs:overlay_compact --port 8046


# This one suffered from a bug:

# - This is the hybrid graph (run_combined_attribution job 16179855) — the combined base+adapter full-replacement graph (MLP = T_base + T_adapter + Err in one graph), with the fine-tuned transcoders on the base side and ~20-example/feature dashboards from the ms100000_dtk20 HF repos.
# - It uses the deployed/uploaded fine-tuned weights — the 2M-token, per-layer-sparsity-penalty set (the HF repo). That's the "best" one to serve: the 10M-token run gave only a marginal FVU gain at L25 (0.221 → 0.211, plateaued) and over-sparsified L24/L25's L0 below base, so it's not actually a better artifact — I didn't build a hybrid graph from it for that reason.
# Caveats: 2M tokens, not 10M. See my_notes/07-09-26/figures/transcoder_finetune_token_curve.png for analysis: using 10M tokens helps reconstruction error slightly but needs different params to not become too sparse (did not run).
# The issue: see my_notes/07-09-26/07-15-26 hybrid graph instruct-completions bug.md
# uv run --extra viz python -m analysis.attribution.serve_comparison_graphs --graph_file_dir /nlp/scr/siddharth/transcoder-adapters/base_adapter_comparisons/hybrid_finetuned_ftL0-24-25 --port 8044



