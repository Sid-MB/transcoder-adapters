#!/usr/bin/env bash



# - This is the newest hybrid graph (run_combined_attribution job 16179855) — the combined base+adapter full-replacement graph (MLP = T_base + T_adapter + Err in one graph), with the fine-tuned transcoders on the base side and ~20-example/feature dashboards from the ms100000_dtk20 HF repos.
# - It uses the deployed/uploaded fine-tuned weights — the 2M-token, per-layer-sparsity-penalty set (the HF repo). That's the "best" one to serve: the 10M-token run gave only a marginal FVU gain at L25 (0.221 → 0.211, plateaued) and over-sparsified L24/L25's L0 below base, so it's not actually a better artifact — I didn't build a hybrid graph from it for that reason.
# Caveats: 2M tokens, not 10M. See my_notes/07-09-26/figures/transcoder_finetune_token_curve.png for analysis: using 10M tokens helps reconstruction error slightly but needs different params to not become too sparse (did not run).
uv run --extra viz python -m analysis.attribution.serve_comparison_graphs --graph_file_dir /nlp/scr/siddharth/transcoder-adapters/base_adapter_comparisons/hybrid_finetuned_ftL0-24-25 --port 8044
