#!/usr/bin/env bash
#
# [implement: new 07/02 gemmascope transcoder experiments] 2026-07-15
#
# Copy-paste commands to visualize the INSTRUCT-fine-tuned GemmaScope transcoders
# (gemma-2-2b, layers 0/24/25). See my_notes/07-09-26/README.md for full context, metrics, figures.
#
# The fine-tuned weights are on HuggingFace and load directly by repo id (no local files needed):
FT=siddharthmb/2026.TA.gemma2_2b_gemmascope_transcoders_instruct_ft_L0-24-25
#   https://huggingface.co/siddharthmb/2026.TA.gemma2_2b_gemmascope_transcoders_instruct_ft_L0-24-25
#   -> layers 0/24/25 replace the pretrained GemmaScope layers; the rest stay pretrained.
#
# NOTE: GPU Slurm jobs run on jagupard (--partition=jag-standard --constraint=48G), not sphinx.

BASE=google/gemma-2-2b
WIDTH=width_16k
L0=average_l0_76
LAYERS="0 24 25"
PROMPTS=analysis/attribution/prompts/interesting_small

# ===========================================================================
# READY NOW (no GPU): serve the pre-built ORIGINAL-vs-FINE-TUNED comparison graphs,
# side by side, with feature examples (click a node). Graphs are bundled in the repo.
#   :8050 original GemmaScope   ·   :8051 fine-tuned (layers 0/24/25)
# ===========================================================================
serve_bundled() {
  ./my_notes/07-09-26/serve_graphs.sh
  # From your laptop: ssh -L 8050:localhost:8050 -L 8051:localhost:8051 <node>
  #   http://localhost:8050 (original)   http://localhost:8051 (fine-tuned)
}

# ===========================================================================
# REBUILD the original-vs-fine-tuned comparison graphs from scratch, pulling the
# fine-tuned weights straight from HuggingFace (GPU job on jagupard). Output dir is
# printed in the logs; serve it with GRAPH_DIR=<out> ./my_notes/07-09-26/serve_graphs.sh
# ===========================================================================
rebuild_comparison() {
  ./sh/slurm_batch_compare_finetuned_graphs.sh \
    --finetune_dir "$FT" --finetuned_layers $LAYERS \
    --gemmascope_width "$WIDTH" --gemmascope_l0 "$L0" --ft_layers $LAYERS \
    --prompts "$PROMPTS" --max_prompts 3 --max_feature_nodes 256 --max_n_logits 5
}

# ===========================================================================
# PRODUCTION visualizer with the fine-tuned transcoders applied: standard base
# attribution + base-vs-adapter overlay + serve, but with layers 0/24/25 replaced
# by the fine-tuned weights (from HF). Add your usual --adapter_checkpoint etc.
# ===========================================================================
production_serve() {
  uv run --extra viz python -m analysis.attribution.run_base_adapter_comparison \
    --base_model "$BASE" --gemmascope_width "$WIDTH" --gemmascope_l0 "$L0" \
    --finetuned_transcoder_dir "$FT" --finetuned_layers $LAYERS \
    --prompts "$PROMPTS" --prompt_format chat \
    --max_feature_nodes 4096 --batch_size 4 --max_n_logits 5 --serve --port 8044 \
    "$@"
}

# ===========================================================================
# EVALUATE the fine-tuned transcoders (FVU / L0 on base vs instruct hidden states),
# applying the HF weights. --layers all for the full 26-layer profile.
# ===========================================================================
evaluate() {
  uv run --extra viz python -m analysis.features.transcoder_input_shift \
    --gemmascope_width "$WIDTH" --gemmascope_l0 "$L0" --sources base instruct --layers $LAYERS \
    --finetuned_transcoder_dir "$FT" --finetuned_layers $LAYERS --max_tokens 200000 --no-wandb
}

# ===========================================================================
# HYBRID (combined base+adapter full-replacement) graphs, with the fine-tuned
# transcoders on the BASE side: MLP(x) = T_base_finetuned(x) + T_adapter(x) + Err,
# all in one graph (base + adapter features + real error triangles). This is the
# "hybrid graphs from before" but with the instruct-fine-tuned base transcoders.
# ===========================================================================
ADAPTER=siddharthmb/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860
HYBRID_OUT=$LARGE_ARTIFACTS_DIR/transcoder-adapters/base_adapter_comparisons/hybrid_finetuned_ftL0-24-25
BASE_FEATURES=siddharthmb/2026.TA.features_gemma-2-2b_gemmascope_width_16k_average_l0_76_ms100000_ml1024_tk2_h12ad59325ffd
ADAPTER_FEATURES=siddharthmb/2026.TA.features_2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr_he94e9602bafa

# Serve the ALREADY-BUILT hybrid graphs (job 16179855; 12 interesting_small prompts). Feature
# examples load from HuggingFace (scans baked in). Open http://localhost:8044 (forward the port).
serve_hybrid() {
  uv run --extra viz python -m analysis.attribution.serve_comparison_graphs \
    --graph_file_dir "$HYBRID_OUT" --port 8044
}

# Rebuild the hybrid graphs from scratch (GPU job on jagupard), fine-tuned base transcoders from HF.
rebuild_hybrid() {
  ./sh/sbatch --gres=gpu:1 --constraint=48G --mem=64G --partition=jag-standard --time=0-04:00:00 --job-name=hybrid_ft \
    ./run_on_gpu/run_combined_attribution.sh \
      --adapter_checkpoint "$ADAPTER" --base_model "$BASE" \
      --prompts "$PROMPTS" --prompt_format chat \
      --gemmascope_width "$WIDTH" --gemmascope_l0 "$L0" \
      --finetuned_transcoder_dir "$FT" --finetuned_layers $LAYERS \
      --base_feature_data_path "$BASE_FEATURES" --adapter_feature_data_path "$ADAPTER_FEATURES" \
      --max_feature_nodes 4096 --batch_size 4 --max_n_logits 5 --max_error_nodes 32 \
      --run_name hybrid_finetuned_ftL0-24-25 --output_dir "$HYBRID_OUT"
}

# ===========================================================================
# Alternatively, a fully-materialized 26-layer transcoder set (fine-tune baked in)
# for tools that take --transcoder_set directly (no per-layer flags):
#   local: $LARGE_ARTIFACTS_DIR/transcoder-adapters/finetuned_transcoder_sets/gemma2_2b_width16k_l0_76nearest_ftL0-24-25_sparsity
#   (re)build: analysis/attribution/export_finetuned_transcoder_set.py --finetune_dir "$FT" --finetuned_layers 0 24 25
# ===========================================================================

echo "Source this file, then run one of: serve_bundled | rebuild_comparison | production_serve | evaluate | serve_hybrid | rebuild_hybrid"
