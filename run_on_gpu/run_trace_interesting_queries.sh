#!/usr/bin/env bash
#
# [implement: transcoder adapters larger feature collection]
# Incremental base-vs-adapter circuit tracing of the interesting_queries prompt set.
#
# Traces every prompt under analysis/attribution/prompts/interesting_queries/<category>/*.txt into ONE
# overlay directory with a shared --run_name ("iq"). Two properties of run_combined_attribution make this
# incremental with ZERO wasted recompute:
#   1. Each graph is named {run_name}__{stem}__h{content_hash}.json and a prompt is SKIPPED if its graph
#      already exists (so re-running only traces NEW or content-CHANGED prompts).
#   2. graph-metadata.json (the dropdown index) is written cumulatively + idempotently, so it simply grows
#      as prompts/categories are added — no separate merge step.
# => After you add prompts (to an existing category or a brand-new category subdir), just re-run this
#    script; only the additions are traced and the served overlay picks them up automatically.
#
# _list_prompt_files is non-recursive, so we invoke once per prompt-bearing directory (root + each subdir),
# all writing to the same $IQ_OUT. Standard budget (4096/5/32) + dtk20 feature scans (~20 examples/dataset
# on node-click) + the deployed adapter sl14793860 — matching the graph_combined_dtk20 headline overlay.
#
# Run on jagupard:
#   ./sh/sbatch --gres=gpu:1 --constraint=48G --mem=128G --cpus-per-task=8 --partition=jag-standard \
#     --job-name=trace_iq ./run_on_gpu/run_trace_interesting_queries.sh
# Any extra args after the script name are appended to every run_combined_attribution call (e.g. a denser
# budget). Overridable via env: IQ_PROMPTS_ROOT, IQ_OUT, IQ_RUN_NAME.
# Serve afterward (then open http://localhost:8048):
#   uv run --extra viz python -m analysis.attribution.serve_comparison_graphs --graph_file_dir "$IQ_OUT" --port 8048

SLURM_LOG_DIR="logs/combined_attribution"
source run_on_gpu/common.sh

uv sync --extra viz --inexact

PROMPTS_ROOT="${IQ_PROMPTS_ROOT:-analysis/attribution/prompts/interesting_queries}"
OUT="${IQ_OUT:-/nlp/scr/siddharth/transcoder-adapters/graph_interesting_queries_dtk20}"
RUN_NAME="${IQ_RUN_NAME:-iq}"

ADAPTER=siddharthmb/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860
BASE_SCAN=siddharthmb/2026.TA.features_gemma-2-2b_gemmascope_width_16k_average_l0_76_ms100000_ml1024_tk2_h12ad59325ffd
ADAPTER_SCAN=siddharthmb/2026.TA.features_2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr_he94e9602bafa

traced_any=0
for dir in "$PROMPTS_ROOT" "$PROMPTS_ROOT"/*/; do
  [ -d "$dir" ] || continue
  ls "$dir"/*.txt >/dev/null 2>&1 || continue   # skip dirs with no top-level .txt prompts
  traced_any=1
  run uv run --no-sync python -m analysis.attribution.run_combined_attribution \
    --adapter_checkpoint "$ADAPTER" --base_model google/gemma-2-2b \
    --prompts "$dir" --prompt_format chat \
    --gemmascope_width width_16k --gemmascope_l0 average_l0_76 \
    --base_feature_data_path "$BASE_SCAN" --adapter_feature_data_path "$ADAPTER_SCAN" \
    --max_feature_nodes 4096 --max_n_logits 5 --max_error_nodes 32 --batch_size 4 \
    --run_name "$RUN_NAME" --output_dir "$OUT" "$@"
done

[ "$traced_any" = 1 ] || { echo "[trace_iq] No .txt prompts found under $PROMPTS_ROOT" >&2; exit 1; }

# Tag every graph with a dropdown title_prefix "[<category> · <stem>]" so the served dropdown
# distinguishes harmful/divergent/adv_suffix (run_combined_attribution doesn't set title_prefix).
# Idempotent + filesystem-derived, so it re-tags the full set (including freshly added prompts).
run uv run --no-sync python -m analysis.attribution.annotate_graph_title_prefix \
  --graph_dir "$OUT" --prompts_root "$PROMPTS_ROOT"

echo "[trace_iq] Done. Graphs + cumulative graph-metadata.json in $OUT"
echo "[trace_iq] Serve: uv run --extra viz python -m analysis.attribution.serve_comparison_graphs --graph_file_dir $OUT --port 8048"
