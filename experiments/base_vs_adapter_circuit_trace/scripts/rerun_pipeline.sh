#!/usr/bin/env bash
#
# Re-run the agree-vs-diverge circuit-tracing experiment end to end with the enriched,
# behaviorally-diverse prompt set (refusal pivots, format obedience, persona, sycophancy,
# hedging, system-prompt following, context use -- via measure_agreement.py assistant_prefix).
# Submitted to jagupard (jag-standard, 48G) via ./sh/sbatch.

SLURM_LOG_DIR="logs/experiment_rerun"
source run_on_gpu/common.sh

# run_combined_attribution needs circuit_tracer (the viz extra); --inexact so the shared venv
# used by other running jobs is not pruned.
uv sync --extra viz --inexact

ROOT=experiments/base_vs_adapter_circuit_trace
ADAPTER=siddharthmb/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860

# helpers/paths/__init__.py requires this at import time (run_combined_attribution imports it).
# Graphs go to --output_dir below regardless; this just satisfies the import + any incidental use.
export LARGE_ARTIFACTS_DIR="${LARGE_ARTIFACTS_DIR:-/nlp/scr/siddharth}"

# Clean stale prompts/graphs so the rebuild reflects only the new selection (regenerable outputs).
rm -f "$ROOT"/prompts/agree/*.txt "$ROOT"/prompts/diverge/*.txt
rm -f "$ROOT"/graphs/agree/*.json "$ROOT"/graphs/diverge/*.json "$ROOT"/graphs/all/*.json

# 1) Measure base-vs-instruct next-token agreement on the enriched candidates; select 14/bucket.
run env PYTHONPATH=. uv run --no-sync python "$ROOT/scripts/measure_agreement.py" --n_select 14

# 2) Build combined full-replacement graphs. Empty --base_feature_data_path => counting-only
#    (no feature-example download), matching the original experiment's bare graphs.
run uv run --no-sync python -m analysis.attribution.run_combined_attribution --adapter_checkpoint "$ADAPTER" --base_model google/gemma-2-2b --prompts "$ROOT/prompts/agree" --prompt_format chat --gemmascope_width width_16k --gemmascope_l0 average_l0_76 --base_feature_data_path "" --max_feature_nodes 4096 --batch_size 4 --max_n_logits 5 --max_error_nodes 32 --run_name agree --output_dir "$ROOT/graphs/agree"
run uv run --no-sync python -m analysis.attribution.run_combined_attribution --adapter_checkpoint "$ADAPTER" --base_model google/gemma-2-2b --prompts "$ROOT/prompts/diverge" --prompt_format chat --gemmascope_width width_16k --gemmascope_l0 average_l0_76 --base_feature_data_path "" --max_feature_nodes 4096 --batch_size 4 --max_n_logits 5 --max_error_nodes 32 --run_name diverge --output_dir "$ROOT/graphs/diverge"

# 3) Aggregate node composition + regenerate the figure.
run env PYTHONPATH=. uv run --no-sync python "$ROOT/scripts/analyze_graphs.py"

# 4) Rebuild the combined dropdown (graph JSONs only; exclude manifests).
cp "$ROOT"/graphs/agree/agree__*.json "$ROOT"/graphs/diverge/diverge__*.json "$ROOT"/graphs/all/ 2>/dev/null
echo "PIPELINE DONE: $(ls "$ROOT"/graphs/all/*.json 2>/dev/null | wc -l) graphs in graphs/all"
