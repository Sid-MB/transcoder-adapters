#!/usr/bin/env bash
set -euo pipefail

# ── Auto-queue: rebuild the base-vs-adapter overlay when a training run finishes ──
# Submitted as a lightweight CPU job (sc-freecpu). It holds NO GPU while waiting: it polls
# HF until the trained adapter has been pushed (final model-*.safetensors present), then
# submits the GPU chain on jagupard:
#   1) collect the NEW adapter's feature activations (sparse -> fast)            [GPU]
#   2) run_combined_attribution: base GemmaScope (reusable ms100000) vs NEW adapter,
#      over the comprehensive prompt set -> one combined overlay graph per prompt [GPU, afterok:1]
#
# Why CPU-poll instead of a Slurm dependency: training runs on an external RunPod B200, not
# Slurm, so there's no job to depend on -- the adapter appearing on HF is the cross-system
# "done" signal.
#
# Submit with:
#   ./sh/sbatch --partition=sc-freecpu --cpus-per-task=2 --mem=4G --time=3-00:00:00 \
#       --job-name=autoqueue_overlay ./sh/autoqueue_overlay_rebuild.sh
# (./sh/sbatch adds --account=nlp --export=ALL so LARGE_ARTIFACTS_DIR/HF_TOKEN propagate.)

# ── Parameters (override via env at submit time) ──────────────────────────────
NEW_ADAPTER="${NEW_ADAPTER:-siddharthmb/2026.TA.gemma2_2b_huge_tc16384_decb_l1w0.0003_norm_sch_tarbb_lb2.0_ln1.0_dr500000_lr2e-04_bs8_sl}"
BASE_FEATURES="${BASE_FEATURES:-siddharthmb/2026.TA.features_gemma-2-2b_gemmascope_width_16k_average_l0_76_ms100000_ml1024_tk1_hf83e96d574d2}"
PROMPTS="${PROMPTS:-analysis/attribution/prompts/comprehensive}"
LARGE_ARTIFACTS_DIR="${LARGE_ARTIFACTS_DIR:-/nlp/scr/$USER}"
WORK="${WORK:-$LARGE_ARTIFACTS_DIR/transcoder-adapters/overlay_huge}"        # shared scratch (jagupard-visible)
ADAPTER_FEATURES_DIR="$WORK/adapter_features"
GRAPH_OUT="$WORK/graph"
# Node budgets. MAX_FEATURE_NODES=32768 (4x attr_big) keeps essentially every
# meaningful-influence feature while staying loadable in the viewer (the long tail beyond
# this is near-zero influence; the viewer's pruning slider thins further at view time).
# Compute is ~unchanged by this cap -- influence is computed for ALL active features; the
# cap only limits what's stored. MAX_N_LOGITS/MAX_ERROR_NODES stay at the attr_big values.
MAX_FEATURE_NODES="${MAX_FEATURE_NODES:-32768}"
MAX_N_LOGITS="${MAX_N_LOGITS:-10}"
MAX_ERROR_NODES="${MAX_ERROR_NODES:-64}"
# Adapter feature collection scale: 100000 = all of the val split (lmsys val=100k,
# fineweb val=100k), matching the reusable base ms100000 for a fair comparison. ~50M tokens
# already saturates the per-feature proportion stats; the collector reads the val split only.
COLLECT_MAX_SAMPLES="${COLLECT_MAX_SAMPLES:-100000}"
COLLECT_MAX_LENGTH="${COLLECT_MAX_LENGTH:-1024}"
POLL_SECONDS="${POLL_SECONDS:-300}"
# The adapter collection is fanned across sc-loprio as NUM_SHARDS data-parallel shards
# (preemption-resilient: --requeue + skip-if-done + gap-fill re-runs). The shards read from a
# SHARED HF cache (NOT node-local /scr) so the new adapter is downloaded ONCE here (pre-warm)
# rather than 64x at once (which would 429). huggingface_hub prefers HF_HUB_CACHE over HF_HOME,
# so set both.
NUM_SHARDS="${NUM_SHARDS:-64}"
# FORCE the shared cache (don't inherit): the trigger often runs with a node-local HF_HOME
# (/scr/$USER) from --export=ALL, which made the pre-warm download to node-local disk while the
# shards read the shared cache -> cache miss. Match the launcher's shared default exactly.
HF_HOME=/nlp/scr/siddharth/.caches/huggingface
HF_HUB_CACHE=/nlp/scr/siddharth/.caches/huggingface/hub
export HF_HOME HF_HUB_CACHE

mkdir -p "$WORK"
echo "[autoqueue] waiting for adapter on HF: $NEW_ADAPTER"
echo "[autoqueue] poll every ${POLL_SECONDS}s; base=$BASE_FEATURES; prompts=$PROMPTS"

# ── Poll HF until the adapter's final weights are pushed ──────────────────────
until uv run --no-sync python - "$NEW_ADAPTER" <<'PY'
import sys
from huggingface_hub import HfApi
repo = sys.argv[1]
try:
    files = HfApi().list_repo_files(repo)
except Exception as e:
    print(f"[autoqueue] repo not ready ({e})"); sys.exit(1)
has_model = any(f.endswith(".safetensors") for f in files)
print(f"[autoqueue] {repo}: {len(files)} files, model weights present={has_model}")
sys.exit(0 if has_model else 1)
PY
do
  sleep "$POLL_SECONDS"
done
echo "[autoqueue] adapter detected on HF — submitting GPU chain."

# ── 1) Collect NEW adapter features, SHARDED across sc-loprio (data-parallel) ──────
# First pre-warm the shared HF cache (the new adapter + datasets) so the 64 shards read from
# cache instead of all downloading the adapter at once (429 thundering-herd, cf. base20k).
echo "[autoqueue] pre-warming shared HF cache (adapter + datasets) into $HF_HUB_CACHE ..."
uv run --no-sync python - "$NEW_ADAPTER" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1])  # the adapter (model repo)
for ds in ("siddharthmb/2026.transcoder-adapters.lmsys-chat-1m-splits",
           "science-of-finetuning/fineweb-1m-sample"):
    snapshot_download(ds, repo_type="dataset")
print("[autoqueue] pre-warm complete")
PY

# Fan the collection across NUM_SHARDS shards (preemption-resilient launcher); capture the
# MERGE job id so the attribution can depend on it. The merge writes the combined collection
# to $ADAPTER_FEATURES_DIR/circuit_tracer_features (shared /nlp/scr, visible to jag-standard).
echo "[autoqueue] launching sharded adapter collection ($NUM_SHARDS shards on sc-loprio) ..."
LAUNCH_OUT=$(NUM_SHARDS="$NUM_SHARDS" SHARD_OUT="$ADAPTER_FEATURES_DIR" \
  ./sh/slurm_loprio_shard_collect_adapter.sh \
    --model_path "$NEW_ADAPTER" \
    --val_data chat:siddharthmb/2026.transcoder-adapters.lmsys-chat-1m-splits fineweb:science-of-finetuning/fineweb-1m-sample \
    --max_samples "$COLLECT_MAX_SAMPLES" --max_length "$COLLECT_MAX_LENGTH" \
    --top_k 10 --domain_top_k 5 --n_random 0 --activation_range_examples_per_domain 0 \
    --context_before 30 --context_after 8 --no_per_feature_json)
echo "$LAUNCH_OUT"
COLLECT_MERGE_ID=$(echo "$LAUNCH_OUT" | grep -oE "Merge job submitted: *job [0-9]+" | grep -oE "[0-9]+$")
[ -z "$COLLECT_MERGE_ID" ] && { echo "[autoqueue] ERROR: could not parse merge job id from launcher output"; exit 1; }
echo "[autoqueue] adapter-collection merge job: $COLLECT_MERGE_ID -> $ADAPTER_FEATURES_DIR"

# ── 2) Combined attribution: base ms100000 vs NEW adapter (afterok) ───────────
ATTR_ID=$(./sh/sbatch \
  --gres=gpu:1 --constraint=48G --mem=128G --partition=jag-standard \
  --job-name=huge_overlay_attr --time=1-00:00:00 --parsable \
  --dependency=afterok:"$COLLECT_MERGE_ID" \
  ./run_on_gpu/run_combined_attribution.sh \
    --adapter_checkpoint "$NEW_ADAPTER" --base_model google/gemma-2-2b \
    --prompts "$PROMPTS" --prompt_format chat \
    --gemmascope_width width_16k --gemmascope_l0 average_l0_76 \
    --base_feature_data_path "$BASE_FEATURES" \
    --adapter_feature_data_path "$ADAPTER_FEATURES_DIR/circuit_tracer_features" \
    --max_feature_nodes "$MAX_FEATURE_NODES" --batch_size 4 \
    --max_n_logits "$MAX_N_LOGITS" --max_error_nodes "$MAX_ERROR_NODES" \
    --run_name huge_overlay --output_dir "$GRAPH_OUT")
echo "[autoqueue] attribution job: $ATTR_ID (afterok:$COLLECT_MERGE_ID) -> $GRAPH_OUT"
echo "[autoqueue] DONE submitting. Serve afterward with:"
echo "  uv run --extra viz python -m analysis.attribution.serve_comparison_graphs --graph_file_dir $GRAPH_OUT --port 8044"
