#!/usr/bin/env bash
set -euo pipefail

# ── Fan an ADAPTER feature collection across sc-loprio as N data-parallel shards ──
# Feature collection is inference, so the correct multi-GPU tool is sharding (independent
# processes + lossless merge), NOT DDP. sc-loprio gives 120+ concurrency but preempts by
# REQUEUE, so this is built to be preemption-resilient:
#   * --requeue           -> Slurm auto-restarts preempted shards.
#   * shards skip-if-done -> a finished shard's pickle is never recomputed.
#   * GAP-FILL            -> re-running this launcher submits ONLY the shards whose pickle is
#                            still missing (i.e. "redo the cancelled ones" is just re-running it).
#   * merge gated afterok -> the merge runs once all shards exist, then recombines + uploads.
#
# Usage:
#   export HF_TOKEN=$(cat ~/.shell/secrets/hf_token_write)     # needed by the merge upload
#   NUM_SHARDS=64 \
#   SHARD_OUT=/nlp/scr/siddharth/transcoder-adapters/feature_data/<name>_sharded \
#   ./sh/slurm_loprio_shard_collect_adapter.sh \
#     --model_path <adapter HF repo or path> \
#     --val_data chat:siddharthmb/2026.transcoder-adapters.lmsys-chat-1m-splits fineweb:science-of-finetuning/fineweb-1m-sample \
#     --max_samples 100000 --max_length 1024 \
#     --top_k 10 --domain_top_k 5 --n_random 0 --activation_range_examples_per_domain 0 \
#     --context_before 30 --context_after 8 --no_per_feature_json
#
# To recover after a wave of preemptions: just RE-RUN the exact same command -- it resubmits
# only the missing shards and re-queues the merge.
#
# Tunables (env): NUM_SHARDS (default 64), SHARD_OUT (REQUIRED), ARRAY_CAP (max concurrent,
#   default 64), GPU_GRES (default gpu:1), GPU_CONSTRAINT (GPU-mem class, default "24G|48G";
#   "" = any), SHARD_CPUS (6), SHARD_MEM (48G), MERGE_MEM (160G), SHARD_TIME (8:00:00),
#   HF_HOME_SHARED / HF_HUB_CACHE_SHARED (shared HF cache).

NUM_SHARDS="${NUM_SHARDS:-64}"
SHARD_OUT="${SHARD_OUT:?set SHARD_OUT to a shared output dir}"
# Propagate explicitly (jobs land on nodes without the submitter's shell profile).
LARGE_ARTIFACTS_DIR="${LARGE_ARTIFACTS_DIR:-/nlp/scr/siddharth}"
# Shared HF cache (NOT node-local /scr) so N shards reuse one download instead of 429-ing.
# huggingface_hub prioritizes HF_HUB_CACHE over HF_HOME, so set BOTH.
HF_HOME="${HF_HOME_SHARED:-/nlp/scr/siddharth/.caches/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE_SHARED:-$HF_HOME/hub}"
ARRAY_CAP="${ARRAY_CAP:-64}"
GPU_GRES="${GPU_GRES:-gpu:1}"
GPU_CONSTRAINT="${GPU_CONSTRAINT:-24G|48G}"
# The MERGE loads the full model for the logit lens -> a large adapter (e.g. 16384-feature
# gemma2_2b_huge, ~9GB + activations) OOMs a 24GB card, so give the merge a bigger GPU by
# default. Shards (collection only) are fine on 24G|48G.
MERGE_GPU_CONSTRAINT="${MERGE_GPU_CONSTRAINT:-48G}"
SHARD_CPUS="${SHARD_CPUS:-6}"
SHARD_MEM="${SHARD_MEM:-48G}"
MERGE_MEM="${MERGE_MEM:-160G}"
SHARD_TIME="${SHARD_TIME:-8:00:00}"

mkdir -p "$SHARD_OUT"

CONSTRAINT_ARG=()
[ -n "$GPU_CONSTRAINT" ] && CONSTRAINT_ARG=(--constraint="$GPU_CONSTRAINT")
MERGE_CONSTRAINT_ARG=()
[ -n "$MERGE_GPU_CONSTRAINT" ] && MERGE_CONSTRAINT_ARG=(--constraint="$MERGE_GPU_CONSTRAINT")

# Gap-fill: submit only shards whose pickle is still missing.
MISSING=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  printf -v idx "%03d" "$i"
  [ -f "$SHARD_OUT/shard_${idx}.pkl" ] || MISSING+=("$i")
done

ARRAY_ID=""
if [ ${#MISSING[@]} -gt 0 ]; then
  ARRAY_SPEC=$(IFS=,; echo "${MISSING[*]}")   # e.g. "3,7,12"
  ARRAY_ID=$(sbatch --parsable \
    --account=nlp --partition=sc-loprio --requeue \
    --array="${ARRAY_SPEC}%${ARRAY_CAP}" \
    --gres="$GPU_GRES" "${CONSTRAINT_ARG[@]}" --cpus-per-task="$SHARD_CPUS" --mem="$SHARD_MEM" --time="$SHARD_TIME" \
    --job-name=adapter_shard \
    --export=ALL,LARGE_ARTIFACTS_DIR="${LARGE_ARTIFACTS_DIR}",HF_HOME="${HF_HOME}",HF_HUB_CACHE="${HF_HUB_CACHE}",NUM_SHARDS="${NUM_SHARDS}",SHARD_OUT="${SHARD_OUT}" \
    ./run_on_gpu/run_collect_adapter_sharded.sh "$@")
  echo "Shard array submitted: job $ARRAY_ID  (${#MISSING[@]} missing of ${NUM_SHARDS} shards, up to ${ARRAY_CAP} at once)"
else
  echo "All ${NUM_SHARDS} shard pickles already present in $SHARD_OUT; skipping shard array, going straight to merge."
fi

# Merge job: waits for the shard array (if any was submitted), then recombines + uploads.
DEP_ARG=()
[ -n "$ARRAY_ID" ] && DEP_ARG=(--dependency=afterok:"$ARRAY_ID")
MERGE_ID=$(sbatch --parsable \
  --account=nlp --partition=sc-loprio --requeue \
  "${DEP_ARG[@]}" \
  --gres="$GPU_GRES" "${MERGE_CONSTRAINT_ARG[@]}" --cpus-per-task=8 --mem="$MERGE_MEM" --time=6:00:00 \
  --job-name=adapter_merge \
  --export=ALL,LARGE_ARTIFACTS_DIR="${LARGE_ARTIFACTS_DIR}",HF_HOME="${HF_HOME}",HF_HUB_CACHE="${HF_HUB_CACHE}",MERGE_SHARDS=1,SHARD_OUT="${SHARD_OUT}" \
  ./run_on_gpu/run_collect_adapter_sharded.sh "$@")
echo "Merge job submitted:   job $MERGE_ID  ${ARRAY_ID:+(afterok:$ARRAY_ID)}"
echo
echo "Watch:    squeue -u \$USER | grep adapter_"
echo "Recover:  re-run this exact command to resubmit only the shards that got preempted/cancelled."
echo "Output:   $SHARD_OUT   (shard_*.pkl, then the merged + uploaded collection)"
