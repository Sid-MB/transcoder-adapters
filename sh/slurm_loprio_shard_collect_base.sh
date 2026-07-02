#!/usr/bin/env bash
set -euo pipefail

# ── Fan a BASE GemmaScope feature collection across sc-loprio as N small shards ──
# GemmaScope collection is CPU-bound (dense per-feature accumulation over ~425k
# features), so the win is MANY small GPU+few-core jobs, not one big GPU. sc-loprio
# is low-priority (auto-yields to others' normal-priority jobs) and preempts by
# REQUEUE with no grace period -- so keep shards small (a requeue reruns a shard from
# scratch; small shards make that cheap). The merge step recombines shards into a
# result identical to a single run, then uploads to HF.
#
# Usage:
#   export HF_TOKEN=$(cat ~/.shell/secrets/hf_token_write)        # needed by the merge job
#   NUM_SHARDS=64 \
#   SHARD_OUT=/nlp/scr/siddharth/transcoder-adapters/feature_data/base_ms20000_sharded \
#   ./sh/slurm_loprio_shard_collect_base.sh \
#     --base_model google/gemma-2-2b --prompt_tokenizer_model google/gemma-2-2b-it \
#     --val_data chat:siddharthmb/2026.transcoder-adapters.lmsys-chat-1m-splits fineweb:science-of-finetuning/fineweb-1m-sample \
#     --gemmascope_width width_16k --gemmascope_l0 average_l0_76 \
#     --max_samples 20000 --max_length 1024 \
#     --top_k 10 --domain_top_k 5 --n_random 0 --activation_range_examples_per_domain 0 \
#     --no_per_feature_json --context_before 75 --context_after 20
#
# Tunables (env):
#   NUM_SHARDS   number of shards / array tasks (default 64). Smaller shard = cheaper requeue.
#   SHARD_OUT    shared output dir for shard_<i>.pkl + final export (REQUIRED).
#   ARRAY_CAP    max array tasks running at once (default = NUM_SHARDS, i.e. all). Lower it
#                to throttle footprint (e.g. ARRAY_CAP=30).
#   GPU_GRES     GPU request per task (default gpu:1 = any type; the GPU is barely used).
#   GPU_CONSTRAINT  node-feature constraint for GPU memory class (default "24G|48G"):
#                excludes the OOM-prone 12G volta/pascal cards AND politely skips the
#                premium 80G a100 / 141G h200 boxes, leaving a ~340-GPU mid-tier pool
#                (a5000/3090/a6000/a6000ada/titanrtx/l40s/a40). Set "" to allow any.
#   SHARD_CPUS   CPU cores per shard (default 6).   SHARD_MEM  per-shard RAM (default 48G).
#   MERGE_MEM    RAM for the merge job (default 160G; it holds the full merged collector).
#   SHARD_TIME   walltime per shard (default 8:00:00, generous to absorb requeues).

NUM_SHARDS="${NUM_SHARDS:-64}"
SHARD_OUT="${SHARD_OUT:?set SHARD_OUT to a shared output dir}"
# Jobs land on nodes without the submitter's shell profile, so LARGE_ARTIFACTS_DIR
# (required by helpers.paths) must be propagated explicitly via --export -- inheriting
# it through --export=ALL only works when the SUBMITTING shell already had it set.
LARGE_ARTIFACTS_DIR="${LARGE_ARTIFACTS_DIR:-/nlp/scr/siddharth}"
# HF cache MUST be on a cross-cluster shared filesystem (NOT node-local /scr): otherwise
# every shard on every node re-downloads the GemmaScope transcoders, and 64 simultaneous
# downloads get HTTP 429 (rate limited). With a shared, pre-warmed cache the shards reuse
# the files and only make a few lightweight metadata calls (throttled via ARRAY_CAP).
# NOTE: huggingface_hub prioritizes HF_HUB_CACHE over HF_HOME for downloads, so we must set
# BOTH -- otherwise an inherited node-local HF_HUB_CACHE silently wins and the transcoders
# land on per-node disk (exactly the bug that caused repeated 429s).
HF_HOME="${HF_HOME_SHARED:-/nlp/scr/siddharth/.caches/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE_SHARED:-$HF_HOME/hub}"
ARRAY_CAP="${ARRAY_CAP:-$NUM_SHARDS}"
GPU_GRES="${GPU_GRES:-gpu:1}"
GPU_CONSTRAINT="${GPU_CONSTRAINT:-24G|48G}"
# Merge loads the full model for the logit lens -> give it a bigger GPU (shards are fine on 24G).
MERGE_GPU_CONSTRAINT="${MERGE_GPU_CONSTRAINT:-48G}"
SHARD_CPUS="${SHARD_CPUS:-6}"
SHARD_MEM="${SHARD_MEM:-48G}"
MERGE_MEM="${MERGE_MEM:-160G}"
SHARD_TIME="${SHARD_TIME:-8:00:00}"
LAST=$((NUM_SHARDS - 1))

mkdir -p "$SHARD_OUT"

# Only pass --constraint when GPU_CONSTRAINT is non-empty.
CONSTRAINT_ARG=()
[ -n "$GPU_CONSTRAINT" ] && CONSTRAINT_ARG=(--constraint="$GPU_CONSTRAINT")
MERGE_CONSTRAINT_ARG=()
[ -n "$MERGE_GPU_CONSTRAINT" ] && MERGE_CONSTRAINT_ARG=(--constraint="$MERGE_GPU_CONSTRAINT")

# Optional: make the shard array wait for a cache-warmup job (afterok). Set WARMUP_DEP to
# that job id so the offline shards only start once the shared HF cache is fully populated.
ARRAY_DEP=()
[ -n "${WARMUP_DEP:-}" ] && ARRAY_DEP=(--dependency=afterok:"${WARMUP_DEP}")

# 1) Shard array — GAP-FILL: submit only shards whose pickle is still missing, so re-running
#    this launcher after a wave of preemptions/failures redoes ONLY the cancelled shards.
MISSING=()
for i in $(seq 0 "$LAST"); do
  printf -v idx "%03d" "$i"
  [ -f "$SHARD_OUT/shard_${idx}.pkl" ] || MISSING+=("$i")
done

ARRAY_ID=""
if [ ${#MISSING[@]} -gt 0 ]; then
  ARRAY_SPEC=$(IFS=,; echo "${MISSING[*]}")   # e.g. "14" or "3,7,12"
  ARRAY_ID=$(sbatch --parsable \
    --account=nlp --partition=sc-loprio --requeue \
    --array="${ARRAY_SPEC}%${ARRAY_CAP}" "${ARRAY_DEP[@]}" \
    --gres="$GPU_GRES" "${CONSTRAINT_ARG[@]}" --cpus-per-task="$SHARD_CPUS" --mem="$SHARD_MEM" --time="$SHARD_TIME" \
    --job-name=base20k_shard \
    --export=ALL,LARGE_ARTIFACTS_DIR="${LARGE_ARTIFACTS_DIR}",HF_HOME="${HF_HOME}",HF_HUB_CACHE="${HF_HUB_CACHE}",NUM_SHARDS="${NUM_SHARDS}",SHARD_OUT="${SHARD_OUT}" \
    ./run_on_gpu/run_collect_base_sharded.sh "$@")
  echo "Shard array submitted: job $ARRAY_ID  (${#MISSING[@]} missing of ${NUM_SHARDS} shards, up to ${ARRAY_CAP} at once)"
else
  echo "All ${NUM_SHARDS} shard pickles already present in $SHARD_OUT; skipping shard array, going straight to merge."
fi

# 2) Merge job: waits for the shard array (if any), then exports + uploads to HF.
DEP_ARG=()
[ -n "$ARRAY_ID" ] && DEP_ARG=(--dependency=afterok:"$ARRAY_ID")
MERGE_ID=$(sbatch --parsable \
  --account=nlp --partition=sc-loprio --requeue \
  "${DEP_ARG[@]}" \
  --gres="$GPU_GRES" "${MERGE_CONSTRAINT_ARG[@]}" --cpus-per-task=8 --mem="$MERGE_MEM" --time=6:00:00 \
  --job-name=base20k_merge \
  --export=ALL,LARGE_ARTIFACTS_DIR="${LARGE_ARTIFACTS_DIR}",HF_HOME="${HF_HOME}",HF_HUB_CACHE="${HF_HUB_CACHE}",MERGE_SHARDS=1,SHARD_OUT="${SHARD_OUT}" \
  ./run_on_gpu/run_collect_base_sharded.sh "$@")
echo "Merge job submitted:   job $MERGE_ID  ${ARRAY_ID:+(afterok:$ARRAY_ID)}"
echo
echo "Watch:    squeue -u \$USER | grep base20k"
echo "Recover:  re-run this exact command to resubmit only the shards that got preempted/cancelled."
echo "Output:   $SHARD_OUT   (shard_*.pkl, then the merged + uploaded collection)"
