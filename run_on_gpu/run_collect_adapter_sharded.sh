#!/usr/bin/env bash

# ── Sharded ADAPTER feature-collection worker ─────────────────────────────────
# Mirror of run_collect_base_sharded.sh, for the adapter collector. Feature collection is
# pure inference, so multi-GPU = data-parallel sharding (NOT DDP): each shard collects a
# disjoint slice and pickles its collector state; a merge job recombines them losslessly.
# Because the config uses --n_random 0 and --activation_range_examples_per_domain 0, the
# merged result is IDENTICAL to a single unsharded run (see FeatureCollector.merge_from).
#
# Two modes, selected by environment variable:
#   SHARD (default):  processes shard $SLURM_ARRAY_TASK_ID of $NUM_SHARDS, writes
#                     $SHARD_OUT/shard_<i>.pkl, skips export/upload. A shard whose pickle
#                     already exists exits immediately (cheap re-run after preemption).
#   MERGE_SHARDS=1:   merges $SHARD_OUT/shard_*.pkl, exports to disk (--no-upload), then
#                     pushes via the salvage uploader (idempotent; works whether or not the
#                     repo was already reserved). Needs HF_TOKEN + the big-RAM box.
#
# Pass the collection config (--model_path, --val_data, sample/top-k knobs) as "$@" -- the
# SAME args for both modes. Do NOT include --num_shards/--shard_index/--merge_shards/
# --output_dir/--*upload* in "$@"; this script supplies them.
#
# Normally invoked via sh/slurm_loprio_shard_collect_adapter.sh, not directly.
# Logs: logs/collect_features/<timestamp>_<job_id>.{out,err}
# ─────────────────────────────────────────────────────────────────────────────

SLURM_LOG_DIR="logs/collect_features"
source run_on_gpu/common.sh

if [ "${MERGE_SHARDS:-0}" = "1" ]; then
  run uv run --no-sync python -m analysis.features.collect_feature_activations \
    --merge_shards "${SHARD_OUT:?set SHARD_OUT}/shard_*.pkl" \
    --output_dir "$SHARD_OUT" \
    --no-upload_circuit_tracer_features_to_hub \
    "$@"
  run uv run --no-sync python -m misc_scripts.upload_existing_feature_collection "$SHARD_OUT"
else
  # Read model + datasets from the shared HF cache (HF_HOME/HF_HUB_CACHE on a cross-cluster
  # filesystem, NOT node-local /scr) so N shards don't each re-download and hit HTTP 429.
  run uv run --no-sync python -m analysis.features.collect_feature_activations \
    --num_shards "${NUM_SHARDS:?set NUM_SHARDS}" \
    --shard_index "${SLURM_ARRAY_TASK_ID:-0}" \
    --output_dir "${SHARD_OUT:?set SHARD_OUT}" \
    --no-upload_circuit_tracer_features_to_hub \
    "$@"
fi
