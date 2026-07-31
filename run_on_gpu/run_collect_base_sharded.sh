#!/usr/bin/env bash

# ── Sharded BASE-model (GemmaScope) feature-collection worker ─────────────────
# Designed for sc-loprio array jobs: a CPU-bound dense accumulation (the bottleneck
# for GemmaScope) is split into many independent shards that each persist their
# collector state, then a single merge job recombines + exports + uploads. Because
# the base config uses --n_random 0 and no activation ranges, the merged result is
# IDENTICAL to a single unsharded run (see FeatureCollector.merge_from).
#
# Two modes, selected by environment variable:
#   SHARD (default):  processes shard $SLURM_ARRAY_TASK_ID of $NUM_SHARDS, writes
#                     $SHARD_OUT/shard_<i>.pkl, skips export/upload. One small GPU +
#                     a few CPU cores each; the GPU is barely used (forward pass only).
#   MERGE_SHARDS=1:   merges $SHARD_OUT/shard_*.pkl into one collection, runs the
#                     export path (packed cache + metadata + annotations) and uploads
#                     to HF. Needs the big-RAM box (holds the full merged collector)
#                     and HF_TOKEN in the environment.
#
# Pass the collection config (model, gemmascope, val_data, sample/top-k knobs) as
# "$@" -- the SAME args for both modes. Do NOT include --num_shards/--shard_index/
# --merge_shards/--output_dir/--*upload* in "$@"; this script supplies them.
#
# Usage (normally via sh/slurm_loprio_shard_collect_base.sh, not directly):
#   sbatch --array=0-63 --export=ALL,NUM_SHARDS=64,SHARD_OUT=/path \
#          ./run_on_gpu/run_collect_base_sharded.sh <collection args>
#   sbatch --dependency=afterok:<arr> --export=ALL,MERGE_SHARDS=1,SHARD_OUT=/path \
#          ./run_on_gpu/run_collect_base_sharded.sh <collection args>
#
# Logs: logs/collect_base_features/<timestamp>_<job_id>.{out,err}
# ─────────────────────────────────────────────────────────────────────────────

SLURM_LOG_DIR="logs/collect_base_features"
source run_on_gpu/common.sh

# viz extra (circuit_tracer) without pruning the shared venv; --no-sync at runtime.
uv sync --extra viz --inexact

if [ "${MERGE_SHARDS:-0}" = "1" ]; then
  # Merge the shard pickles and EXPORT to disk with --no-upload: the collector's
  # built-in upload path refuses an already-reserved repo (the original killed run
  # reserved it), so we export here, then push via the salvage uploader, which is
  # designed to upload on-disk data into an existing reserved repo. With sharding
  # flags excluded from the repo fingerprint (analysis/features/hub_upload.py), this
  # lands in the SAME deterministic repo a single unsharded run would have produced.
  run uv run --no-sync python -m analysis.features.collect_base_feature_activations \
    --merge_shards "${SHARD_OUT:?set SHARD_OUT}/shard_*.pkl" \
    --output_dir "$SHARD_OUT" \
    --no-upload_circuit_tracer_features_to_hub \
    "$@"
  run uv run --no-sync python -m misc_scripts.upload_existing_feature_collection "$SHARD_OUT"
else
  # Read model + datasets from the shared, pre-warmed HF cache (HF_HOME). We DON'T use
  # HF_HUB_OFFLINE: the load path makes a metadata model_info() API call that offline mode
  # forbids. Instead we rely on (a) the warm cache so no FILES are re-downloaded (the 429s
  # were from 64 parallel xet file downloads, not metadata) and (b) ARRAY_CAP throttling so
  # only a handful of lightweight metadata calls hit HF at once.
  run uv run --no-sync python -m analysis.features.collect_base_feature_activations \
    --num_shards "${NUM_SHARDS:?set NUM_SHARDS}" \
    --shard_index "${SLURM_ARRAY_TASK_ID:-0}" \
    --output_dir "${SHARD_OUT:?set SHARD_OUT}" \
    --no-upload_circuit_tracer_features_to_hub \
    "$@"
fi
