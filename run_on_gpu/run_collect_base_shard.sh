#!/usr/bin/env bash

# Array-task wrapper for sharded BASE/GemmaScope feature collection. Injects
# --shard_index from $SLURM_ARRAY_TASK_ID so each array task collects a disjoint shard
# of the data (the dense accumulation is CPU-bound, so sharding fans it across independent
# GPU nodes with no contention). Each shard pickles its partial FeatureCollector to
# <output_dir>/shard_<i>.pkl and skips export; recombine afterward with a single
# --merge_shards run (see run_collect_base_features.sh) which uploads the merged collection.
#
# Submit: ./sh/sbatch --array=0-<N-1> ... ./run_on_gpu/run_collect_base_shard.sh <collector args incl. --num_shards N>
# Logs: logs/collect_base_features/<job>.{out,err}

SLURM_LOG_DIR="logs/collect_base_features"
source run_on_gpu/common.sh

uv sync --extra viz --inexact

SHARD="${SLURM_ARRAY_TASK_ID:-0}"
run uv run --no-sync python -m analysis.features.collect_base_feature_activations \
  "$@" --shard_index "$SHARD" --wandb_run_name "base_ms100000_shard${SHARD}"
