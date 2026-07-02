#!/usr/bin/env bash
set -euo pipefail

# ── Saved launch command: base GemmaScope ms20000 feature collection, sharded on sc-loprio ──
# This is the exact, ready-to-rerun invocation for the large base-feature collection that the
# unsharded single-GPU job (15992120) could never finish (cancelled at 56% after 22.5h). It
# fans the CPU-bound collection into NUM_SHARDS small low-priority jobs + one merge job that
# exports and uploads to HF. See sh/slurm_loprio_shard_collect_base.sh for the mechanics and
# tunables (ARRAY_CAP, GPU_CONSTRAINT, SHARD_MEM, ...), and run_on_gpu/run_collect_base_sharded.sh
# for the per-job worker.
#
# Usage:
#   ./sh/launch_base_ms20000_loprio.sh                 # 64 shards (default)
#   NUM_SHARDS=32 ./sh/launch_base_ms20000_loprio.sh   # gentler footprint
#
# Notes:
#   - HF_TOKEN is read from the standard secret file below (needed by the merge job's upload).
#   - The collection args below are byte-for-byte the original base_20000 config, so this
#     produces the scientifically identical collection (only the HF repo NAME differs from the
#     stale h5a3a62ce4dd3 placeholder, because the CLI gained batch_size/wandb args since then).

export HF_TOKEN="${HF_TOKEN:-$(cat ~/.shell/secrets/hf_token_write)}"
export NUM_SHARDS="${NUM_SHARDS:-64}"
export SHARD_OUT="${SHARD_OUT:-/nlp/scr/siddharth/transcoder-adapters/feature_data/base_ms20000_sharded}"

exec ./sh/slurm_loprio_shard_collect_base.sh \
  --base_model google/gemma-2-2b \
  --prompt_tokenizer_model google/gemma-2-2b-it \
  --val_data chat:siddharthmb/2026.transcoder-adapters.lmsys-chat-1m-splits fineweb:science-of-finetuning/fineweb-1m-sample \
  --gemmascope_repo google/gemma-scope-2b-pt-transcoders \
  --gemmascope_width width_16k --gemmascope_l0 average_l0_76 --gemmascope_l0_match nearest --gemmascope_n_layers 26 \
  --feature_input_hook ln2.hook_normalized --feature_output_hook hook_mlp_out \
  --base_backend transformerlens --device cuda --dtype bfloat16 \
  --max_samples 20000 --max_length 1024 \
  --top_k 10 --domain_top_k 5 --n_random 0 --context_before 75 --context_after 20 \
  --activation_example_ranges 0.5:1.0,1.0:2.0,2.0:2.5,2.5:3.0,3.0:4.0,4.0:8.0 \
  --activation_range_examples_per_domain 0 \
  --relative_target_domain chat --relative_baseline_domain fineweb \
  --no_per_feature_json
