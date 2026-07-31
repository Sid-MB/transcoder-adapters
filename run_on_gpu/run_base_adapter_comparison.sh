#!/usr/bin/env bash

# Run matched base-vs-adapter circuit-tracer attribution and overlay export.

SLURM_LOG_DIR="logs/attribution"
source run_on_gpu/common.sh

run uv run --extra viz python -m analysis.attribution.run_base_adapter_comparison "$@"
