#!/usr/bin/env bash

MODEL_PATH="siddharthmb/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860"
GRAPH_DIR="$SCRATCH_DIR/sparse-adaptation/ct_attribution_graphs/"
DATETIME_FILESAFE=$(date +"Analysis_%Y%m%d_%H%M%S")
INPUT_PROMPT="Hello"

uv run circuit-tracer attribute --slug="$DATETIME_FILESAFE" --graph_file_dir="$GRAPH_DIR" --transcoder_set="$MODEL_PATH" --prompt="$INPUT_PROMPT" --server

