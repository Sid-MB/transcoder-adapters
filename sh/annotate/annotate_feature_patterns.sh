#!/usr/bin/env bash

# Ex: ./sh/annotate/annotate_feature_patterns.sh --data_dir /nlp/scr/siddharth/sparse-adaptation/feature_data/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860_20260430_002526_15298226
#
# Ex with selected annotators and separate output:
# ./sh/annotate/annotate_feature_patterns.sh --data_dir /path/to/run --annotations_file /tmp/feature_patterns.json --annotators logit_effect,token_surface,reasoning_move --top_k 50
#
# By default, annotations are saved to <data_dir>/feature_annotations.json.
# Default behavior updates only these annotators' tags/scores in place.
# Use --replace_all to archive any existing annotations file to <data_dir>/archive/ and write a fresh file.

LOG_DIR="logs/annotate_feature_patterns"
LOG_PREFIX="local"
source sh/common_logging.sh
source sh/annotate/common.sh

ensure_writable_uv_cache
uv run python -m analysis.features.annotate.annotate_feature_patterns "$@"

# After, restart visualizer to see changes.
