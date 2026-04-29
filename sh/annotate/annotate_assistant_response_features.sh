#!/usr/bin/env bash

# Ex: ./sh/annotate/annotate_assistant_response_features.sh --data_dir /nlp/scr/siddharth/sparse-adaptation/feature_data/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl147_20260416_151353_15175430
#
# Annotate small collected feature test set: ./sh/annotate/annotate_assistant_response_features.sh --data_dir /nlp/scr/siddharth/sparse-adaptation/feature_data/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860_20260429_124541_15292235
#
# Ex with output and thresholds: ./sh/annotate/annotate_assistant_response_features.sh --data_dir /nlp/scr/siddharth/sparse-adaptation/feature_data/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl147_20260416_151353_15175430 --annotations_file /nlp/scr/siddharth/sparse-adaptation/feature_data/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl147_20260416_151353_15175430/feature_annotations.json --assistant_fraction_threshold 0.35 --marker_fraction_threshold 0.20 --density_lift_threshold 2.0 --top_k 50
#
# By default, annotations are saved to <data_dir>/feature_annotations.json.
# Default behavior archives any existing annotations file to <data_dir>/archive/ and writes a fresh file, so threshold changes produce clean results.
# Use --merge to keep the existing annotations file and add newly detected tags into it instead of archiving/replacing it.

LOG_DIR="logs/annotate_assistant_response_features"
LOG_PREFIX="local"
source sh/common_logging.sh

uv run python -m analysis.features.annotate_assistant_response_features "$@"
