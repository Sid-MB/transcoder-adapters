#!/usr/bin/env sh

# First, if needed, run `sh/slurm_batch_collect_features.sh` to create the feature data.
# And, run annotators `sh/annotate` to tag features if you want.

# Ex:
# ./sh/visualize_features.sh --data_dir /nlp/scr/siddharth/sparse-adaptation/feature_data/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860_20260519_171751_15493160
#
# or, for the l1=0.016 run (is this better? idk): ./sh/visualize_features.sh --data_dir /nlp/scr/siddharth/sparse-adaptation/feature_data/2026.TA.gemma2_2b_full_100k_l1_0p016_gemma2_2b_full_100k_tc8192_decb_l1w0.016_tarbb_sl15516187_20260522_115513_15529706
#
# Run the feature dashboard
uv run python -m analysis.features.visualize.feature_dashboard "$@"
