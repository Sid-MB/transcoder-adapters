#!/usr/bin/env sh

# First, if needed, run `./sh/slurm_batch_collect_features.sh` to create the feature data.

# Run the feature dashboard
uv run python -m analysis.features.visualize.feature_dashboard --data_dir /nlp/scr/siddharth/sparse-adaptation/feature_data/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl147_20260416_151353_15175430
