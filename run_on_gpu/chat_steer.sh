#!/usr/bin/env bash

# Ex:
# ./run_on_gpu/chat_steer.sh siddharthmb/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr10000_lr8e-04_bs4_sl14754432
#
# Or, with steering
# ./run_on_gpu/chat_steer.sh siddharthmb/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860 --steer 11551209:4

uv run python -m analysis.simple_load.simple_load "$@"

