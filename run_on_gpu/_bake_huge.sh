#!/bin/bash
cd /juice2/u/siddharth/transcoder-adapters
PYTHONPATH=/juice2/u/siddharth/transcoder-adapters uv run --extra viz python -m analysis.attribution.bake_overlay_continuations   --overlay_dir /nlp/scr/siddharth/transcoder-adapters/_overlay_huge_conts/overlay --compact_dir /nlp/scr/siddharth/transcoder-adapters/_overlay_huge_conts/overlay_compact   --base_model google/gemma-2-2b   --adapter_checkpoint siddharthmb/2026.TA.gemma2_2b_huge_tc16384_decb_l1w0.0003_norm_sch_tarbb_lb2.0_ln1.0_dr500000_lr2e-04_bs8_sl   --max_new_tokens 200
