#!/usr/bin/env bash
# [train: gemma 2 very large runpod] (session 74ddb13a-f03d-4f55-bd14-df05cd2353c6)
#
# Serve the BASE-vs-ADAPTER circuit-tracing comparison overlays — the base GemmaScope
# transcoders (hexagon ⬢) vs a sparse transcoder adapter (circle ●) plus real
# reconstruction-error nodes (triangle ▲), in one full-replacement graph
# (MLP(x) = T_base(x) + T_adapter(x) + Err). Click a node for per-feature proportions
# ("fires on X% of tokens" / top-token specificity) + activation examples on our chat+web data.
#
# This is the counterpart to `visualize_hybrid.sh` (which serves the *fine-tuned-base* /
# "hybrid_ft" overlays). Full write-up + artifact links: my_notes/06-18-26 to 06-25-26 products.md
# (top "Quick reference" table = these same overlays; §7 = the huge-adapter one).
#
# Overlays #1–#4 are the deployed tc8192 adapter; #5 is the new 16384-feature huge adapter.
# All use the current-best full-corpus `ms100000` / `dtk20` feature examples (~20 examples per
# dataset on node-click). Graph dirs live on shared scratch /nlp/scr/siddharth (jagupard-visible);
# the underlying feature collections are on HF (linked per-overlay below).
#
# Usage: run from the repo root. Each command serves on its own port and BLOCKS, so run the one
# you want (or several in separate terminals — the ports differ so they don't collide). Remote?
# forward the port: `ssh -L 8044:localhost:8044 <host>`, then open http://localhost:<port>.
#
# Backing feature collections on HF (baked into the graphs above; shared by #1–#4):
#   base   (dtk20): https://huggingface.co/siddharthmb/2026.TA.features_gemma-2-2b_gemmascope_width_16k_average_l0_76_ms100000_ml1024_tk2_h12ad59325ffd
#   adapter(dtk20): https://huggingface.co/siddharthmb/2026.TA.features_2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr_he94e9602bafa
#   deployed adapter model: https://huggingface.co/siddharthmb/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860

set -euo pipefail

# ── #1 · Headline / default (full-corpus, standard budget 4096 / 5 logits / 32 err) ──────────
# 12 `interesting_small` prompts. Cleanest + fastest to load — the default for browsing or a demo.
uv run --extra viz python -m analysis.attribution.serve_comparison_graphs --graph_file_dir /nlp/scr/siddharth/transcoder-adapters/graph_combined_dtk20 --port 8044

# ── #2 · Agree/diverge experiment (28 purpose-built prompts) ─────────────────────────────────
# 14 agree (factual, base does the work) + 14 diverge (behavioral), each dropdown entry labeled
# [agree]/[diverge]. Use to explore where the adapter fires on content tokens. Different prompt
# set — NOT comparable to the others.
# uv run --extra viz python -m analysis.attribution.serve_comparison_graphs --graph_file_dir /nlp/scr/siddharth/transcoder-adapters/agree_diverge_dtk20/all --port 8045

# ── #3 · Denser (attr_big, 2× budget 8192 / 10 / 64) — THE EVERYDAY BEST FOR ANALYZING A CIRCUIT ─
# Same 12 prompts as #1, deeper circuits, still readable.
# uv run --extra viz python -m analysis.attribution.serve_comparison_graphs --graph_file_dir /nlp/scr/siddharth/transcoder-adapters/attribution_big_dtk20 --port 8046

# ── #4 · Densest (attr_2x, 4× budget 16384 / 20 / 128) ──────────────────────────────────────
# Same 12 prompts, most complete but heavy/cluttered — for deep single-prompt dives; thin it with
# the pruning slider.
# uv run --extra viz python -m analysis.attribution.serve_comparison_graphs --graph_file_dir /nlp/scr/siddharth/transcoder-adapters/attribution_2x_dtk20 --port 8047

# ── #5 · NEW: base vs the HUGE adapter (§7) — 40 comprehensive prompts ───────────────────────
# The freshly-trained 16384-feature `gemma2_2b_huge` adapter (2× capacity; B200 warm-start,
# final eval KL 0.142 / 86.35% top-1). The only overlay against the *huge* adapter (all others
# are the deployed tc8192). Node budgets vary per prompt (32 at 32768, 1 at 12288, 7 at 8192 —
# the assistant-prefix prompts' mid-response tracing needs ~79 GB regardless of node count);
# each graph's banner shows its actual composition.
#   huge adapter model:      https://huggingface.co/siddharthmb/2026.TA.gemma2_2b_huge_tc16384_decb_l1w0.0003_norm_sch_tarbb_lb2.0_ln1.0_dr500000_lr2e-04_bs8_sl
#   huge adapter collection: https://huggingface.co/siddharthmb/2026.TA.features_2026.TA.gemma2_2b_huge_tc16384_decb_l1w0.0003_norm_sch_tarbb_lb2_hff15229f5fe1
#   base collection (dtk5):  https://huggingface.co/siddharthmb/2026.TA.features_gemma-2-2b_gemmascope_width_16k_average_l0_76_ms100000_ml1024_tk1_hf83e96d574d2
# uv run --extra viz python -m analysis.attribution.serve_comparison_graphs --graph_file_dir /nlp/scr/siddharth/transcoder-adapters/overlay_huge/graph --port 8048
