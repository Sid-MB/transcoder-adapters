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

# [Some runs moved here](./archive/07-29-26 old.md)

# ── #6 · FIXED huge-adapter overlay (supersedes #5) — instruct completions ───────────────────
# #5 above was built with run_combined_attribution, which runs ONE model on the base backbone: its
# forward collapses to base, so its completion/logit nodes are the BASE model's (e.g. "Answer"/"\n" —
# base echoing/continuing the prompt), NOT the instruct-bridging huge adapter's. This #6 is the
# corrected rebuild with run_base_adapter_comparison (runs the REAL adapter model + overlays a
# separate base run), so the adapter side shows instruction-tuned completions: on the one-word-capital
# prompt, base→"Answer" p=0.55 vs adapter→"Paris" p=0.99. Same 40 comprehensive prompts, huge
# 16384-feature adapter, ms100000 base features. Rebuilt on a RunPod B200 (run_base_adapter_comparison).
#   graphs on HF:       https://huggingface.co/datasets/siddharthmb/2026.TA.overlay_huge_graphs
#   huge adapter model: https://huggingface.co/siddharthmb/2026.TA.gemma2_2b_huge_tc16384_decb_l1w0.0003_norm_sch_tarbb_lb2.0_ln1.0_dr500000_lr2e-04_bs8_sl
#   adapter training wandb: https://wandb.ai/siddharth-stanford/sparse-adaptation/runs/c7vfz8o0
#   the fix write-up:   my_notes/07-09-26/07-15-26 hybrid graph instruct-completions bug.md
# Serve straight from HF (downloaded + cached on first run; no local copy needed):
uv run --extra viz python -m analysis.attribution.serve_comparison_graphs --graph_file_dir siddharthmb/2026.TA.overlay_huge_graphs:overlay --port 8049
# compact set:  ...:overlay_compact

# NOTE: BASE MODEL IS GIBBERISH HERE DUE TO SPECIAL TOKENS
# Refusals on the smaller adapter
uv run --extra viz python -m analysis.attribution.serve_comparison_graphs \
  --graph_file_dir /nlp/scr/siddharth/transcoder-adapters/base_adapter_comparisons/tc8192_strict_refusal/overlay --port 8050

# Refusals on larger correct one
uv run --extra viz python -m analysis.attribution.serve_comparison_graphs \
  --graph_file_dir /nlp/scr/siddharth/transcoder-adapters/base_adapter_comparisons/overlay_huge_strict_refusal/overlay --port 8051

# ── #7 · FIRST-TEN-REFUSAL-TOKENS overlays (token-selection-for-tracing study) ────────────────
# Instead of tracing only position 0 ("I"), this builds a base-vs-adapter overlay at EACH of the
# first 10 positions of the instruct refusal (I / cannot / and / will / not / provide / ... ) for 3
# strict-flip prompts (harm_125 meth, harm_139 DDoS, harm_116 pipe bomb) = 30 graphs. Same tc8192
# config as #2. Use the graph dropdown to step "I -> cannot -> provide -> ..." and watch where the
# adapter's refusal/harmful features come in. Companion prefill-flip eval (does transplanting the
# instruct refusal opening make BASE refuse? no) + full write-up:
#   my_notes/08-10-26/refusal_token_tracing/refusal_token_tracing.pdf
uv run --extra viz python -m analysis.attribution.serve_comparison_graphs \
  --graph_file_dir /nlp/scr/siddharth/transcoder-adapters/base_adapter_comparisons/tc8192_refusal_first10/overlay --port 8052

# ── #7 · Refusal-token circuits (08-10-26) — where does refusal commit, "I" or "cannot"? ──────
# 3 prompts where the huge adapter AND instruct both refuse (base does not), each traced at TWO
# positions: __tok_I (first response token "I") and __tok_cannot (prefix "I", target " cannot").
# Built with run_base_adapter_comparison (job 16717643). Each graph shows a "Refusal ladder" panel
# with every model's judged verdict for that prompt -- harm_006 is the sharpest case: with the
# transcoder zeroed the model WRITES the phishing text (COMPLIANCE), with it the model REFUSES.
#   writeup + bar chart: my_notes/08-10-26/refusal_ladder.md
# uv run --extra viz python -m analysis.attribution.serve_comparison_graphs --graph_file_dir /nlp/scr/siddharth/transcoder-adapters/base_adapter_comparisons/refusal_tokens/overlay --port 8052
