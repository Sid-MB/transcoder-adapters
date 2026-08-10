#!/usr/bin/env bash
# [08-10-26] (session 1fee7826-a308-4d81-b53c-9cf6bf8512ea)
#
# Serve the BASE-vs-ADAPTER circuit-tracing comparison overlays REBUILT with the neutral-plaintext
# prompt template (`--prompt_format plain`). Successor to `visualize_base_adapter 07-29-2026.sh`:
# same graphs (base GemmaScope ⬢ vs sparse transcoder adapter ● + reconstruction-error ▲ in one
# full-replacement graph, MLP(x) = T_base(x) + T_adapter(x) + Err), but the prompts are rendered as
# `User: ...\nAssistant:` instead of the gemma chat template `<start_of_turn>user ... <start_of_turn>model`.
#
# WHY THE REBUILD ─────────────────────────────────────────────────────────────────────────────────
# The 07-29 refusal overlays (ports 8050/8051 in the old script) render every prompt through the gemma
# CHAT template. That is in-distribution for the adapter (trained on lmsys via apply_chat_template) but
# OUT-of-distribution for base gemma-2-2b, which never saw those markers in that role: the base side
# degenerates into echoing/looping and its top logit is a prompt-echo token, so the base graph is a
# circuit for echoing, not refusal. (This is the "BASE MODEL IS GIBBERISH DUE TO SPECIAL TOKENS" note
# on the old script.) The `plain` template is legible to BOTH models, so the base side becomes a real
# "does base refuse?" picture aligned to the adapter's on identical tokens.
#
# Measured by the template-sensitivity probe (analysis/evals/template_sensitivity_probe.py; results
# /nlp/scr/siddharth/transcoder-adapters/template_probe/08-10-26_strict_refusal.json), 12 prompts:
#   cell               p(target "I")  argmax%  entropy  loop%
#   base   · chat          0.020         8%      5.84    100%   <- pathological: target ranked as low as 93
#   base   · plain         0.045         8%      4.03     83%   <- coherent (rank→≤4), still doesn't refuse
#   adapter· chat          0.931        100%     0.28      0%
#   adapter· plain         0.723        100%     1.23     17%   <- refusal SURVIVES losing its training template
#   instruct·chat          0.988        100%     0.06      0%
#   instruct·plain         0.733         92%     1.07      0%
# Takeaways: (1) plain fixes base's confusion (entropy/target-rank) without making it falsely refuse,
# so "base doesn't refuse" is now a clean finding; (2) the adapter still refuses at 100% argmax under
# plain, so `plain` is safe to adopt; (3) base still loops under plain -- that is a greedy-decode
# artifact (base has no EOS habit), ORTHOGONAL to the template, so the completion panel below the graph
# may still look repetitive even though the graphs are now sound.
#
# NOTE ON BASE ATTRIBUTION TARGETS: these two runs were built with `--prompt_format plain` but the
# OLD `--base_attribution_targets top_logits` (they were already running when the union-target flag
# landed). A strictly-better base graph attributes toward the base's salient logits UNION the adapter's
# top token (`--base_attribution_targets union_adapter_top`, now the default); that is an additive
# base-only rebuild (clear base/ + overlay/, keep the adapter graphs) and can be done later without
# touching the adapter side. See run_base_adapter_comparison.py --base_attribution_targets help.
#
# ★ STARRED PROMPTS: in the huge-adapter refusal set the dropdown stars (★) the 5 strict-refusal prompts
# the huge adapter COMPLIES with instead of refusing (harm_031/034/035 = disinfo "persuasive article"
# framings, harm_094 = huff-paint persuasion, harm_187 = DNS-amplification how-to). Finding + transcripts:
# my_notes/08-10-26/huge_adapter_refusal_split.md. Stars are applied post-build with
# analysis.attribution.star_graphs_in_dropdown (see POST-BUILD below); re-run after any rebuild.
#
# Usage: run from the repo root. Each command serves on its own port and BLOCKS -- run the one you want
# (or several in separate terminals; ports differ so they don't collide). Remote? forward the port:
# `ssh -L 8052:localhost:8052 <host>`, then open http://localhost:<port>.
#
# Artifacts (shared with the 07-29 runs; only the prompt template changed):
#   huge adapter model:  https://huggingface.co/siddharthmb/2026.TA.gemma2_2b_huge_tc16384_decb_l1w0.0003_norm_sch_tarbb_lb2.0_ln1.0_dr500000_lr2e-04_bs8_sl
#   deployed tc8192 model: https://huggingface.co/siddharthmb/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860
#   base features (ms100000): https://huggingface.co/siddharthmb/2026.TA.features_gemma-2-2b_gemmascope_width_16k_average_l0_76_ms100000_ml1024_tk1_hf83e96d574d2
#   prompt set: experiments/interesting_queries/results/strict_compliance_refusal/selected_prompts

set -euo pipefail

# ⚠ STATUS: the two overlays below are being rebuilt on sphinx (jobs 16717287 huge, 16717292 tc8192).
#   Watch:  squeue -u $USER | grep refus_.*_plain
#   They serve only once each job's overlay/ dir is populated.

# ── #1 · Refusals on the HUGE (tc16384) adapter — plain template ──────────────────────────────────
# 63 strict-refusal prompts. ★ marks the 5 the adapter complies with (see header). Adapter refuses the
# other 58 ("I cannot..."); base side (plain) is now coherent rather than gibberish, and mostly does
# NOT refuse -- the intended base-vs-adapter contrast.
uv run --extra viz python -m analysis.attribution.serve_comparison_graphs \
  --graph_file_dir /nlp/scr/siddharth/transcoder-adapters/base_adapter_comparisons/overlay_huge_strict_refusal_plain/overlay --port 8053

# ── #2 · Refusals on the deployed tc8192 adapter — plain template ─────────────────────────────────
uv run --extra viz python -m analysis.attribution.serve_comparison_graphs \
  --graph_file_dir /nlp/scr/siddharth/transcoder-adapters/base_adapter_comparisons/tc8192_strict_refusal_plain/overlay --port 8052

# ── POST-BUILD (run once each job finishes) ───────────────────────────────────────────────────────
# Re-apply the ★ compliance markers to the rebuilt huge-adapter dropdown (idempotent):
#   uv run python -m analysis.attribution.star_graphs_in_dropdown \
#     --graph_dir /nlp/scr/siddharth/transcoder-adapters/base_adapter_comparisons/overlay_huge_strict_refusal_plain/overlay \
#     --stems harm_031 harm_034 harm_035 harm_094 harm_187
#   (and the same for .../overlay_huge_strict_refusal_plain/overlay_compact)
