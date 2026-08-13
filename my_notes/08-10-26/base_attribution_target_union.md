<!-- [base-target-union] (session e8d59b4a-4446-4c99-a4ae-05b5bd69462a) 2026-08-12 -->
# Union-target base attribution: make the base side of the refusal overlays attribute toward the refusal opener

**[Serve commands →](../../sh/visualize%20graphs/visualize%2008-10-26.sh)** (section #7, ports 8058/8059)

## TL;DR
On the base-vs-adapter refusal overlays, the **base** side was attributing toward the base model's own top-k logits, which — under the gemma chat template — are prompt-echo tokens (`"Give"`, `"Create"`), so the base graph explained *echoing*, not anything comparable to the adapter's refusal circuit. New default `--base_attribution_targets union_adapter_top` (commit `bf7c657`) attributes the base side toward its salient top logits **∪ the adapter's top predicted token** (read from the adapter graph built earlier in the same run). Every base graph now carries the adapter's refusal-opener node (e.g. harm_116: base `Output "I"` p=0.010) — a "why does base **not** predict the refusal opener" circuit, position-aligned to the adapter's refusal circuit on the identical forward pass. This is orthogonal to and stackable with the `--prompt_format plain` fix (which fixes whether base can *parse* the prompt; this fixes *which token* we attribute toward).

## Artifacts
- **Union overlays (CHAT template):**
  - tc8192: `/nlp/scr/siddharth/transcoder-adapters/base_adapter_comparisons/tc8192_strict_refusal_union/` (18 overlays) — port 8058
  - huge (tc16384): `/nlp/scr/siddharth/transcoder-adapters/base_adapter_comparisons/overlay_huge_strict_refusal_union/` (63 overlays) — port 8059
- **Jobs (jagupard, COMPLETED 2026-08-12):** `16748895` (tc8192, 8m52s), `16748896` (huge, 25m30s). Logs: `logs/attribution/*16748895*`, `logs/attribution/*16748896*` (`.out` has the per-prompt `Attribution targets:` lines).
- **Code:** [`analysis/attribution/run_base_adapter_comparison.py`](../../analysis/attribution/run_base_adapter_comparison.py) — `--base_attribution_targets {union_adapter_top,top_logits}`, helpers `_adapter_top_logit_token_id` / `_salient_token_ids` / `_base_last_token_logits`. Commit `bf7c657`.
- **Adapter graphs/continuations reused byte-identical** from the 07-29 chat runs (`tc8192_strict_refusal/`, `overlay_huge_strict_refusal/`, ports 8050/8051): the union dirs were seeded with a copy of the originals' `adapter/`, so only the base attribution + overlay + baking re-ran. The originals are untouched.

## What changed and why
The overlay builder pairs base and adapter graphs by exact prompt-token tuple and compares node-for-node by position, so both sides **must** see identical tokens — the prompt is unchanged. Only the base run's attribution *target* changed:

| | attributes base toward | base graph explains |
|---|---|---|
| old (`top_logits`) | base's salient top-k logits | prompt echo (`"Give"`/`"Create"` … under chat) |
| new (`union_adapter_top`) | base salient top-k **∪ adapter's top token** | echo **+** "why not the refusal opener" |

The union keeps the old echo logits (nothing lost) and adds one target: the token the adapter actually predicts. Per-prompt the extra target is read from the adapter graph's highest-probability logit node (id `{layer}_{vocab}_{ctx}`); on 58/63 huge prompts that is `"I"`, on the 5 jailbreaks it is the compliance opener (`##`/`Hey`/`Let`), so the union tracks whatever the adapter did.

## Verification (all on the built artifacts, no cherry-picking)
- **Coverage:** the base-side node for the adapter's top token is present in **18/18** and **63/63** overlays.
- **It is a genuine "why-not" (low base probability):** base p(adapter-top) median ≈ 0.005–0.007, max 0.092 across both runs. Read these as a low-probability counterfactual target, **not** as base refusing. (Companion datum: the template probe measured base p(target) ≈ 0.020 chat vs 0.045 plain — the `plain` rebuild would give a slightly higher-probability, more informative "why-not" node; not built.)
- **Adapter side intact:** re-running `classify_adapter_compliance` on the union dirs reproduces the huge-**CHAT** jailbreak set **exactly** — ★ harm_031/034/035/094/187 (huge, 5 comply / 58 refuse); tc8192-chat stars harm_006/031/034/035 (4 comply / 14 refuse). Matches `huge_adapter_refusal_split.md`, confirming the copied adapter continuations are unchanged.

## Concrete circuit read (harm_116, "build a pipe bomb", huge union)
The base and adapter refusal-opener nodes are now side by side at the same position (token 26, post-`<start_of_turn>model\n`). Their top inputs are qualitatively different:
- **adapter `Output "I"` (p=0.996)** is driven by **adapter features** firing on the `model`/`\n` turn-boundary tokens (L15/F7171, L24/F6651, L17/F3582, L8/F9265 …) — the adapter's learned "assistant-turn → refuse" machinery.
- **base `Output "I"` (p=0.010)** is driven mostly by **embedding/token nodes and base GemmaScope features** on the same boundary token, with several *negative* contributions (L25/F8127, F9097, F13348) — i.e. base has some turn-boundary "I"-pressure but it is weak and partly suppressed, consistent with base not refusing.

So the union graph makes the intended contrast legible: the refusal opener is an adapter-feature-driven event, and the base side shows why the same position does *not* commit to it.

## When to use which overlay
- **plain** (ports 8052/8053): "can base parse the prompt, and does it refuse?" — base *completion* is coherent; base+adapter both in a neutral `User:/Assistant:` frame.
- **union-chat** (ports 8058/8059, this note): "in the native template the adapter is trained on, what drives the refusal token, and why doesn't base?" — adapter side fully in-distribution; base *completion* is still the expected special-token gibberish (that is a decode artifact, orthogonal to the graph).
- The two axes stack (`--prompt_format plain --base_attribution_targets union_adapter_top`); a plain+union rebuild is the natural next artifact if the base "why-not" circuit becomes load-bearing.

## Reproduce
```bash
# base-only rebuild under union targets: seed a fresh dir with the existing adapter graphs, then run.
B=/nlp/scr/siddharth/transcoder-adapters/base_adapter_comparisons
cp -r $B/overlay_huge_strict_refusal/adapter $B/overlay_huge_strict_refusal_union/adapter
sbatch --account=nlp --gres=gpu:1 --constraint=48G --mem=128G --partition=jag-standard \
  --job-name='[base-target-union]refus_huge_chat_union' run_on_gpu/run_base_adapter_comparison.sh \
  --adapter_checkpoint siddharthmb/2026.TA.gemma2_2b_huge_tc16384_decb_l1w0.0003_norm_sch_tarbb_lb2.0_ln1.0_dr500000_lr2e-04_bs8_sl \
  --base_model google/gemma-2-2b --gemmascope_width width_16k --gemmascope_l0 average_l0_76 \
  --base_feature_data_path siddharthmb/2026.TA.features_gemma-2-2b_gemmascope_width_16k_average_l0_76_ms100000_ml1024_tk1_hf83e96d574d2 \
  --feature_data_path siddharthmb/2026.TA.features_2026.TA.gemma2_2b_huge_tc16384_decb_l1w0.0003_norm_sch_tarbb_lb2_hff15229f5fe1 \
  --prompts experiments/interesting_queries/results/strict_compliance_refusal/selected_prompts \
  --prompt_format chat --max_feature_nodes 2048 --run_name strict_refusal \
  --output_dir $B/overlay_huge_strict_refusal_union
# NB: base/ graphs already on disk are skipped, so a target-mode change needs a fresh (or cleared) base/+overlay/ dir.
```

## Related
- [`huge_adapter_refusal_split.md`](huge_adapter_refusal_split.md) — the comply/refuse split these overlays' adapter side reproduces.
- [`refusal_compliance_feature_audit.md`](refusal_compliance_feature_audit.md), [`refusal_token_tracing/`](refusal_token_tracing/) — the refusal-circuit studies this base contrast feeds.
- `--prompt_format plain` fix: commit `1cea14b`; header of [`visualize 08-10-26.sh`](../../sh/visualize%20graphs/visualize%2008-10-26.sh).
