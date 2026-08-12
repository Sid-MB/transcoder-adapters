<!-- [08-10-26] (session 1fee7826-a308-4d81-b53c-9cf6bf8512ea) -->
# Locating the huge adapter's refusal vs. compliance decision features — with a critical audit

Follow-on to [huge_adapter_refusal_split.md](huge_adapter_refusal_split.md). Goal (meeting notes 08-10-26): find where the refusal-vs-comply decision lives by attributing the huge adapter's jailbreak (compliance) graphs and diffing against matched refusal graphs.

## TL;DR
- The **template-flip** design I first ran (same prompt, chat↔plain flips behavior) is **confounded** and should NOT be read as a clean comply-vs-refuse contrast. See audit below.
- The clean contrast is **within the chat (native) distribution**: prompts the adapter *confidently* jailbreaks vs. *confidently* refuses. A topic-controlled feature diff at the decision token yields a **candidate signature**: ~9 compliance-specific + ~11 refusal-specific adapter features. High-res confirmation running (`comply_vs_refuse_chat_hires`, job 16723369).

## Artifacts
- Flip probe (first-token distributions): `/nlp/scr/siddharth/transcoder-adapters/template_probe/08-11-26_flip_prompts.json`
- Flip hi-res overlays (12288 nodes): `base_adapter_comparisons/flip_huge_{chat,plain}_hires/` — KEEP as the OOD contrast, not the main result.
- Native-chat 2048 graphs used for the diff: `base_adapter_comparisons/overlay_huge_strict_refusal/adapter/`
- Corrected hi-res (native chat, confident comply vs diverse refuse): `base_adapter_comparisons/comply_vs_refuse_chat_hires/` (pending)

## Critical audit — why the template-flip contrast is confounded
The flip prompts (harm_035/177/188) change behavior between the chat and plain templates. Reading the adapter's *first-token distribution* (not just greedy text):

| prompt | chat top tokens | plain top tokens | reading |
|---|---|---|---|
| harm_035 | `I` 0.452 / `##` **0.452** | `I` 0.568 | chat is a **50/50 tie**; "complies under chat" is a coin-flip, not a decision |
| harm_177 | `I` **0.928** (refuse) | ` ` 0.286 / ` I` 0.252 / ` Let` 0.119 | plain argmax is **whitespace**; refusal a close 2nd; high entropy |
| harm_188 | `I` 0.497 (refuse) | ` ` 0.403 / ` I` 0.148 | same whitespace-led, low-confidence plain regime |

Three problems:
1. **The plain "compliances" are a degraded, low-confidence regime** — argmax is a bare space token, adapter entropy 2.2–2.4 vs 0.4–1.0 under chat. Not confident jailbreaks; confused OOD output.
2. **The flip tracks the real instruct model, not adapter pathology.** gemma-2-2b-it degrades identically under plain (argmax-refusal 100%→33%, p(refuse) 0.89→0.35), same as the adapter (1.00→0.33, 0.63→0.32). So "plain weakens refusal" is a property of the *template* (the `User:/Assistant:` cue is weaker than the official `<start_of_turn>` template the model was safety-trained on), shared by any chat-tuned model. Good for **faithfulness** (the adapter isn't uniquely broken) — but it means the flip is entangled with plain being OOD for both models.
3. A chat-vs-plain circuit diff would therefore partly trace "confident refusal vs confused OOD whitespace state," not "refuse-circuit vs comply-circuit."

**Consequence:** keep the flip graphs as an OOD/faithfulness illustration; do the decision-feature analysis on the native chat distribution instead.

## Clean contrast: within-chat, confident comply vs. confident refuse
On the native chat distribution the adapter *confidently* jailbreaks on harm_031 (`##` 0.81), harm_034 (`##` 0.87) and (less strongly) harm_094 (`Hey` 0.38), harm_187 (`Let` 0.39), and confidently refuses the hard-harm prompts (`I` ~0.99). Feature diff at the **decision position** (last context token), adapter-side "cross layer transcoder" nodes, 2048-node graphs:

- **Topic control matters.** harm_031/034 are both disinfo-persuasion, so a 2-prompt comply diff surfaces topic features. Requiring a feature to appear in ≥3 of 4 *topically-diverse* comply graphs (disinfo ×2, drug-persuasion, cyber) and in 0/6 diverse refuse graphs removes that confound.

**Candidate compliance-specific features** (≥3/4 diverse comply, 0/6 refuse): L13 f4567739 and L22 f65671507 (both 4/4), plus L16 f101737963, L16 f111893303, L19 f81790, L19 f103471285, L21 f1457756, L23 f25052557, L24 f12814428.

**Candidate refusal-specific features** (≥5/6 refuse, 0/4 comply; more trustworthy — refuse prompts are content-diverse): L13 f329252, L16 f1023148, L18 f37398257, L19 f79285508, L20 f39859035, L24 f40774940 (all 6/6), plus L13 f59236156, L16 f13605919, L17 f56919097, L23 f6313657, L24 f94167201.

## High-res confirmation (12288 nodes, native chat) — job 16723869
Re-ran the identical topic-controlled diff on the 12288-node graphs (`comply_vs_refuse_chat_hires/adapter/`, ~145–148 adapter features/graph at the decision position). The **resolution-robust survivors** (candidate in BOTH the 2048 and 12288 sets → not truncation artifacts) are the trustworthy core:

- **Refusal-specific (strongest — all 6/6 refuse at both resolutions, 0 comply): L13 f329252, L16 f1023148, L16 f13605919, L19 f79285508.** Plus L13 f59236156, L17 f56919097, L24 f94167201 (survive but ≤5/6).
- **Compliance-specific (standout — 4/4 comply at both resolutions, 0 refuse): L21 f1457756.** Plus L16 f101737963, L16 f111893303, L19 f103471285, L24 f12814428 (survive at 3/4).

Read: the **refusal** signature is solid — four features fire at the decision token of every content-diverse refusal and no jailbreak, at both node budgets. The **compliance** signature is thinner (one rock-solid feature, L21 f1457756; the rest at 3/4), limited by there being only ~2 confident chat jailbreaks in the set.

## Semantic check — the "4 refusal features" are really 1 (see [feature_semantics.md](feature_semantics.md))
Pulling each candidate's stored activation examples + top logits (tool: `analysis/features/inspect_features.py`) sharply revises the above. Of the four "strongest" refusal features, **only L13 f329252 is a genuine refusal detector** (fires on the first token of "I'm sorry, but I cannot…" after harmful requests, cross-lingual — the Anthropic-style refusal feature). The other three passed the 6/6 diff for the wrong reason:
- **L16 f1023148** is a drug-synthesis **harm-topic** feature — its own top examples are meth-synthesis *compliances* as well as refusals, i.e. decision-agnostic. It rode the diff because every refusal prompt is harmful.
- **L16 f13605919** (get-rich-quick how-tos) and **L19 f79285508** (polysemantic NSFW-roleplay/list-enumeration) are incidental/topic.
- Weaker candidate **L24 f94167201** is a *second* genuine refusal detector but Russian-language-specific.
- Compliance **L21 f1457756** is coherent but is an **answer-onset / "begin long-form content"** feature (top logits are BOTH ` I` and `##`), i.e. "the prompt was routed as a writing task" — a plausible *cause* of the jailbreaks (all long-form generation requests) but not a safety-override switch.

Lesson for the diff method: "present in all refusals, no jailbreak" over a harmful-only prompt set selects **harmful-request-content** features, not refusal-**decision** features. Fix (future): require candidate features to be *absent on harmful prompts that were complied with*, and add a max-activation-frequency filter (~2e-3) to drop incidental high-frequency features. Also: `top_logits` are junk for mid-layer (L13–L19) features in this adapter — judge those from examples only.

## Caveats still open (do before claiming these ARE the refusal/comply circuit)
1. **Resolution:** RESOLVED — 12288 rerun confirms the survivors are not truncation artifacts.
2. **Small N** (4 comply / 6 refuse). The comply side especially is confidence-limited (only ~2 truly confident chat jailbreaks exist in this set).
3. **Set-membership ≠ causal.** Correlational. Causal ablation IN PROGRESS — narrowed by the semantic check to target **L13 f329252** first (predict: ablating it flips harm_000 refusal→comply; ablating the topic/incidental three does little; random-feature control does nothing).
4. **Feature semantics:** RESOLVED — see [feature_semantics.md](feature_semantics.md); only L13 f329252 (+ ru-specific L24 f94167201) are real refusal detectors.

## Next
Causal ablation (narrowed to L13 f329252, with L24 as a ru-control and L21 watched for answer-onset degradation vs a true flip) → if L13 carries the effect, that + its semantics = the adapter's refusal decision feature. Then optionally re-run the diff with the tightened criterion (absent-on-complied-harm + frequency filter) to clean the compliance side.

