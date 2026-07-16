<!-- Claude Code session "explore: weird circuit tracer issues". 2026-07-15. -->
# Hybrid graph showed BASE completions instead of instruct — root cause & fix

**The setup.** We build "hybrid" circuit-tracer graphs to interpret the base↔instruct gap on `google/gemma-2-2b`. Two kinds of module are involved, and they do **opposite** jobs:
- **GemmaScope transcoders** (base + our fine-tuned L0/24/25): *replace* each MLP with a sparse reconstruction that **imitates the base model's MLP**. Interpretability. They keep the model **base-like**.
- **Trained transcoder *adapters*** (`…sl14793860`, `…sl15516187`): *add* a branch on top of each MLP (`MLP(x) = base_MLP(x) + adapter(x)`) trained to **bridge base → instruct behavior**. They make the model **instruct-like**.

**The problem (symptom).** The new fine-tuned hybrid graph (`base_adapter_comparisons/hybrid_finetuned_ftL0-24-25`, built with `run_combined_attribution`) showed **base-model completions**, not instruct ones. On the prompt `Answer with one word. What city is the Eiffel Tower in? … model\nParis`, its target logits were `'\n'`(0.62), `','`, `'.'`, `'\n\n'`, `' is'` — the model wanting to **continue the document with a new question**, exactly like base gemma, instead of stopping (`<end_of_turn>`) as the instruct model does.

**The investigation (what ruled out what).** Running each model directly on that prompt, top token after "Paris":

| model | top token(s) | verdict |
|---|---|---|
| base `gemma-2-2b` | `'\n'`, `'.'`, `','`, `'\n\n'`, `' is'` | continue document |
| instruct `gemma-2-2b-it` | `' '`, `'.'`, `'\n'`, `'  '`, `'<end_of_turn>'` | stop / answer |
| adapter model, **weak** `sl14793860` (real base MLPs + adapter) | `' '`, `'.'`, `'  '`, `'\n'`, `'\n\n'`, `'<end_of_turn>'` | **instruct-like** ✅ |
| adapter model, **strong** `sl15516187` | `' '`, `'\n'`, `'\n\n'`, `'.'`, `'  '`, `'<end_of_turn>'` | **instruct-like** ✅ |
| **our `run_combined_attribution` graph** | `'\n'`(0.62), `','`, `'.'`, `'\n\n'`, `' is'` | **base — matches base gemma token-for-token** ❌ |

So it was **not** the adapter (both adapters emulate instruct fine as adapter models, even the weak one), and **not** feature-example alignment. The combined graph's logits *equal the base model's* exactly.

**Root cause — the wrong attribution tool.** `run_combined_attribution` builds a *single* circuit-tracer `ReplacementModel` on the **base backbone** (`--base_model google/gemma-2-2b`), then replaces each MLP with `T_base_gemmascope(x) + T_adapter(x)`. Because the GemmaScope transcoders reconstruct the **base** MLP and the reconstruction error is added back to keep the forward faithful, the net forward **collapses to the base model** — so the completions are base's. The `T_base + T_adapter` split is only a *decomposition for attribution*; it does not run the adapter/instruct model. Hence base completions regardless of which adapter you use or whether the GemmaScope layers are fine-tuned.

Contrast the tools:

| graph | tool | how the model runs | completions |
|---|---|---|---|
| `attribution_graphs/…sl15516187…` | `run_circuit_tracer_pipeline` | the adapter model itself | instruct ✅ |
| `calibrated_…/overlay_compact…` | **`run_base_adapter_comparison`** | base run **+** adapter run, overlaid | instruct ✅ (adapter side answers, e.g. "Rome" p=0.91) |
| `hybrid_finetuned_ftL0-24-25` | **`run_combined_attribution`** | one *base-backbone* combined model | **base ❌** |

**The fix.** Rebuild the fine-tuned hybrid with **`run_base_adapter_comparison`** (the base-vs-adapter *overlay*), which runs the **real adapter model** (instruct completions) and overlays it with the base GemmaScope run. This preserves everything we want: base side = **fine-tuned GemmaScope** transcoders (via the `--finetuned_transcoder_dir` passthrough), adapter side = the actual adapter model → **instruct completions**, both shown together. We also upgraded the feature examples from the 20k-sample collection to the project's **current-best `ms100000_dtk20`** (full 100k+100k corpus, ~20 examples/feature) for **both** base and adapter sides.

**One-sentence version.** The combined graph showed base completions because `run_combined_attribution` runs a *base-backbone* model whose forward collapses to base; the fix is to build the hybrid with `run_base_adapter_comparison` (which runs the real instruct-bridging adapter model and overlays the base run), keeping the fine-tuned GemmaScope base transcoders and using the 100k-corpus (`ms100000_dtk20`) feature examples.

## The corrected graph

- **Command / build:** `run_base_adapter_comparison` overlay, job **16194543** (COMPLETED, 3m39s on jagupard35), with `--finetuned_transcoder_dir siddharthmb/2026.TA.gemma2_2b_gemmascope_transcoders_instruct_ft_L0-24-25 --finetuned_layers 0 24 25`, `--base_feature_data_path …ms100000…h12ad59325ffd`, `--feature_data_path …lr_he94e9602bafa`, prompts `interesting_small`, 2048 nodes / 10 logits. (Two arg-parse crashes en route — `is_hf_feature_ref` and `normalize_hf_feature_ref` were used in `run_base_adapter_comparison` but not imported from `run_circuit_tracer_pipeline`; fixed and committed.)
- **Output:** `$LARGE_ARTIFACTS_DIR/transcoder-adapters/base_adapter_comparisons/hybrid_ft_overlay_ms100k/` (sides: `base/`, `adapter/`, `overlay/`, `overlay_compact/`).
- **Serve:**
  ```bash
  uv run --extra viz python -m analysis.attribution.serve_comparison_graphs \
    --graph_file_dir $LARGE_ARTIFACTS_DIR/transcoder-adapters/base_adapter_comparisons/hybrid_ft_overlay_ms100k/overlay --port 8046
  ```
- **RESULT — fixed. ✅** On the `Answer with one word. What city is the Eiffel Tower in? … model\nParis` prompt, the two sides now diverge exactly as they should:

  | side | top logits after "Paris" | reads as |
  |---|---|---|
  | **base** (fine-tuned GemmaScope) | `'\n'` (p=0.617), `','`, `'.'`, `'\n\n'`, `' is'` | continue document — **base-like** ✅ |
  | **adapter** (real adapter model) | `' '` (p=0.902), `'.'` | stop / answer — **instruct-like** ✅ |

  The adapter side matches the instruct model (`gemma-2-2b-it` top token = `' '`), where the old `run_combined_attribution` graph had collapsed to base's `'\n'`(0.62). Base side stays base-like by construction (fine-tuned GemmaScope reconstructs the base MLP). Both feature-example sets are the 100k-corpus `ms100000_dtk20` collections.

## Takeaways / gotchas
- To trace **instruct** behavior, use `run_base_adapter_comparison` (overlay) or the adapter-model pipeline — **not** `run_combined_attribution`, which is a *base-model decomposition* (base completions by construction).
- The fine-tuned GemmaScope transcoders reconstruct the **base** MLP; they improve base-side interpretability/fidelity but carry **no** instruction-following — that lives entirely in the adapter.
- Related: [`my_notes/07-09-26/README.md`](07-09-26/README.md) (the fine-tune + how to apply it).
