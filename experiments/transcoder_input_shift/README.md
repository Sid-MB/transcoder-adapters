<!-- Created by Claude Code session "implement: new 07/02 gemmascope transcoder experiments". 2026-07-09. -->
# GemmaScope transcoder input-distribution shift (Experiment 1)

**[Code → [`analysis/features/transcoder_input_shift.py`](../../analysis/features/transcoder_input_shift.py) · [`analysis/features/analyze_fire_freq_drift.py`](../../analysis/features/analyze_fire_freq_drift.py) · runners [`sh/slurm_batch_transcoder_input_shift.sh`](../../sh/slurm_batch_transcoder_input_shift.sh), [`run_on_gpu/run_transcoder_input_shift.sh`](../../run_on_gpu/run_transcoder_input_shift.sh)]**
**[Plan → `~/.claude/plans/cached-doodling-horizon.md`]**

## Question

Pretrained GemmaScope transcoders (`google/gemma-scope-2b-pt-transcoders`, `width_16k`) were trained to replace each **base-model MLP inside the full base model** — on the *base model's* per-layer input distribution. We want to plug them into circuit-tracer to study `google/gemma-2-2b` (base) vs `google/gemma-2-2b-it` (instruct). Switching the source model shifts the distribution of inputs to each layer.

**For each layer, is the transcoder's sparsity (L0) and reconstruction error the same when its input hidden states come from the base model vs. the instruct model?** If ~same, the transcoders transfer; if materially worse on instruct inputs, the shift warrants re-fine-tuning the transcoders (or a pivot to adapter-only / ReLP).

## Method

For each source model and layer L, over ~1M tokens: take `x = blocks.L.ln2.hook_normalized` (the transcoder's input), compute `feats = transcoder.encode(x)` (→ **L0**) and `recon = transcoder.decode(feats)`. The target is **always the base MLP**: `MLP_base(x)`, obtained via a *patched* base-model forward (overwrite `ln2.hook_normalized` with `x`, read `hook_mlp_out`) — exact because the gemma-2 MLP sub-block is position-wise. Models are loaded as TransformerLens `HookedTransformer`s with the same processing (`fold_ln=False, center_writing_weights=False, center_unembed=False`) that circuit-tracer's `ReplacementModel` uses, so `ln2.hook_normalized` matches GemmaScope's convention. Metrics: **FVU** (fraction of variance unexplained), MSE, cosine, and per-feature firing frequency. BOS (position 0) and padding excluded.

## Figure

Overview (6 panels): `.claude/products/transcoder_input_shift/transcoder_input_shift_overview.png` (gitignored; regenerate with [`analysis/features/plot_input_shift.py`](../../analysis/features/plot_input_shift.py)). Panels: FVU vs layer (base vs instruct, 26 layers); FVU increase % per layer; L0 vs layer; per-feature firing-rate correlation; chat-vs-web FVU gap; base-input FVU baseline.

## Results — reconstruction error (FVU) and sparsity (L0), base vs instruct inputs

Chat data ([lmsys splits](https://huggingface.co/datasets/siddharthmb/2026.transcoder-adapters.lmsys-chat-1m-splits), rendered with the -it chat template), 1M tokens/layer:

| layer | base L0 | instruct L0 | base FVU | instruct FVU | ΔFVU |
|---|---|---|---|---|---|
| 0 | 80.3 | 73.2 | 0.082 | 0.109 | **+33%** |
| 6 | 97.5 | 88.4 | 0.352 | 0.392 | +11% |
| 12 | 59.0 | 63.4 | 0.499 | 0.537 | +8% |
| 18 | 53.6 | 59.2 | 0.340 | 0.375 | +10% |
| 25 | 64.9 | 59.6 | 0.152 | 0.310 | **+104%** |

Web data ([fineweb-1m-sample](https://huggingface.co/datasets/science-of-finetuning/fineweb-1m-sample)), 1M tokens/layer — tests whether the shift is chat-specific:

| layer | base FVU | instruct FVU | ΔFVU |
|---|---|---|---|
| 0 | 0.075 | 0.098 | +31% |
| 6 | 0.342 | 0.388 | +13% |
| 12 | 0.486 | 0.550 | +13% |
| 18 | 0.320 | 0.348 | +9% |
| 25 | 0.177 | 0.406 | **+130%** |

Full 26-layer depth profile (chat, 500k tokens/layer, job 16113808) confirms and localizes the pattern: instruct FVU ≥ base FVU at **every** layer. The degradation is concentrated at the **endpoints** — L0 **+33%**, L24 **+32%**, L25 **+104%** — while middle layers sit at ~+5–13% (a couple mid-layers are within ±1%, i.e. noise). Base FVU itself peaks mid-stack (L12 = 0.50) and is lowest at the ends (L0 = 0.08, L25 = 0.15). So the input shift hurts reconstruction most exactly where the transcoders are otherwise most accurate (first/last layers).

### Per-feature firing-frequency drift (base vs instruct)

Fire-set Jaccard ≈ 1.0 at every layer (the *same* dense features fire in both configs), but the per-feature firing-**rate** Pearson r falls with depth — the same features fire at drifted rates:

| layer | freq r (chat) | freq r (web) |
|---|---|---|
| 0 | 0.965 | 0.958 |
| 6 | 0.945 | 0.918 |
| 12 | 0.795 | 0.682 |
| 18 | 0.864 | 0.908 |
| 25 | 0.808 | 0.712 |

Individual features move sharply, e.g. chat L25 `f13822`: fires on 35% of base tokens → **99.9%** of instruct tokens.

## Conclusions

1. **The GemmaScope transcoders do not transfer cleanly to instruct hidden states.** Reconstruction error is higher on instruct inputs at *every* layer — modest in the middle (~8–13%) but large at the endpoints (L0 +~32%, **last layer roughly doubles**). Sparsity (L0) is consistently a bit lower on instruct.
2. **The shift is general fine-tuning drift, not chat-formatting.** The base→instruct gap is essentially the same on plain web text as on chat (both ~+30% at L0, both ~2× at L25), and firing-frequency drift is if anything *larger* on web.
3. **Same features, drifted rates.** The set of active features is unchanged (Jaccard ≈ 1); what moves is how often each fires (freq r drops to ~0.7–0.8 in deep layers), concentrated in the same mid/late layers where FVU is worst.
4. **Caveat:** base FVU is already high in mid-layers (0.34–0.50) — GemmaScope transcoders reconstruct gemma-2-2b's mid-layer MLPs imperfectly on our data even for base, so mid-layer absolute reconstruction is limited regardless of source.

**Implication:** a short re-fine-tune of the transcoders on instruct-distribution inputs (loss `‖transcoder(x) − MLP_base(x)‖` with `x` = instruct hidden states) is justified, prioritizing the last layer. Alternatively pivot to adapter-only / difference circuits or ReLP neuron-level attribution (Experiment 2, deferred).

## Artifacts

| run | job | data | output dir | wandb |
|---|---|---|---|---|
| smoke | 16113462 | chat, L0, 8k tok | `…/transcoder_input_shift/gemmascope_width_16k_average_l0_76_base-instruct_20260709_024443_16113462` | — |
| **chat** | 16113538 | chat, 5 layers, 1M tok | `…_20260709_024942_16113538` | [kqlkler4](https://wandb.ai/siddharth-stanford/transcoder-feature-collection/runs/kqlkler4) |
| **web** | 16113604 | fineweb, 5 layers, 1M tok | `…_20260709_030301_16113604` | [259j6xnp](https://wandb.ai/siddharth-stanford/transcoder-feature-collection/runs/259j6xnp) |
| **all-layers** | 16113808 | chat, all 26 layers, 500k tok | `…_20260709_031454_16113808` | [i5yt4w7e](https://wandb.ai/siddharth-stanford/transcoder-feature-collection/runs/i5yt4w7e) |

Output root: `$LARGE_ARTIFACTS_DIR/transcoder-adapters/transcoder_input_shift/` (`$LARGE_ARTIFACTS_DIR=/nlp/scr/siddharth`). Each run dir has `results.json`, `summary.md`, `per_feature_fire_freq.npz`, `fire_freq_drift.{json,md}`, `gemmascope_config.json`. Slurm logs: `logs/transcoder_input_shift/*_{16113462,16113538,16113604}.{out,err}`.

## Reproduce

```bash
# Smoke (layer 0, 2k tokens):
./sh/slurm_batch_transcoder_input_shift.sh --gemmascope_width width_16k --gemmascope_l0 average_l0_76 \
  --sources base instruct --layers 0 --max_tokens 2000 --no-wandb
# Chat, 5 layers, 1M tokens:
./sh/slurm_batch_transcoder_input_shift.sh --gemmascope_width width_16k --gemmascope_l0 average_l0_76 \
  --sources base instruct --layers 0 6 12 18 25 --max_tokens 1000000 --wandb
# Web variant: add --val_data fineweb:science-of-finetuning/fineweb-1m-sample
# Firing-frequency drift (CPU, local):
uv run --no-sync python -m analysis.features.analyze_fire_freq_drift --run_dir <run_dir>
```

## Follow-up experiments (tasks 1, 2, 4)

### Task 1 — Re-fine-tune the transcoders on instruct inputs ✅

Code: [`analysis/features/finetune_transcoder_shift.py`](../../analysis/features/finetune_transcoder_shift.py) (runners `sh/slurm_batch_finetune_transcoder_shift.sh`, `run_on_gpu/run_finetune_transcoder_shift.sh`). Fine-tuned the worst-hit endpoint layers (0, 24, 25) starting from the pretrained GemmaScope weights, minimizing `MSE(transcoder(x), MLP_base(x))` with `x` = instruct `ln2.hook_normalized`, ~2M instruct tokens, lr 1e-4, JumpReLU threshold frozen (preserves L0). FVU/L0 on 200k held-out instruct tokens, before vs after (job 16114293, [wandb 61xzqknr](https://wandb.ai/siddharth-stanford/transcoder-feature-collection/runs/61xzqknr)):

| layer | FVU before (instruct) | FVU after | ΔFVU | base-input FVU (target) |
|---|---|---|---|---|
| 0 | 0.108 | **0.075** | −31% | 0.082 |
| 24 | 0.281 | **0.221** | −22% | 0.21 |
| 25 | 0.317 | **0.212** | −33% | 0.15 |

**A short (2M-token, ~14 min) re-fine-tune closes most of the shift**: L0 and L24 recover to their base-input reconstruction level; L25 improves −33% (still above base — more tokens/epochs would close it further). Figure: `.claude/products/transcoder_input_shift/transcoder_finetune_before_after.png`. Fine-tuned params saved as `finetuned_layer_{0,24,25}.safetensors` in the run dir. This confirms the "re-fine-tune the transcoders" fix is cheap and effective.

### Tasks 2 & 4 — Difference circuits + graph clarity ✅

Code: [`analysis/attribution/analyze_graph_clarity.py`](../../analysis/attribution/analyze_graph_clarity.py) over the existing base-vs-adapter combined-attribution graphs ([`experiments/base_vs_adapter_circuit_trace/`](../base_vs_adapter_circuit_trace/), 28 prompts). The graph-level analog of reconstruction failure is the **error node** (`true_mlp_out − transcoder_out`); a graph dominated by feature nodes is "clean". Result (`graph_clarity.json`):

| bucket | error-node fraction | adapter fraction | adapter content / template features |
|---|---|---|---|
| agree | 0.028 | 0.039 | 4.0 / 40.1 |
| diverge | 0.026 | 0.048 | 21.6 / 39.4 |

- **Task 4 (clarity):** error nodes are only **~2.7%** of feature+error nodes — the base GemmaScope graphs stay **feature-dominated and clean** even though Exp 1 shows the reconstruction degrades on shifted inputs. So the FVU shift, while real, does not blow up graph interpretability at these node thresholds.
- **Task 2 (difference circuits):** the adapter contributes only ~4–5% of feature nodes, but its *content-token* work jumps ~5× on divergent prompts (adapter_content 21.6 vs 4.0) — the difference circuit fires exactly where base and instruct diverge.

### Circuit-tracer graphs: original vs fine-tuned transcoders ✅

Code: [`analysis/attribution/compare_finetuned_transcoder_graphs.py`](../../analysis/attribution/compare_finetuned_transcoder_graphs.py) (runners `sh/slurm_batch_compare_finetuned_graphs.sh`). Builds each prompt's attribution graph twice on the base model — pretrained GemmaScope vs the fine-tuned layers 0/24/25 patched in — and compares error-node fraction (job 16118628, 3 `interesting_small` prompts):

| prompt | error-frac original → fine-tuned |
|---|---|
| bomb_refusal_help_I | 0.253 → **0.239** |
| capital_colesseum | 0.275 → **0.261** |
| capital_colesseum_mispelling | 0.273 → **0.260** |

Fine-tuning **just 3 of 26 layers** consistently lowers the error-node fraction (~5% relative) — fewer MLP-reconstruction-error nodes, more of the graph carried by interpretable features. So the fine-tune improves the actual graphs, not just FVU. Serve side by side: `circuit-tracer serve --graph_file_dir <out>/original --port 8050` and `.../finetuned --port 8051` (see `my_notes/07-09-26.md` for the full commands).

## Next steps

- **Push L25 further** — more tokens / a second epoch, or also fine-tune the threshold, to fully close the last-layer gap; then export the fine-tuned set to circuit-tracer and re-run graphs.
- **Experiment 2** (deferred): pivot to ReLP neuron-level attribution using base MLP neurons directly ([`analysis/attribution/relp_model.py`](../../analysis/attribution/relp_model.py)).

## Follow-up artifacts

| run | job | what | output dir | wandb |
|---|---|---|---|---|
| ft smoke | 16114278 | L25, 20k tok | `…/transcoder_input_shift_finetune/ft_..._L25_20260709_041359_16114278` | — |
| **re-finetune** | 16114293 | L0/24/25, 2M tok | `…/transcoder_input_shift_finetune/ft_..._L0-24-25_20260709_041813_16114293` | [61xzqknr](https://wandb.ai/siddharth-stanford/transcoder-feature-collection/runs/61xzqknr) |

Figures under `.claude/products/transcoder_input_shift/` (gitignored): `transcoder_input_shift_overview.png`, `transcoder_finetune_before_after.png`.
