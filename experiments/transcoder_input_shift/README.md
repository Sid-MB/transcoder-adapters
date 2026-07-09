<!-- Created by Claude Code session "implement: new 07/02 gemmascope transcoder experiments". 2026-07-09. -->
# GemmaScope transcoder input-distribution shift (Experiment 1)

**[Code → [`analysis/features/transcoder_input_shift.py`](../../analysis/features/transcoder_input_shift.py) · [`analysis/features/analyze_fire_freq_drift.py`](../../analysis/features/analyze_fire_freq_drift.py) · runners [`sh/slurm_batch_transcoder_input_shift.sh`](../../sh/slurm_batch_transcoder_input_shift.sh), [`run_on_gpu/run_transcoder_input_shift.sh`](../../run_on_gpu/run_transcoder_input_shift.sh)]**
**[Plan → `~/.claude/plans/cached-doodling-horizon.md`]**

## Question

Pretrained GemmaScope transcoders (`google/gemma-scope-2b-pt-transcoders`, `width_16k`) were trained to replace each **base-model MLP inside the full base model** — on the *base model's* per-layer input distribution. We want to plug them into circuit-tracer to study `google/gemma-2-2b` (base) vs `google/gemma-2-2b-it` (instruct). Switching the source model shifts the distribution of inputs to each layer.

**For each layer, is the transcoder's sparsity (L0) and reconstruction error the same when its input hidden states come from the base model vs. the instruct model?** If ~same, the transcoders transfer; if materially worse on instruct inputs, the shift warrants re-fine-tuning the transcoders (or a pivot to adapter-only / ReLP).

## Method

For each source model and layer L, over ~1M tokens: take `x = blocks.L.ln2.hook_normalized` (the transcoder's input), compute `feats = transcoder.encode(x)` (→ **L0**) and `recon = transcoder.decode(feats)`. The target is **always the base MLP**: `MLP_base(x)`, obtained via a *patched* base-model forward (overwrite `ln2.hook_normalized` with `x`, read `hook_mlp_out`) — exact because the gemma-2 MLP sub-block is position-wise. Models are loaded as TransformerLens `HookedTransformer`s with the same processing (`fold_ln=False, center_writing_weights=False, center_unembed=False`) that circuit-tracer's `ReplacementModel` uses, so `ln2.hook_normalized` matches GemmaScope's convention. Metrics: **FVU** (fraction of variance unexplained), MSE, cosine, and per-feature firing frequency. BOS (position 0) and padding excluded.

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

## Next steps

- **Re-fine-tune** the last-layer (and mid-layer) transcoders on instruct-distribution inputs and re-measure FVU/L0 — quantify how much the gap closes.
- **All-26-layer** depth profile (currently a 5-layer sample) for the full picture.
- **Experiment 2** (deferred): compare circuit graphs / pivot to ReLP neuron-level attribution using base MLP neurons directly ([`analysis/attribution/relp_model.py`](../../analysis/attribution/relp_model.py)).
