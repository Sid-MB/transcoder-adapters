# Transcoder input-shift eval — google/gemma-scope-2b-pt-transcoders/width_16k/average_l0_76_nearest

GemmaScope `width_16k` / `average_l0_76` transcoders on `google/gemma-2-2b`; target = MLP_base(x). Reconstruction FVU and L0 per source × layer.

| source | layer | n_tokens | L0 | FVU | MSE | cosine |
|---|---|---|---|---|---|---|
| base | 0 | 1006931 | 80.3 | 0.0823 | 0.05777 | 0.9608 |
| base | 6 | 1006931 | 97.5 | 0.3522 | 0.19005 | 0.8633 |
| base | 12 | 1006931 | 59.0 | 0.4993 | 0.51854 | 0.7490 |
| base | 18 | 1006931 | 53.6 | 0.3397 | 1.26884 | 0.8657 |
| base | 25 | 1006931 | 64.9 | 0.1522 | 4.04161 | 0.9206 |
| instruct | 0 | 1006931 | 73.2 | 0.1094 | 0.07883 | 0.9521 |
| instruct | 6 | 1006931 | 88.4 | 0.3917 | 0.22836 | 0.8422 |
| instruct | 12 | 1006931 | 63.4 | 0.5368 | 0.67097 | 0.7088 |
| instruct | 18 | 1006931 | 59.2 | 0.3749 | 1.68491 | 0.8390 |
| instruct | 25 | 1006931 | 59.6 | 0.3095 | 6.25224 | 0.9357 |

**Reading it:** if `instruct` L0/FVU ≈ `base`, the GemmaScope transcoders transfer; if materially higher, the input shift warrants re-fine-tuning (or a pivot to adapter-only / ReLP). Per-feature fire frequencies are in `per_feature_fire_freq.npz`.

Invocation: `/juice2/u/siddharth/transcoder-adapters/analysis/features/transcoder_input_shift.py --gemmascope_width width_16k --gemmascope_l0 average_l0_76 --sources base instruct --layers 0 6 12 18 25 --max_tokens 1000000 --wandb`
