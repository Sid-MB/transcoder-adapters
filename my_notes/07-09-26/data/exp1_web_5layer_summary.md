# Transcoder input-shift eval — google/gemma-scope-2b-pt-transcoders/width_16k/average_l0_76_nearest

GemmaScope `width_16k` / `average_l0_76` transcoders on `google/gemma-2-2b`; target = MLP_base(x). Reconstruction FVU and L0 per source × layer.

| source | layer | n_tokens | L0 | FVU | MSE | cosine |
|---|---|---|---|---|---|---|
| base | 0 | 1000435 | 75.1 | 0.0748 | 0.05132 | 0.9630 |
| base | 6 | 1000435 | 99.7 | 0.3424 | 0.19567 | 0.8629 |
| base | 12 | 1000435 | 60.6 | 0.4862 | 0.51656 | 0.7466 |
| base | 18 | 1000435 | 61.1 | 0.3195 | 1.37948 | 0.8703 |
| base | 25 | 1000435 | 71.2 | 0.1769 | 4.42887 | 0.9137 |
| instruct | 0 | 1000435 | 65.2 | 0.0977 | 0.06808 | 0.9566 |
| instruct | 6 | 1000435 | 95.6 | 0.3881 | 0.23510 | 0.8387 |
| instruct | 12 | 1000435 | 65.0 | 0.5495 | 0.66876 | 0.7056 |
| instruct | 18 | 1000435 | 56.4 | 0.3484 | 1.48328 | 0.8569 |
| instruct | 25 | 1000435 | 61.5 | 0.4064 | 6.01132 | 0.9529 |

**Reading it:** if `instruct` L0/FVU ≈ `base`, the GemmaScope transcoders transfer; if materially higher, the input shift warrants re-fine-tuning (or a pivot to adapter-only / ReLP). Per-feature fire frequencies are in `per_feature_fire_freq.npz`.

Invocation: `/juice2/u/siddharth/transcoder-adapters/analysis/features/transcoder_input_shift.py --gemmascope_width width_16k --gemmascope_l0 average_l0_76 --sources base instruct --layers 0 6 12 18 25 --max_tokens 1000000 --val_data fineweb:science-of-finetuning/fineweb-1m-sample --wandb`
