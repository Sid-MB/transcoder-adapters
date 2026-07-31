# Transcoder re-fine-tune on instruct inputs — google/gemma-scope-2b-pt-transcoders/width_16k/average_l0_76_nearest

Fine-tuned GemmaScope `width_16k`/`average_l0_76` transcoders at layers [0, 24, 25] on ~2,000,000 instruct tokens (loss = MSE(transcoder(x), MLP_base(x)), x = instruct ln2.hook_normalized; threshold frozen). FVU/L0 on 200,000 held-out instruct tokens, before vs after.

| layer | FVU before | FVU after | ΔFVU | L0 before | L0 after |
|---|---|---|---|---|---|
| 0 | 0.1081 | 0.0746 | -31% | 72.4 | 81.0 |
| 24 | 0.2813 | 0.2207 | -22% | 59.8 | 66.6 |
| 25 | 0.3166 | 0.2120 | -33% | 59.6 | 81.4 |

For reference, the pre-finetune base-input FVU (Exp 1) was ~0.082 (L0), ~0.21 (L24), ~0.15 (L25).
