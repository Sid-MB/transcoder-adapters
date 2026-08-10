# Refusal ladder: does the transcoder adapter reproduce instruct refusal behaviour?

Every arm is prompted with the **same** `google/gemma-2-2b-it` chat template, greedy-decoded for 256 tokens, and labelled by an independent judge (`Qwen/Qwen3-8B`).

## Harmful requests (n=63) — refusal is the DESIRED behaviour

| arm | refusal % (95% CI) | REFUSAL | COMPLIANCE | INCONCLUSIVE | GIBBERISH |
|---|---:|---:|---:|---:|---:|
| `base` | **8%** (3–17) | 5 | 1 | 11 | 46 |
| `base_plus_attn` | **38%** (27–50) | 24 | 20 | 12 | 7 |
| `hybrid_ft` | **10%** (4–19) | 6 | 4 | 13 | 40 |
| `adapter_tc8192` | **94%** (85–98) | 59 | 0 | 4 | 0 |
| `adapter_huge` | **95%** (87–98) | 60 | 0 | 3 | 0 |
| `instruct` | **100%** (94–100) | 63 | 0 | 0 | 0 |

## Benign requests (n=30) — refusal here is OVER-refusal (bad)

| arm | refusal % (95% CI) | REFUSAL | COMPLIANCE | INCONCLUSIVE | GIBBERISH |
|---|---:|---:|---:|---:|---:|
| `base` | **0%** (0–11) | 0 | 0 | 12 | 18 |
| `base_plus_attn` | **3%** (1–17) | 1 | 20 | 8 | 1 |
| `hybrid_ft` | **0%** (0–11) | 0 | 2 | 13 | 15 |
| `adapter_tc8192` | **0%** (0–11) | 0 | 23 | 7 | 0 |
| `adapter_huge` | **0%** (0–11) | 0 | 24 | 6 | 0 |
| `instruct` | **0%** (0–11) | 0 | 25 | 5 | 0 |

**Transcoder contribution** (adapter_huge − base_plus_attn on harmful): **+57 pp** — the part of refusal behaviour attributable to the trained transcoder rather than to instruct attention alone.

Per-prompt labels for every arm: [`per_prompt_labels.md`](per_prompt_labels.md). Transcripts: [`transcripts/`](transcripts) (every model's full response per prompt). Chart: `refusal_ladder.png`.

