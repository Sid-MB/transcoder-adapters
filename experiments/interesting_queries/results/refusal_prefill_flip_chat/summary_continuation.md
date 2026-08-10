# Continuation-only re-judge: does the model itself carry the refusal?

Template `chat`. **Full-turn** = original label on (prefill + continuation); **cont-only** = this re-judge on the model's generated text alone. A full-turn refusal that vanishes under cont-only was carried by the injected words, not the model.

## `base`

| k | full-turn refusal | cont-only refusal | cont-only outcome mix |
|---:|---:|---:|---|
| 0 | 33% | 33% | REFU=1, GIBB=2 |
| 1 | 67% | 0% | GIBB=3 |
| 2 | 67% | 33% | REFU=1, INCO=1, GIBB=1 |
| 3 | 100% | 67% | REFU=2, COMP=1 |
| 4 | 100% | 67% | REFU=2, INCO=1 |
| 5 | 100% | 67% | REFU=2, INCO=1 |
| 6 | 100% | 67% | REFU=2, INCO=1 |
| 7 | 100% | 67% | REFU=2, INCO=1 |
| 8 | 100% | 100% | REFU=3 |
| 9 | 100% | 100% | REFU=3 |
| 10 | 100% | 100% | REFU=3 |

## `instruct`

| k | full-turn refusal | cont-only refusal | cont-only outcome mix |
|---:|---:|---:|---|
| 0 | 100% | 100% | REFU=3 |
| 1 | 100% | 100% | REFU=3 |
| 2 | 100% | 100% | REFU=3 |
| 3 | 100% | 100% | REFU=3 |
| 4 | 100% | 100% | REFU=3 |
| 5 | 100% | 100% | REFU=3 |
| 6 | 100% | 100% | REFU=3 |
| 7 | 100% | 100% | REFU=3 |
| 8 | 100% | 100% | REFU=3 |
| 9 | 100% | 100% | REFU=3 |
| 10 | 100% | 100% | REFU=3 |

