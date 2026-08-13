# Continuation-only re-judge: does the model itself carry the refusal?

Template `chat`. **Full-turn** = original label on (prefill + continuation); **cont-only** = this re-judge on the model's generated text alone. A full-turn refusal that vanishes under cont-only was carried by the injected words, not the model.

## `base`

| k | full-turn refusal | cont-only refusal | cont-only outcome mix |
|---:|---:|---:|---|
| 0 | 4% | 4% | REFU=2, COMP=1, INCO=8, GIBB=41 |
| 1 | 10% | 6% | REFU=3, COMP=2, INCO=20, GIBB=27 |
| 2 | 60% | 19% | REFU=10, COMP=1, INCO=25, GIBB=16 |
| 3 | 88% | 50% | REFU=26, COMP=2, INCO=18, GIBB=6 |
| 4 | 92% | 48% | REFU=25, COMP=3, INCO=17, GIBB=7 |
| 5 | 92% | 40% | REFU=21, COMP=2, INCO=19, GIBB=10 |
| 6 | 100% | 46% | REFU=24, COMP=2, INCO=17, GIBB=9 |
| 7 | 96% | 37% | REFU=19, COMP=4, INCO=18, GIBB=11 |
| 8 | 98% | 46% | REFU=24, COMP=3, INCO=18, GIBB=7 |
| 9 | 98% | 56% | REFU=29, COMP=3, INCO=11, GIBB=9 |
| 10 | 98% | 67% | REFU=35, COMP=2, INCO=11, GIBB=4 |

## `instruct`

| k | full-turn refusal | cont-only refusal | cont-only outcome mix |
|---:|---:|---:|---|
| 0 | 100% | 100% | REFU=52 |
| 1 | 100% | 100% | REFU=52 |
| 2 | 100% | 100% | REFU=52 |
| 3 | 100% | 100% | REFU=52 |
| 4 | 100% | 100% | REFU=52 |
| 5 | 100% | 100% | REFU=52 |
| 6 | 100% | 100% | REFU=52 |
| 7 | 100% | 100% | REFU=52 |
| 8 | 100% | 100% | REFU=52 |
| 9 | 100% | 100% | REFU=52 |
| 10 | 100% | 100% | REFU=52 |

