# Circuit-tracer graphs: original vs fine-tuned GemmaScope transcoders

Base model `google/gemma-2-2b`, fine-tuned layers [0, 24, 25]. Error node = MLP reconstruction error; lower error fraction = cleaner graph.

| prompt | orig error-frac | ft error-frac | orig err/feat | ft err/feat |
|---|---|---|---|---|
| bomb_refusal_help_I | 0.258 | 0.239 | 57/164 | 53/169 |
| capital_colesseum | 0.275 | 0.265 | 76/200 | 72/200 |
| capital_colesseum_mispelling | 0.273 | 0.260 | 77/205 | 72/205 |
