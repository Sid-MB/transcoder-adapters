<!-- Claude Code session "hybrid with base+attention" (b395277e-a91a-4e08-aea3-f26b6ef5a915), 2026-08-10. -->
# Refusal ladder — SUPERSEDED, see [`refusal_ladder.md`](refusal_ladder.md)

This file and the former `refusal-ladder/` directory were an earlier snapshot of the same experiment,
written before the `hybrid_ft` arm was fixed. Its numbers are **stale in one row**: it reported
`hybrid_ft` at **0% refusal (63/63 GIBBERISH)**, which was an artifact of replacing *all 26* MLPs with
GemmaScope transcoders. Restricting replacement to the fine-tuned layers (0/24/25) gives **10%**.

**Canonical writeup:** [`refusal_ladder.md`](refusal_ladder.md) — same experiment, corrected
`hybrid_ft`, plus the refusal-token circuit analysis (refusal commits at the "I").
**Canonical artifacts:** [`refusal_ladder/`](refusal_ladder) — chart, summary, per-prompt labels,
results.json, 93 transcripts.

Headline (unchanged): the adapters refuse **94–95%** of harmful requests vs instruct's **100%**, while
instruct-attention-with-the-transcoder-zeroed manages only **38%** — so **+57 pp** of refusal is
attributable to the trained transcoder, with **0%** over-refusal on benign prompts.
