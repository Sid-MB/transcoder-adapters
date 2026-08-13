#!/usr/bin/env python
"""Figure: does prefilling the instruct model's refusal opening make the BASE model refuse? (n=52)

Claude Code session "hybrid with base+attention" (b395277e-a91a-4e08-aea3-f26b6ef5a915), 2026-08-12.

Two panels (chat / plain prompt template). Per arm, two curves:
  * dashed, faded = FULL-TURN judging (prefill + continuation). CONFOUNDED: at k>=2 the prefilled text
    literally contains "I cannot ...", so the judge sees a refusal regardless of what the model wrote.
  * solid = CONTINUATION-ONLY judging (only the tokens the model itself generated). This is the
    measure to read.

Run: uv run --no-sync python my_notes/08-10-26/refusal_prefill_n52/make_figure.py
"""
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "experiments/interesting_queries/results/refusal_prefill_flip_n52_%s"
OUT = "my_notes/08-10-26/refusal_prefill_n52"
COL = {"base": "#c53030", "instruct": "#2b6cb0"}


def load(tmpl):
    d = json.load(open((SRC % tmpl) + "/results_continuation.json"))
    arms = d["configuration"]["continue_with"]
    K = d["configuration"]["max_prefill"]
    get = lambda block, a, k: block[a][str(k)]["refusal_rate"] * 100 if str(k) in block[a] else block[a][k]["refusal_rate"] * 100
    full = {a: {k: get(d["stats"], a, k) for k in range(K + 1)} for a in arms}
    cont = {a: {k: get(d["stats_continuation"], a, k) for k in range(K + 1)} for a in arms}
    n = len({r["id"] for r in d["records"]})
    return full, cont, K, n


fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
for ax, tmpl in zip(axes, ["chat", "plain"]):
    full, cont, K, n = load(tmpl)
    ks = list(range(K + 1))
    for a in full:
        ax.plot(ks, [full[a][k] for k in ks], marker="o", ms=4, ls="--", color=COL[a], alpha=0.40,
                label=f"{a} — full turn (confounded)")
        ax.plot(ks, [cont[a][k] for k in ks], marker="o", ms=5, ls="-", lw=2, color=COL[a],
                label=f"{a} — continuation only")
    ax.axvline(1, color="#888", ls=":", lw=1)
    ax.annotate('prefill = "I"', xy=(1, 96), fontsize=7, color="#555", rotation=90, va="top", ha="right")
    ax.set_title(f"{tmpl} template  (n={n} prompts)", fontsize=10)
    ax.set_xlabel('# tokens of the instruct refusal prefilled onto the model  (k)')
    ax.set_xticks(ks)
    ax.set_ylim(-4, 104)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("% judged REFUSAL")
axes[0].legend(fontsize=7, loc="center right")
fig.suptitle("Transplanting the instruct model's refusal opening onto the base model:\n"
             'the "I" alone does not carry the refusal, and even the full opening only partly transfers',
             fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUT}/prefill_flip_n52.pdf")
fig.savefig(f"{OUT}/prefill_flip_n52.png", dpi=160)
print("wrote", f"{OUT}/prefill_flip_n52.{{pdf,png}}")
