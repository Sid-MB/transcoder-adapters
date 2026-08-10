#!/usr/bin/env python
"""Re-judge a refusal_prefill_flip run on the CONTINUATION ONLY, removing the prefill confound.

Session "[refusal-token-tracing]" (fd1f0d19-f5d1-48cc-9002-3f06f2abe2fd), 2026-08-10.

refusal_prefill_flip.py judges the FULL assistant turn (prefilled opening + continuation). At k>=2 the
prefilled opening literally contains "I cannot ...", so the judge calls the turn REFUSAL no matter what the
model continued -- inflating the base refusal curve into a language-modelling artifact. The scientific
question is whether the model, *on its own*, carries the refusal forward. This script re-scores only the
text the model actually generated after the seed (record["continuation"]), reusing the same judge/rubric,
and writes the honest curve alongside the original full-turn one.

Reads ``--results`` (a refusal_prefill_flip results.json), judges each record's continuation, and writes,
next to it: results_continuation.json (records gain label_continuation/confidence_continuation/rationale_
continuation), summary_continuation.md, and prefill_flip_continuation.png (full-turn vs continuation-only
refusal rate per arm). Interpretation: if the base continuation-only refusal rate stays ~0% across k while
the full-turn rate climbs to 100%, the opening tokens do NOT load a genuine refusal -- the base model just
rambles/loops after the injected words; the refusal is the instruct machinery's, not the token identity's.

Example:
    PYTHONPATH=. uv run --extra viz python experiments/interesting_queries/scripts/rejudge_prefill_continuation.py \\
        --results experiments/interesting_queries/results/refusal_prefill_flip_chat/results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from helpers.log import logger, setup_logging  # noqa: E402
from experiments.interesting_queries.scripts.find_strict_compliance_refusal import (  # noqa: E402
    JUDGE_SYSTEM,
    generate_texts,
    judge_user_prompt,
    parse_judgment,
    release_model,
)

OUTCOMES = ("REFUSAL", "COMPLIANCE", "INCONCLUSIVE", "GIBBERISH")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", required=True, type=Path, help="results.json from refusal_prefill_flip.py to re-judge.")
    p.add_argument("--judge_model", default="Qwen/Qwen3-8B", help="Judge model; keep it the same as the original run so labels are comparable.")
    p.add_argument("--judge_max_new_tokens", type=int, default=160, help="Token budget per judge verdict.")
    p.add_argument("--judge_batch_size", type=int, default=8, help="Judge generation batch size.")
    p.add_argument("--device", default="cuda", help="Torch device.")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Judge dtype.")
    return p.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def main() -> None:
    setup_logging()
    args = parse_args()
    logging.getLogger().setLevel(logging.INFO)

    payload = json.loads(args.results.read_text())
    records = payload["records"]
    logger.info("Re-judging %d records (continuation only) from %s", len(records), args.results)

    jt = AutoTokenizer.from_pretrained(args.judge_model)
    if jt.pad_token_id is None:
        jt.pad_token = jt.eos_token
    jt.padding_side = "left"
    judge = AutoModelForCausalLM.from_pretrained(args.judge_model, dtype=_dtype(args.dtype)).to(args.device).eval()

    try:
        encoded = []
        for rec in records:
            # Judge ONLY the model's own words. Empty continuation -> explicit sentinel so the judge sees "nothing".
            response = (rec.get("continuation") or "").strip() or "(the model generated nothing)"
            msgs = [{"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": judge_user_prompt(rec["request"], response)}]
            encoded.append(jt.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False))
        raw = generate_texts(judge, jt, encoded, device=args.device, batch_size=args.judge_batch_size,
                             max_new_tokens=args.judge_max_new_tokens, do_sample=False)
    finally:
        release_model(judge)

    for rec, text in zip(records, raw):
        v = parse_judgment(text)
        rec["label_continuation"] = v["label"]
        rec["confidence_continuation"] = v["confidence"]
        rec["rationale_continuation"] = v["rationale"]

    # Recompute per-(arm, k) stats for both the original full-turn label and the new continuation-only label.
    max_prefill = payload["configuration"]["max_prefill"]
    arms = payload["configuration"]["continue_with"]
    n_prompts = len(payload["configuration"]["prompt_ids"])

    def rate(field: str, arm: str, k: int) -> tuple[float, dict[str, int]]:
        labels = [r[field] for r in records if r["arm"] == arm and r["k"] == k]
        counts = {o: labels.count(o) for o in OUTCOMES}
        return (counts["REFUSAL"] / len(labels) if labels else 0.0), counts

    stats_cont: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for arm in arms:
        for k in range(max_prefill + 1):
            r, counts = rate("label_continuation", arm, k)
            stats_cont[arm][k] = {"n": n_prompts, "counts": counts, "refusal_rate": r}
    payload["stats_continuation"] = stats_cont

    out_json = args.results.with_name("results_continuation.json")
    out_json.write_text(json.dumps(payload, indent=2) + "\n")

    # summary_continuation.md: full-turn vs continuation-only refusal, side by side.
    tmpl = payload["configuration"]["prompt_template"]
    lines = ["# Continuation-only re-judge: does the model itself carry the refusal?", "",
             f"Template `{tmpl}`. **Full-turn** = original label on (prefill + continuation); **cont-only** = "
             "this re-judge on the model's generated text alone. A full-turn refusal that vanishes under "
             "cont-only was carried by the injected words, not the model.", ""]
    for arm in arms:
        lines += [f"## `{arm}`", "", "| k | full-turn refusal | cont-only refusal | cont-only outcome mix |",
                  "|---:|---:|---:|---|"]
        for k in range(max_prefill + 1):
            full_r = payload["stats"][arm][str(k)]["refusal_rate"] if str(k) in payload["stats"][arm] else payload["stats"][arm][k]["refusal_rate"]
            cont = stats_cont[arm][k]
            mix = ", ".join(f"{o[:4]}={cont['counts'][o]}" for o in OUTCOMES if cont['counts'][o])
            lines.append(f"| {k} | {full_r*100:.0f}% | {cont['refusal_rate']*100:.0f}% | {mix} |")
        lines.append("")
    out_md = args.results.with_name("summary_continuation.md")
    out_md.write_text("\n".join(lines) + "\n")

    # Plot: full-turn (dashed) vs continuation-only (solid) refusal per arm.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = list(range(max_prefill + 1))
    colours = {"base": "#c53030", "instruct": "#2b6cb0", "adapter": "#2f855a"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for arm in arms:
        full = [(payload["stats"][arm][str(k)]["refusal_rate"] if str(k) in payload["stats"][arm] else payload["stats"][arm][k]["refusal_rate"]) * 100 for k in ks]
        cont = [stats_cont[arm][k]["refusal_rate"] * 100 for k in ks]
        c = colours.get(arm)
        ax.plot(ks, full, marker="o", linestyle="--", color=c, alpha=0.5, label=f"{arm} full-turn")
        ax.plot(ks, cont, marker="o", linestyle="-", color=c, label=f"{arm} cont-only")
    ax.set_xlabel("# of instruct refusal tokens prefilled (k)")
    ax.set_ylabel("% of prompts judged REFUSAL")
    ax.set_ylim(-3, 103)
    ax.set_xticks(ks)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(f"Full-turn vs continuation-only refusal ({tmpl} template)\n"
                 "cont-only staying low = model does not carry the refusal itself")
    fig.tight_layout()
    out_png = args.results.with_name("prefill_flip_continuation.png")
    fig.savefig(out_png, dpi=160)

    logger.info("Wrote %s, %s, %s", out_json, out_md, out_png)
    for arm in arms:
        cont = " ".join(f"k{k}:{stats_cont[arm][k]['refusal_rate']*100:.0f}%" for k in ks)
        logger.info("  cont-only %-10s %s", arm, cont)


if __name__ == "__main__":
    main()
