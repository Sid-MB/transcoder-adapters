#!/usr/bin/env python
"""Pick the refusal prompts worth circuit-tracing, and emit prompts targeting the "I" / "cannot" tokens.

Created by Claude Code session "hybrid with base+attention" (b395277e-a91a-4e08-aea3-f26b6ef5a915).

Consumes ``eval_refusal_ladder.py``'s results.json and selects the prompts in the "very nearby
range" -- where the transcoder adapter reproduces the instruct model's refusal -- because a circuit
traced in the adapter only explains the real model's behaviour on prompts where the two agree.

Selection (all must hold):
  * instruct and the chosen adapter arm are BOTH judged REFUSAL, each at >= --min_confidence;
  * both responses begin with "I cannot" / "I can't" (case-insensitive), so the two traced token
    positions are unambiguous rather than assumed;
  * base does NOT refuse -- these are the strict flips, so the refusal is genuinely added by the
    instruct-side machinery rather than already present in base.

For each selected prompt it writes two attribution prompts (per the meeting note "the refusals are
probably in the 'I' or the 'cannot'"; Anthropic trace the "I" because the model is trained to open
refusals with it):
  <id>__tok_I.txt        assistant prefix ""  -> target token "I"       (is refusal decided at the very first response token?)
  <id>__tok_cannot.txt   assistant prefix "I" -> target token " cannot" (or is "I" uncommitted, and the decision happens at the next token?)

The .txt files use the DeepSeek markers that run_base_adapter_comparison re-renders with the real
chat template (``--prompt_format chat``); the assistant content's LAST token is the held-out target.

Example:
    uv run --no-sync python experiments/interesting_queries/scripts/select_refusal_trace_prompts.py \\
        --results experiments/interesting_queries/results/refusal_ladder/results.json --n_select 3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.interesting_queries.scripts.common import write_prompt_txt  # noqa: E402
from helpers.log import logger, setup_logging  # noqa: E402

REFUSAL_OPENER = re.compile(r"^\s*I\s+(cannot|can't|can not)\b", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="experiments/interesting_queries/results/refusal_ladder/results.json",
                   help="results.json written by eval_refusal_ladder.py.")
    p.add_argument("--adapter_arm", default="adapter_huge",
                   help="Which adapter arm must agree with instruct; this is also the model you should trace (use the arm whose refusal rate is closest to instruct).")
    p.add_argument("--base_arm", default="base", help="Arm that must NOT refuse, so the traced behaviour is genuinely added rather than pre-existing.")
    p.add_argument("--instruct_arm", default="instruct", help="Reference arm defining the target behaviour.")
    p.add_argument("--min_confidence", type=float, default=0.8, help="Minimum judge confidence for both REFUSAL labels; raise for a stricter, smaller set.")
    p.add_argument("--n_select", type=int, default=3, help="How many prompts to emit graphs for. The meeting note says do FEWER graphs with potential decisions, so 2-3 is the intent.")
    p.add_argument("--output_dir", default="experiments/interesting_queries/prompts/refusal_tokens",
                   help="Where the <id>__tok_{I,cannot}.txt attribution prompts are written.")
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    data = json.loads(Path(args.results).read_text())
    records = data["records"]

    candidates = []
    for rec in records:
        if rec["set"] != "harmful":
            continue
        resp = rec["responses"]
        if not all(a in resp for a in (args.instruct_arm, args.adapter_arm, args.base_arm)):
            continue
        ins, ada, bas = resp[args.instruct_arm], resp[args.adapter_arm], resp[args.base_arm]
        if ins.get("label") != "REFUSAL" or ada.get("label") != "REFUSAL":
            continue
        if min(ins.get("confidence", 0), ada.get("confidence", 0)) < args.min_confidence:
            continue
        if bas.get("label") == "REFUSAL":
            continue  # not a flip: base already refuses, so there is nothing added to explain
        # Require the literal opener in BOTH so "I" and " cannot" are really the next two tokens.
        if not (REFUSAL_OPENER.match(ins["text"]) and REFUSAL_OPENER.match(ada["text"])):
            continue
        candidates.append({
            "id": rec["id"], "category": rec["category"], "request": rec["request"],
            "instruct_text": ins["text"], "adapter_text": ada["text"],
            "score": min(ins["confidence"], ada["confidence"]),
        })

    candidates.sort(key=lambda c: -c["score"])
    selected = candidates[: args.n_select]
    logger.info("%d/%d harmful prompts qualify (adapter+instruct both refuse with 'I cannot', base does not); emitting %d",
                len(candidates), sum(r["set"] == "harmful" for r in records), len(selected))
    if not selected:
        raise SystemExit("no prompts satisfied the agreement criteria -- loosen --min_confidence or check the arms")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for c in selected:
        # Target the first response token: is the refusal already decided at "I"?
        p_i = out / f"{c['id']}__tok_I.txt"
        write_prompt_txt(p_i, c["request"], "", "I")
        # Target the next token given "I": or is "I" uncommitted and the decision lands on "cannot"?
        p_c = out / f"{c['id']}__tok_cannot.txt"
        write_prompt_txt(p_c, c["request"], "I", " cannot")
        written += [str(p_i), str(p_c)]
        logger.info("  %s (conf %.2f) -> %s | %s", c["id"], c["score"], p_i.name, p_c.name)

    manifest = out / "selection.json"
    manifest.write_text(json.dumps({
        "source_results": args.results,
        "criteria": {"adapter_arm": args.adapter_arm, "instruct_arm": args.instruct_arm, "base_arm": args.base_arm,
                     "min_confidence": args.min_confidence,
                     "rule": "adapter+instruct both REFUSAL and both open with 'I cannot'; base not REFUSAL"},
        "n_candidates": len(candidates), "selected": selected, "prompt_files": written,
    }, indent=2) + "\n")
    logger.info("Wrote %s and %d prompt files in %s", manifest, len(written), out)


if __name__ == "__main__":
    main()
