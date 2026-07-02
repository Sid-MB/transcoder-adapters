# Created by Claude Code session "explore: queries to analyze".
"""FINDER #1 (harmful): find HarmBench prompts where the INSTRUCT model refuses but the BASE
model does NOT (base is untuned -- it just continues; "no refusal markers" = non-refusal).

These isolate the instruction-tuned refusal circuit (the interesting case for circuit tracing):
the adapter must add a refusal that the base lacks. Target token = the FIRST assistant-response
token (where refuse "I..." vs comply "Sure...") is decided; we report its index in the rendered
chat sequence and emit a .txt prompt for run_combined_attribution.

Usage:
  LARGE_ARTIFACTS_DIR=/nlp/scr/siddharth uv run --no-sync python \
    experiments/interesting_queries/scripts/find_harmful.py --n_select 18
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Put repo root (for helpers/models) and this scripts dir (for common) on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402

from helpers.log import logger, setup_logging  # noqa: E402
import common  # noqa: E402

EXP_ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_scan", type=int, default=200, help="How many HarmBench 'standard' prompts to scan (max 200).")
    parser.add_argument("--n_select", type=int, default=18, help="How many instruct-refuses/base-complies prompts to keep, spanning categories.")
    parser.add_argument("--max_new_tokens", type=int, default=40, help="Greedy continuation length for the refusal detector.")
    parser.add_argument("--output_root", type=Path, default=EXP_ROOT)
    args = parser.parse_args()

    logger.info(f"INVOCATION: find_harmful.py {vars(args)}")
    ds = load_dataset("walledai/HarmBench", "standard")["train"]
    prompts = list(zip(ds["prompt"], ds["category"]))[: args.n_scan]
    logger.info(f"Scanning {len(prompts)} HarmBench standard prompts.")

    tokenizer, base, instruct = common.load_models(args.device)

    records = []
    for i, (prompt, category) in enumerate(prompts):
        instruct_r = common.detect_refusal(instruct, tokenizer, prompt, args.max_new_tokens, args.device)
        base_r = common.detect_refusal(base, tokenizer, prompt, args.max_new_tokens, args.device)
        prompt_ids = common.render_chat_ids(tokenizer, prompt)
        tgt = common.first_response_target(instruct, tokenizer, prompt_ids, args.device)
        keep = instruct_r["is_refusal"] and (not base_r["is_refusal"])
        rec = {
            "id": f"harm_{i:03d}",
            "category": category,
            "prompt": prompt,
            "instruct_refuses": instruct_r["is_refusal"],
            "base_refuses": base_r["is_refusal"],
            "instruct_first_token": instruct_r["first_token"],
            "base_first_token": base_r["first_token"],
            "instruct_p_affirmative": instruct_r["p_affirmative"],
            "instruct_p_refusal_first": instruct_r["p_refusal_first"],
            "base_p_affirmative": base_r["p_affirmative"],
            "instruct_gen": instruct_r["generated_text"][:200],
            "base_gen": base_r["generated_text"][:200],
            "target_index": tgt["target_index"],
            "target_token": tgt["target_token"],
            "keep": keep,
        }
        records.append(rec)
        flag = "KEEP" if keep else "    "
        logger.info(f"  [{flag}] {rec['id']} [{category:24s}] instruct_refuse={instruct_r['is_refusal']} base_refuse={base_r['is_refusal']} inst1={instruct_r['first_token']!r} base1={base_r['first_token']!r}")

    kept = [r for r in records if r["keep"]]
    logger.info(f"{len(kept)}/{len(records)} prompts: instruct refuses AND base complies.")

    # Select spanning categories: round-robin across categories, prefer high instruct refusal confidence.
    by_cat = defaultdict(list)
    for r in kept:
        by_cat[r["category"]].append(r)
    for c in by_cat:
        by_cat[c].sort(key=lambda r: -r["instruct_p_refusal_first"])
    selected = []
    cats = list(by_cat.keys())
    rnd = 0
    while len(selected) < args.n_select and any(rnd < len(by_cat[c]) for c in cats):
        for c in cats:
            if rnd < len(by_cat[c]) and len(selected) < args.n_select:
                selected.append(by_cat[c][rnd])
        rnd += 1

    sel_ids = {r["id"] for r in selected}
    for r in records:
        r["selected"] = r["id"] in sel_ids

    prompts_dir = args.output_root / "prompts" / "harmful"
    for r in selected:
        common.write_prompt_txt(prompts_dir / f"{r['id']}.txt", r["prompt"], "", r["target_token"])

    out = args.output_root / "results" / "find_harmful.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "finder": "harmful",
        "criterion": "instruct refuses (regex markers) AND base does not",
        "n_scanned": len(records),
        "n_kept": len(kept),
        "n_selected": len(selected),
        "selected_ids": [r["id"] for r in selected],
        "prompts_dir": str(prompts_dir),
        "records": records,
    }, indent=2) + "\n")
    logger.info(f"Wrote {out}; emitted {len(selected)} prompts to {prompts_dir}")
    common.free_models(base, instruct)


if __name__ == "__main__":
    main()
