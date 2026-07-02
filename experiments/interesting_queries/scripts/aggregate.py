# Created by Claude Code session "explore: queries to analyze".
"""Aggregate the three finders' results into results/interesting_queries.json and a readable
results/interesting_queries.md table (type / query / target_token / metric / rendered prompt).

Also (re)emits the combined prompts/ tree already written by the finders; this step only reads the
per-finder json and consolidates. Run after find_harmful / find_divergent / find_adv_suffix.

Usage:
  uv run --no-sync python experiments/interesting_queries/scripts/aggregate.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from helpers.log import logger, setup_logging  # noqa: E402

EXP_ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    p = EXP_ROOT / "results" / name
    return json.loads(p.read_text()) if p.exists() else None


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_root", type=Path, default=EXP_ROOT)
    args = parser.parse_args()

    harmful = _load("find_harmful.json")
    divergent = _load("find_divergent.json")
    adv = _load("find_adv_suffix.json")

    entries = []

    if harmful:
        for r in harmful["records"]:
            if not r.get("selected"):
                continue
            entries.append({
                "type": "#1 harmful",
                "id": r["id"],
                "query": r["prompt"],
                "suffix": "",
                "target_index": r["target_index"],
                "target_token": r["target_token"],
                "why": f"first response token: instruct refuses ({r['instruct_first_token']!r}) but base complies ({r['base_first_token']!r})",
                "metric": f"category={r['category']}; instruct P(I)={r['instruct_p_refusal_first']:.2f}, base P(affirm)={r['base_p_affirmative']:.2f}",
                "prompt_txt": f"prompts/harmful/{r['id']}.txt",
            })

    if divergent:
        for rid in divergent["selected_ids"]:
            r = next(x for x in divergent["records"] if x["id"] == rid)
            entries.append({
                "type": "#2 divergent",
                "id": r["id"],
                "query": r["prompt"],
                "suffix": "",
                "target_index": r.get("argmax_seq_index"),
                "target_token": r.get("target_token_written", r.get("instruct_top1_token")),
                "why": f"argmax-KL response pos {r['argmax_pos']}: base->{r['base_top1_token']!r} vs instruct->{r['instruct_top1_token']!r}",
                "metric": f"max KL={r['max_kl']:.2f} nats ({r['source']})",
                "prompt_txt": f"prompts/divergent/{r['id']}.txt",
            })

    if adv:
        for r in adv["records"]:
            if not r.get("any_flip"):
                continue
            b = r["best"]
            entries.append({
                "type": "#3 adv_suffix",
                "id": r["id"],
                "query": r["prompt"],
                "suffix": b["suffix"],
                "target_index": None,
                "target_token": f"{b['plain_first_token']!r} (no-suffix) -> {b['suffix_first_token']!r} (with suffix)",
                "why": f"first response token flip via {b['method']}: refuses without suffix, complies with",
                "metric": f"flipped=True; suffix P(affirm)={b['suffix_p_affirmative']:.2f}",
                "prompt_txt": f"prompts/adv_suffix/{r['id']}_with_suffix.txt",
            })

    payload = {
        "counts": {
            "#1 harmful": sum(1 for e in entries if e["type"] == "#1 harmful"),
            "#2 divergent": sum(1 for e in entries if e["type"] == "#2 divergent"),
            "#3 adv_suffix": sum(1 for e in entries if e["type"] == "#3 adv_suffix"),
        },
        "entries": entries,
    }
    out_json = args.output_root / "results" / "interesting_queries.json"
    out_json.write_text(json.dumps(payload, indent=2) + "\n")

    lines = ["# Interesting queries for circuit-tracing analysis", "",
             "Created by session \"explore: queries to analyze\". See README.md for reproduction.", "",
             f"Counts: {payload['counts']}", "",
             "| type | query (+suffix) | target token (index) | why | metric | prompt |",
             "|---|---|---|---|---|---|"]
    for e in entries:
        q = e["query"].replace("|", "\\|").replace("\n", " ")
        if e["suffix"]:
            q += f" **[+suffix: {e['suffix'][:60].replace('|', '')}...]**"
        tgt = f"{e['target_token']}".replace("|", "\\|")
        idx = e["target_index"]
        lines.append(f"| {e['type']} | {q} | {tgt} (idx {idx}) | {e['why'].replace('|', chr(92)+'|')} | {e['metric'].replace('|', chr(92)+'|')} | `{e['prompt_txt']}` |")
    out_md = args.output_root / "results" / "interesting_queries.md"
    out_md.write_text("\n".join(lines) + "\n")

    logger.info(f"Aggregated {len(entries)} entries -> {out_json} and {out_md}")
    logger.info(f"Counts: {payload['counts']}")


if __name__ == "__main__":
    main()
