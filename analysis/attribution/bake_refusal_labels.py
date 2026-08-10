#!/usr/bin/env python
"""Bake refusal-ladder verdicts into overlay graphs so the viewer shows them per prompt.

Created by Claude Code session "hybrid with base+attention" (b395277e-a91a-4e08-aea3-f26b6ef5a915).

``experiments/interesting_queries/scripts/eval_refusal_ladder.py`` measures, for each prompt, how
every model on the ladder (base -> base+attention -> hybrid -> adapter -> instruct) actually
responded, judged COMPLIANCE / REFUSAL / INCONCLUSIVE / GIBBERISH. This copies those labels into
each graph's ``metadata.refusal_ladder``; ``comparison_frontend`` then renders them as a small
panel under the graph.

Why it matters in the viewer: a circuit traced in the adapter only explains the real model's
behaviour on prompts where the adapter and the instruct model AGREE. Having the labels on screen
means you can see at a glance whether the graph you are reading is such a prompt (all-REFUSAL) or a
divergence case, instead of cross-referencing a table.

Matching is by prompt id (e.g. ``harm_006``) appearing in the graph's filename/slug, which is how
the refusal prompts are named by select_refusal_trace_prompts.py.

Example:
    uv run --no-sync python -m analysis.attribution.bake_refusal_labels \\
        --results experiments/interesting_queries/results/refusal_ladder/results.json \\
        --graph_dirs /nlp/scr/siddharth/transcoder-adapters/base_adapter_comparisons/refusal_tokens/overlay
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from helpers.log import logger, setup_logging

SKIP = {"graph-metadata.json", "run_attribution_args.json", "combined_attribution_manifest.json", "selection.json"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", required=True, help="results.json from eval_refusal_ladder.py (holds each prompt's per-arm judge labels).")
    p.add_argument("--graph_dirs", nargs="+", required=True, help="Graph directories to annotate in place (overlay and/or compact dirs).")
    p.add_argument("--arms", nargs="+", default=None, help="Restrict/order the arms shown in the viewer panel (default: every arm in results.json, in its stored order).")
    p.add_argument("--dry_run", action="store_true", help="Report which graphs would be annotated without writing.")
    return p.parse_args()


def build_index(results_path: Path, arms: list[str] | None) -> dict[str, dict[str, Any]]:
    """prompt_id -> the payload baked into metadata.refusal_ladder."""
    data = json.loads(results_path.read_text())
    index: dict[str, dict[str, Any]] = {}
    for rec in data["records"]:
        names = arms or list(rec["responses"].keys())
        entries = []
        for arm in names:
            r = rec["responses"].get(arm)
            if not r or r.get("label") is None:
                continue
            entries.append({"arm": arm, "label": r["label"], "confidence": r.get("confidence")})
        if entries:
            index[rec["id"]] = {"prompt_id": rec["id"], "set": rec["set"], "category": rec.get("category", ""),
                                "request": rec["request"], "arms": entries,
                                "source": str(results_path)}
    return index


def main() -> None:
    setup_logging()
    args = parse_args()
    index = build_index(Path(args.results), args.arms)
    logger.info("Loaded labels for %d prompts from %s", len(index), args.results)
    # Longest id first so e.g. harm_01 can't shadow harm_010 when both appear in a filename.
    ids = sorted(index, key=len, reverse=True)

    annotated = skipped = 0
    for gdir in args.graph_dirs:
        for path in sorted(Path(gdir).glob("*.json")):
            if path.name in SKIP:
                continue
            match = next((i for i in ids if re.search(rf"(?<![A-Za-z0-9]){re.escape(i)}(?![A-Za-z0-9])", path.stem)), None)
            if match is None:
                skipped += 1
                continue
            if args.dry_run:
                logger.info("[dry-run] %s -> %s", path.name, match)
                annotated += 1
                continue
            graph = json.loads(path.read_text())
            graph.setdefault("metadata", {})["refusal_ladder"] = index[match]
            path.write_text(json.dumps(graph))
            annotated += 1
            logger.info("annotated %s with %s (%s)", path.name, match,
                        ", ".join(f"{e['arm']}={e['label']}" for e in index[match]["arms"]))
    logger.info("Done: %d graphs annotated, %d had no matching prompt id", annotated, skipped)


if __name__ == "__main__":
    main()
