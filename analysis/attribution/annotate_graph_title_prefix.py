# [implement: transcoder adapters larger feature collection]
"""Annotate comparison-overlay graphs with a dropdown `title_prefix` tag (no re-tracing).

The circuit-tracer dropdown renders each option as ``title_prefix + scan + ' — ' + prompt``
(see the frontend ``util.js``: ``var prefix = d.title_prefix ? d.title_prefix + ' ' : ''``).
``run_combined_attribution`` never sets ``title_prefix``, so multi-category overlays (e.g. the
interesting_queries set traced under a single ``--run_name``) show no per-graph tag and every
entry looks like ``base-vs-adapter — <prompt>``.

This tool derives a tag for each graph from the *source category* of its prompt — the immediate
subdirectory of ``--prompts_root`` that contains ``<stem>.txt`` — and writes
``title_prefix = "[<category> · <stem>]"`` into both ``graph-metadata.json`` (the dropdown index)
and each per-graph JSON's ``metadata``. It is idempotent: re-running (e.g. after adding prompts and
re-tracing) simply re-tags everything, so it is safe to call at the end of a trace pipeline.

The graph ``slug`` is ``<run_name>__<stem>__h<hash>``; the stem (which may contain single
underscores, never ``__``) is ``slug.split("__")[1]``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METADATA_NAME = "graph-metadata.json"
SKIP = {METADATA_NAME, "combined_attribution_manifest.json", "run_attribution_args.json"}


def _stem_to_category(prompts_root: Path) -> dict[str, str]:
    """Map each prompt filename stem -> its category (immediate subdir name, or '.' for root)."""
    mapping: dict[str, str] = {}
    for txt in prompts_root.rglob("*.txt"):
        rel = txt.relative_to(prompts_root)
        category = rel.parts[0] if len(rel.parts) > 1 else prompts_root.name
        mapping[txt.stem] = category
    return mapping


def _stem_from_slug(slug: str) -> str:
    parts = slug.split("__")
    return parts[1] if len(parts) >= 3 else slug


def annotate(graph_dir: Path, prompts_root: Path) -> int:
    stem_to_cat = _stem_to_category(prompts_root)
    if not stem_to_cat:
        raise ValueError(f"No .txt prompts found under {prompts_root}")

    def tag_for(slug: str) -> str:
        stem = _stem_from_slug(slug)
        category = stem_to_cat.get(stem, "?")
        return f"[{category} · {stem}]"

    # 1) per-graph JSON metadata (source of truth if the index is ever regenerated)
    tagged = 0
    for graph_path in sorted(graph_dir.glob("*.json")):
        if graph_path.name in SKIP:
            continue
        payload = json.loads(graph_path.read_text())
        meta = payload.get("metadata")
        if not isinstance(meta, dict) or "slug" not in meta:
            continue
        meta["title_prefix"] = tag_for(meta["slug"])
        graph_path.write_text(json.dumps(payload, indent=2) + "\n")
        tagged += 1

    # 2) the dropdown index the frontend actually reads
    index_path = graph_dir / METADATA_NAME
    if index_path.exists():
        index = json.loads(index_path.read_text())
        for entry in index.get("graphs", []):
            if "slug" in entry:
                entry["title_prefix"] = tag_for(entry["slug"])
        index_path.write_text(json.dumps(index, indent=2) + "\n")
    return tagged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graph_dir", required=True, type=Path, help="Overlay dir holding the graph JSONs + graph-metadata.json to tag.")
    parser.add_argument("--prompts_root", required=True, type=Path, help="Root of the prompt library whose immediate subdirs name the categories (e.g. analysis/attribution/prompts/interesting_queries).")
    args = parser.parse_args()
    n = annotate(args.graph_dir, args.prompts_root)
    print(f"[annotate_graph_title_prefix] tagged {n} graphs in {args.graph_dir} with categories from {args.prompts_root}")


if __name__ == "__main__":
    main()
