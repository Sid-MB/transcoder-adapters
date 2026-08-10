# [08-10-26] (session 1fee7826-a308-4d81-b53c-9cf6bf8512ea)
"""Prefix a ★ (or any marker) onto specific overlay graphs' dropdown labels — no re-tracing.

The circuit-tracer comparison dropdown renders each option as ``title_prefix + scan + ' — ' +
prompt`` (frontend ``util.js``: ``var prefix = d.title_prefix ? d.title_prefix + ' ' : ''``), so
marking a graph is just a matter of writing its ``title_prefix``. ``annotate_graph_title_prefix``
already does this *by prompt category*; this tool is its per-graph counterpart: it marks an EXPLICIT
set of graphs (given by stem, e.g. ``harm_031``, or by full slug) so a chosen handful stands out in
the dropdown — e.g. the 5 strict-refusal prompts the huge adapter COMPLIES with rather than refuses
(see my_notes/08-10-26/huge_adapter_refusal_split.md).

The marker is PREPENDED to any existing ``title_prefix`` (so it composes with category tags from
``annotate_graph_title_prefix``) and the write is idempotent — re-running never stacks duplicate
markers. Applied to both each per-graph JSON's ``metadata`` (the source of truth if the index is
regenerated) and ``graph-metadata.json`` (what the dropdown actually reads). Re-run after any rebuild
of the overlay dir.

Example (star the 5 huge-adapter compliance prompts):
    uv run python -m analysis.attribution.star_graphs_in_dropdown \
        --graph_dir /nlp/scr/siddharth/transcoder-adapters/base_adapter_comparisons/overlay_huge_strict_refusal/overlay \
        --stems harm_031 harm_034 harm_035 harm_094 harm_187
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

METADATA_NAME = "graph-metadata.json"
SKIP = {METADATA_NAME, "combined_attribution_manifest.json", "run_attribution_args.json"}


def _stem_from_slug(slug: str) -> str:
    # slug == "<run_name>__<stem>__h<hash>"; stem may contain single underscores, never "__".
    parts = slug.split("__")
    return parts[1] if len(parts) >= 3 else slug


def _matches(slug: str, targets: set[str]) -> bool:
    return slug in targets or _stem_from_slug(slug) in targets


def _prefix_marker(existing: str | None, marker: str) -> str:
    existing = existing or ""
    if existing.startswith(marker):
        return existing  # idempotent: already starred
    return f"{marker} {existing}".rstrip() if existing else marker


def star(graph_dir: Path, targets: set[str], marker: str) -> tuple[int, set[str]]:
    """Prefix `marker` onto matching graphs. Returns (n_marked, stems_seen)."""
    seen: set[str] = set()

    marked = 0
    for graph_path in sorted(graph_dir.glob("*.json")):
        if graph_path.name in SKIP:
            continue
        payload = json.loads(graph_path.read_text())
        meta = payload.get("metadata")
        if not isinstance(meta, dict) or "slug" not in meta:
            continue
        seen.add(_stem_from_slug(meta["slug"]))
        if not _matches(meta["slug"], targets):
            continue
        meta["title_prefix"] = _prefix_marker(meta.get("title_prefix"), marker)
        graph_path.write_text(json.dumps(payload, indent=2) + "\n")
        marked += 1

    index_path = graph_dir / METADATA_NAME
    if index_path.exists():
        index = json.loads(index_path.read_text())
        for entry in index.get("graphs", []):
            if "slug" in entry and _matches(entry["slug"], targets):
                entry["title_prefix"] = _prefix_marker(entry.get("title_prefix"), marker)
        index_path.write_text(json.dumps(index, indent=2) + "\n")

    return marked, seen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graph_dir", required=True, type=Path, help="Overlay dir holding the graph JSONs + graph-metadata.json to mark.")
    parser.add_argument("--stems", nargs="+", required=True, help="Prompt stems (e.g. harm_031) or full slugs to mark. Matched against each graph's slug stem.")
    parser.add_argument("--marker", default="★", help="Marker to prepend to the dropdown label (default: ★). Any string works, e.g. '⚠' or '[COMPLY]'.")
    args = parser.parse_args()

    targets = set(args.stems)
    marked, seen = star(args.graph_dir, targets, args.marker)
    missing = {t for t in targets if t not in seen and t not in {_stem_from_slug(s) for s in seen}}
    print(f"[star_graphs_in_dropdown] marked {marked} graph(s) in {args.graph_dir} with {args.marker!r}")
    if missing:
        print(f"[star_graphs_in_dropdown] WARNING: {len(missing)} target(s) matched no graph: {sorted(missing)}")


if __name__ == "__main__":
    main()
