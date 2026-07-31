"""Splice per-layer feature examples from one packed collection into another.

<!-- Created by Claude Code session "implement: new 07/02 gemmascope transcoder experiments". 2026-07-09. -->

The re-fine-tune changed only a few transcoder layers, so to build a "fine-tuned" feature-example
set we keep the high-quality base collection for the 23 unchanged layers and splice in freshly
collected examples for the changed layers only. Because the packed circuit-tracer format stores
each layer as a self-contained ``layer_N.bin`` plus a per-layer entry in ``index.json.gz``
(``{filename, offsets}`` with offsets local to that file), swapping a layer = copy its ``.bin``
and replace its index entry. This keeps the two served sides identical everywhere except the
spliced layers, so the visual diff is exactly the fine-tuned change.

Usage:
    uv run --no-sync python -m analysis.features.splice_feature_collections \
        --base_dir  <base_ms20000>/circuit_tracer_features \
        --patch_dir <finetuned_collection>/circuit_tracer_features \
        --layers 0 24 25 \
        --output_dir <out>/circuit_tracer_features
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path

from helpers.log import logger, setup_logging


def _load_index(d: Path) -> dict:
    with gzip.open(d / "index.json.gz", "rt") as f:
        return json.load(f)


def _save_index(d: Path, idx: dict) -> None:
    with gzip.open(d / "index.json.gz", "wt") as f:
        json.dump(idx, f)


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_dir", type=Path, required=True, help="Packed circuit_tracer_features dir kept for the unchanged layers.")
    ap.add_argument("--patch_dir", type=Path, required=True, help="Packed circuit_tracer_features dir whose --layers replace the base.")
    ap.add_argument("--layers", nargs="+", type=int, required=True, help="Layers to take from --patch_dir (e.g. 0 24 25).")
    ap.add_argument("--output_dir", type=Path, required=True, help="Output packed dir to create.")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_idx = _load_index(args.base_dir)
    patch_idx = _load_index(args.patch_dir)

    # Copy every base layer bin, then overwrite the spliced layers from patch.
    for bin_path in sorted(args.base_dir.glob("layer_*.bin")):
        shutil.copy2(bin_path, args.output_dir / bin_path.name)
    out_idx = dict(base_idx)
    for layer in args.layers:
        key = str(layer)
        if key not in patch_idx:
            raise SystemExit(f"Layer {layer} missing from patch index {args.patch_dir}")
        fname = patch_idx[key]["filename"]
        shutil.copy2(args.patch_dir / fname, args.output_dir / fname)
        out_idx[key] = patch_idx[key]
        logger.info("Spliced layer %d from %s (%s)", layer, args.patch_dir, fname)
    _save_index(args.output_dir, out_idx)
    logger.info("Wrote spliced collection: %s (base=%s, patched layers=%s)", args.output_dir, args.base_dir, args.layers)


if __name__ == "__main__":
    main()
