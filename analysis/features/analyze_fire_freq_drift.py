"""Analyze per-feature firing-frequency drift between two input distributions.

<!-- Created by Claude Code session "implement: new 07/02 gemmascope transcoder experiments". 2026-07-09. -->

Companion to ``transcoder_input_shift.py``. That script writes
``per_feature_fire_freq.npz`` with a per-(source, layer) vector of each transcoder feature's
firing frequency (fraction of tokens on which it is active). This script compares two sources
(default ``base`` vs ``instruct``) per layer and quantifies how much the *same* GemmaScope
features change their firing rate under the input-distribution shift -- the signal for whether
re-fine-tuning the transcoders is warranted, and which features would move most.

Per layer it reports:
  * fire-set Jaccard  -- overlap of the sets of ever-active features (near 1.0 for dense SAEs).
  * frequency Pearson r -- how correlated the per-feature firing rates are across the two
                           sources (drops as the input shift grows).
  * mean |Δ freq|, and the top-k features by |Δ freq| (the biggest movers).

Usage:
    uv run --no-sync python -m analysis.features.analyze_fire_freq_drift \
        --run_dir /nlp/scr/$USER/transcoder-adapters/transcoder_input_shift/<run> \
        --source_a base --source_b instruct --top_k 10

Writes ``fire_freq_drift.json`` and ``fire_freq_drift.md`` into ``--run_dir``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from helpers.log import logger, setup_logging


def _layers_for_source(keys: list[str], source: str) -> list[int]:
    pat = re.compile(rf"^{re.escape(source)}__layer_(\d+)$")
    return sorted(int(m.group(1)) for k in keys if (m := pat.match(k)))


def analyze(npz_path: Path, source_a: str, source_b: str, top_k: int) -> dict:
    data = np.load(npz_path)
    layers = _layers_for_source(list(data.keys()), source_a)
    if not layers:
        raise ValueError(f"No arrays for source {source_a!r} in {npz_path}")

    per_layer = {}
    for layer in layers:
        a = data[f"{source_a}__layer_{layer}"].astype(np.float64)
        b = data[f"{source_b}__layer_{layer}"].astype(np.float64)
        a_active, b_active = a > 0, b > 0
        union = (a_active | b_active).sum()
        jaccard = float((a_active & b_active).sum() / max(1, union))
        corr = float(np.corrcoef(a, b)[0, 1])
        d = b - a
        d_abs = np.abs(d)
        movers = np.argsort(-d_abs)[:top_k]
        per_layer[str(layer)] = {
            "n_features": int(a.size),
            "n_active_a": int(a_active.sum()),
            "n_active_b": int(b_active.sum()),
            "fire_set_jaccard": jaccard,
            "freq_pearson_r": corr,
            "mean_abs_delta_freq": float(d_abs.mean()),
            "top_movers": [
                {"feature": int(f), f"freq_{source_a}": float(a[f]), f"freq_{source_b}": float(b[f]), "delta": float(d[f])}
                for f in movers
            ],
        }
    return {"source_a": source_a, "source_b": source_b, "layers": layers, "per_layer": per_layer}


def write_markdown(result: dict, out_path: Path) -> None:
    a, b = result["source_a"], result["source_b"]
    lines = [
        f"# Feature firing-frequency drift — {a} vs {b}",
        "",
        f"How much do the same GemmaScope features change firing rate when the input hidden "
        f"states come from `{b}` instead of `{a}`? High Jaccard + falling Pearson r = same "
        f"features fire, but at drifted rates (input-shift signal).",
        "",
        "| layer | fire-set Jaccard | freq Pearson r | mean \\|Δfreq\\| | top mover (feat: a→b) |",
        "|---|---|---|---|---|",
    ]
    for layer in result["layers"]:
        m = result["per_layer"][str(layer)]
        tm = m["top_movers"][0]
        lines.append(
            f"| {layer} | {m['fire_set_jaccard']:.3f} | {m['freq_pearson_r']:.3f} | "
            f"{m['mean_abs_delta_freq']:.2e} | f{tm['feature']}: {tm[f'freq_{a}']:.3f}→{tm[f'freq_{b}']:.3f} |"
        )
    lines += ["", "Full per-layer top movers in `fire_freq_drift.json`.", ""]
    out_path.write_text("\n".join(lines))


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", type=Path, required=True, help="A transcoder_input_shift run dir containing per_feature_fire_freq.npz.")
    parser.add_argument("--source_a", default="base", help="Reference source (the distribution the transcoders were trained on).")
    parser.add_argument("--source_b", default="instruct", help="Shifted source to compare against.")
    parser.add_argument("--top_k", type=int, default=10, help="How many top-|Δ freq| features to record per layer.")
    args = parser.parse_args()

    npz_path = args.run_dir / "per_feature_fire_freq.npz"
    result = analyze(npz_path, args.source_a, args.source_b, args.top_k)

    json_path = args.run_dir / "fire_freq_drift.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    md_path = args.run_dir / "fire_freq_drift.md"
    write_markdown(result, md_path)

    for layer in result["layers"]:
        m = result["per_layer"][str(layer)]
        logger.info("L%-2d jaccard=%.3f r=%.3f mean|Δ|=%.2e", layer, m["fire_set_jaccard"], m["freq_pearson_r"], m["mean_abs_delta_freq"])
    logger.info("Wrote %s and %s", json_path, md_path)


if __name__ == "__main__":
    main()
