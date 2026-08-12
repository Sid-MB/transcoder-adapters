# [08-10-26 semantics] (session 1fee7826-a308-4d81-b53c-9cf6bf8512ea)
"""Inspect the semantics of individual transcoder features stored in a Hub feature collection.

Given (layer, feature_index) pairs -- where `feature_index` is either the within-layer
index or the cantor-paired global index used by circuit-tracer graph nodes -- this prints
the feature's top/bottom logits, activation frequency, and its top-activating text
examples, which together are what reveal what the feature actually detects.

Only the requested byte slice of `features/layer_N.bin` is fetched (HTTP Range request),
so inspecting a handful of features costs kilobytes rather than the ~40MB/layer file.

Usage:
    uv run python -m analysis.features.inspect_features \
        --repo_id siddharthmb/2026.TA.features_... \
        --features 13:329252 16:1023148 21:1457756 \
        --n_examples 3
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import struct
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests
from huggingface_hub import hf_hub_download, hf_hub_url


def cantor_pair(layer: int, feature_idx: int) -> int:
    return (layer + feature_idx) * (layer + feature_idx + 1) // 2 + feature_idx


def cantor_unpair(z: int) -> tuple[int, int]:
    w = int(((8 * z + 1) ** 0.5 - 1) / 2)
    while (w + 1) * (w + 2) // 2 <= z:
        w += 1
    while w * (w + 1) // 2 > z:
        w -= 1
    y = z - w * (w + 1) // 2
    return w - y, y


def resolve_feature_index(layer: int, feature_index: int) -> int:
    """Return the within-layer index, accepting either a raw or cantor-paired index."""
    unpaired_layer, unpaired_idx = cantor_unpair(feature_index)
    if unpaired_layer == layer:
        return unpaired_idx
    return feature_index


@lru_cache(maxsize=None)
def _load_index(repo_id: str) -> dict[str, Any]:
    path = hf_hub_download(repo_id=repo_id, filename="features/index.json.gz", repo_type="model")
    with gzip.open(path, "rt") as f:
        return json.load(f)


def load_feature(repo_id: str, layer: int, feature_idx: int) -> dict[str, Any] | None:
    """Fetch one packed feature JSON via an HTTP Range request on the layer bin."""
    layer_entry = _load_index(repo_id)[str(layer)]
    offsets = layer_entry["offsets"]
    start, end = int(offsets[feature_idx]), int(offsets[feature_idx + 1])
    if end <= start:
        return None  # feature never fired during collection; nothing was packed
    url = hf_hub_url(repo_id=repo_id, filename=f"features/{layer_entry['filename']}", repo_type="model")
    headers = {"Range": f"bytes={start}-{end - 1}"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = requests.get(url, headers=headers, timeout=120)
    response.raise_for_status()
    raw = response.content
    compressed_len = struct.unpack("<I", raw[:4])[0]
    return json.loads(gzip.decompress(raw[4 : 4 + compressed_len]).decode("utf-8"))


def _format_example(example: dict[str, Any], max_chars: int) -> str:
    """Render one example as text with the peak-activating token wrapped in [[...]]."""
    tokens: list[str] = example.get("tokens") or []
    acts: list[float] = example.get("tokens_acts_list") or []
    peak = max(range(len(acts)), key=lambda i: acts[i]) if acts else None
    text = "".join(f"[[{tok}]]" if i == peak else tok for i, tok in enumerate(tokens))
    # Keep the peak token visible: window the text around it rather than truncating the head.
    if len(text) > max_chars and peak is not None:
        peak_char = text.index(f"[[{tokens[peak]}]]")
        start = max(0, peak_char - max_chars * 3 // 4)
        text = ("..." if start else "") + text[start : start + max_chars] + "..."
    domain = (example.get("source_metadata") or {}).get("domain")
    return f"(peak {example.get('peak_activation'):.2f}, {domain}) {text!r}"


def describe_feature(
    feature: dict[str, Any],
    *,
    n_examples: int,
    n_logits: int,
    max_chars: int,
    quantile_filter: str,
) -> str:
    lines = []
    freq = feature.get("activation_frequency")
    if freq is not None:
        lines.append(f"  activation_frequency: {freq}")
    for key in ("top_logits", "bottom_logits"):
        values = feature.get(key)
        if values:
            lines.append(f"  {key}: {values[:n_logits]}")
    for quantile in feature.get("examples_quantiles") or []:
        label = quantile.get("quantile_name", "top")
        if quantile_filter and quantile_filter not in label:
            continue
        examples = quantile.get("examples", [])
        lines.append(f"  examples [{label}] ({len(examples)} stored):")
        for example in examples[:n_examples]:
            lines.append(f"    - {_format_example(example, max_chars)}")
    return "\n".join(lines)


def parse_feature_arg(value: str) -> tuple[int, int]:
    layer, _, feature = value.partition(":")
    if not feature:
        raise argparse.ArgumentTypeError(f"Expected LAYER:FEATURE, got {value!r}")
    return int(layer), int(feature)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print top logits and top-activating examples for specific transcoder features.",
    )
    parser.add_argument(
        "--repo_id",
        required=True,
        help="Hugging Face repo holding the packed feature collection (features/index.json.gz + layer_N.bin).",
    )
    parser.add_argument(
        "--features",
        required=True,
        nargs="+",
        type=parse_feature_arg,
        metavar="LAYER:FEATURE",
        help=(
            "Features to inspect as LAYER:FEATURE. FEATURE may be the within-layer index or the "
            "cantor-paired global index printed on circuit-tracer graph nodes; the two are "
            "distinguished automatically."
        ),
    )
    parser.add_argument(
        "--n_examples",
        type=int,
        default=3,
        help="How many top-activating text examples to print per feature. Raise to judge a feature whose top few examples look ambiguous.",
    )
    parser.add_argument(
        "--n_logits",
        type=int,
        default=10,
        help="How many top/bottom promoted tokens to print per feature.",
    )
    parser.add_argument(
        "--max_chars",
        type=int,
        default=400,
        help="Truncate each printed example to this many characters. Raise when the peak token's context matters.",
    )
    parser.add_argument(
        "--quantile_filter",
        default="Top activations",
        help=(
            "Only print example buckets whose name contains this string. Stored buckets are "
            "'Top activations', 'Top activations (chat)', 'Top activations (fineweb)' and "
            "'Random samples'; pass '' to print all, or e.g. '(chat)' for chat-domain examples only."
        ),
    )
    parser.add_argument(
        "--json_out",
        type=Path,
        default=None,
        help="Optional path to dump the raw fetched feature JSONs, for deeper offline inspection.",
    )
    args = parser.parse_args()

    fetched: dict[str, Any] = {}
    for layer, raw_feature in args.features:
        feature_idx = resolve_feature_index(layer, raw_feature)
        print(f"\n=== layer {layer} feature {raw_feature} (within-layer index {feature_idx}) ===")
        feature = load_feature(args.repo_id, layer, feature_idx)
        if feature is None:
            print("  <no stored data: feature never activated during collection>")
            continue
        fetched[f"{layer}:{raw_feature}"] = feature
        print(describe_feature(
            feature,
            n_examples=args.n_examples,
            n_logits=args.n_logits,
            max_chars=args.max_chars,
            quantile_filter=args.quantile_filter,
        ))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(fetched, indent=2))
        print(f"\nWrote raw feature JSONs to {args.json_out}")


if __name__ == "__main__":
    main()
