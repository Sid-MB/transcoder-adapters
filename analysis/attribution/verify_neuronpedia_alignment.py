"""Sanity-check that Neuronpedia descriptions apply to OUR transcoder features.

<!-- Created by Claude Code session "implement: new 07/02 gemmascope transcoder experiments". 2026-07-09. -->

Before trusting Neuronpedia's LLM feature descriptions (source ``{layer}-gemmascope-transcoder-16k``)
as labels for our GemmaScope transcoder features, we must confirm they describe the SAME feature —
i.e. Neuronpedia's per-layer transcoder is the same L0 variant as ours (``average_l0_76`` nearest).

A feature's decoder ``W_dec[i]`` defines the tokens it boosts (logit lens ``W_dec[i] @ W_U``). This
prints, side by side per sampled ``(layer, feature)``:
  * OURS       — our transcoder's logit-lens top tokens (computed from our loaded weights),
  * NEURONPEDIA — its ``pos_str`` (top boosted tokens) + the LLM ``description``,
and a token-overlap score. High overlap ⇒ same feature ⇒ descriptions are valid labels.

Usage:
    uv run --no-sync python -m analysis.attribution.verify_neuronpedia_alignment \
        --layers 0 24 25 --features 100 500 3000 --top_k 8
Requires $NEURONPEDIA_API_KEY (or ~/.shell/secrets/neuronpedia_api_key).
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from helpers.log import logger, setup_logging

from analysis.attribution.neuronpedia_descriptions import DEFAULT_SOURCE_TEMPLATE, _load_api_key
from analysis.attribution.run_base_adapter_comparison import (
    _load_gemmascope_transcoders,
    build_gemmascope_transcoder_config,
    resolve_gemmascope_l0_values,
)


def _neuronpedia_feature(model: str, source: str, index: int) -> dict:
    import neuronpedia
    from neuronpedia.np_sae_feature import SAEFeature

    with neuronpedia.api_key(_load_api_key()):
        data = SAEFeature.get(model, source, str(index)).jsonData
    return json.loads(data) if isinstance(data, str) else data


def _norm(tok: str) -> str:
    return tok.replace("▁", " ").strip().lower()


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gemma-2-2b")
    ap.add_argument("--base_model", default="google/gemma-2-2b")
    ap.add_argument("--source_template", default=DEFAULT_SOURCE_TEMPLATE)
    ap.add_argument("--gemmascope_repo", default="google/gemma-scope-2b-pt-transcoders")
    ap.add_argument("--gemmascope_width", default="width_16k")
    ap.add_argument("--gemmascope_l0", default="average_l0_76")
    ap.add_argument("--gemmascope_l0_match", default="nearest")
    ap.add_argument("--gemmascope_n_layers", type=int, default=26)
    ap.add_argument("--layers", nargs="+", type=int, default=[0, 24, 25])
    ap.add_argument("--features", nargs="+", type=int, default=[100, 500, 3000])
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--output", default=None, help="Optional JSON report path.")
    args = ap.parse_args()

    if not _load_api_key():
        raise SystemExit("No Neuronpedia API key ($NEURONPEDIA_API_KEY or ~/.shell/secrets/neuronpedia_api_key).")

    l0v = resolve_gemmascope_l0_values(repo=args.gemmascope_repo, width=args.gemmascope_width, l0=args.gemmascope_l0, n_layers=args.gemmascope_n_layers, l0_match=args.gemmascope_l0_match)
    cfg = build_gemmascope_transcoder_config(repo=args.gemmascope_repo, width=args.gemmascope_width, l0=args.gemmascope_l0, n_layers=args.gemmascope_n_layers, model_name=args.base_model, feature_input_hook="ln2.hook_normalized", feature_output_hook="hook_mlp_out", l0_values=l0v, l0_match=args.gemmascope_l0_match)
    ts = _load_gemmascope_transcoders(cfg, device=torch.device("cpu"), dtype=torch.float32)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base_model)
    logger.info("Loading %s for the unembedding (W_U)...", args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.float32)
    W_U = model.get_output_embeddings().weight.detach().float()  # [vocab, d_model]

    report = []
    n_ok = 0
    for layer in args.layers:
        source = args.source_template.format(layer=layer)
        W_dec = ts[layer].W_dec.float()
        for feat in args.features:
            ours = [tok.decode([t]) for t in torch.topk(W_dec[feat] @ W_U.T, args.top_k).indices.tolist()]
            np_data = _neuronpedia_feature(args.model, source, feat)
            np_pos = (np_data.get("pos_str") or [])[: args.top_k]
            exps = np_data.get("explanations") or []
            desc = (exps[0].get("description") or "").strip() if exps else ""
            overlap = len(set(map(_norm, ours)) & set(map(_norm, np_pos))) / max(1, len(ours))
            ok = overlap >= 0.5
            n_ok += ok
            report.append({"layer": layer, "feature": feat, "overlap": overlap, "aligned": ok, "ours_logit_lens": ours, "neuronpedia_pos_str": np_pos, "neuronpedia_description": desc})
            logger.info("L%d f%d  overlap=%.2f %s", layer, feat, overlap, "OK" if ok else "MISMATCH")
            logger.info("   ours logit-lens : %s", ours)
            logger.info("   neuronpedia pos : %s", np_pos)
            logger.info("   neuronpedia desc: %s", desc or "(none)")

    total = len(report)
    logger.info("Aligned %d/%d sampled features (overlap>=0.5).", n_ok, total)
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(json.dumps({"aligned": n_ok, "total": total, "samples": report}, indent=2) + "\n")
        logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
