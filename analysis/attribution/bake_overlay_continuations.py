"""Bake base + adapter greedy continuations into base-vs-adapter overlay graphs.

For each overlay graph JSON, greedily decode the base model's and the adapter model's own
continuation of the prompt (from ``metadata.prompt_tokens``) and store them under
``metadata.comparison.{base_continuation,adapter_continuation}`` so the viewer can show a
base-vs-adapter completion panel below the graph. Reused by ``run_base_adapter_comparison``
(automatic at the end of every run) and runnable standalone to backfill already-built overlays.

Base continuation = plain base model (``--base_model``, e.g. google/gemma-2-2b). Adapter
continuation = the real transcoder-adapter model (``--adapter_checkpoint``), i.e. the instruct-
bridging behavior. Both greedy/argmax == the top-logit continuation.

Standalone (backfill existing overlays):
    uv run --extra viz python -m analysis.attribution.bake_overlay_continuations \
        --overlay_dir <dir>/overlay --compact_dir <dir>/overlay_compact \
        --base_model google/gemma-2-2b \
        --adapter_checkpoint siddharthmb/2026.TA.gemma2_2b_huge_... \
        --max_new_tokens 200
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any

from helpers.log import logger, setup_logging

DEFAULT_CONTINUATION_MAX_NEW_TOKENS = 200


def _patch_hf_model_info() -> None:
    """transformers' tokenizer load calls huggingface_hub.model_info unconditionally (via a nested
    is_base_mistral) -- a network call that ignores local_files_only and 504s during HF outages.
    Stub it to a tagless result so cached loads work offline; gemma is not mistral."""
    import huggingface_hub as _hh

    class _StubModelInfo:
        tags = None

    _hh.model_info = lambda *a, **k: _StubModelInfo()


def _greedy(logits_fn, ids, tokenizer, max_new_tokens: int) -> dict[str, Any]:
    import torch

    prompt_len = ids.shape[1]
    stopped = False
    with torch.no_grad():
        for _ in range(max_new_tokens):
            nxt = logits_fn(ids)[0, -1].argmax().view(1, 1)
            ids = torch.cat([ids, nxt.to(ids.device)], dim=1)
            if nxt.item() == tokenizer.eos_token_id:
                stopped = True
                break
    text = tokenizer.decode(ids[0, prompt_len:], skip_special_tokens=False)
    return {
        "text": text,
        "n_tokens": int(ids.shape[1] - prompt_len),
        "stopped_at_eos": stopped,
        "max_new_tokens": max_new_tokens,
        "decoding": "greedy",
    }


def bake_continuations(
    *,
    overlay_dir: Path,
    compact_dirs: list[Path] | None = None,
    base_model: str,
    adapter_checkpoint: str,
    max_new_tokens: int = DEFAULT_CONTINUATION_MAX_NEW_TOKENS,
    device: str = "cuda",
) -> int:
    """Generate + inject base/adapter continuations for every overlay graph. Returns count baked.

    Best-effort: on any model-load or generation error, logs a warning and returns 0 (overlays are
    still valid without continuations). Compact-dir graphs reuse the overlay's continuations matched
    by shared ``comparison.base_slug``, so the models run once."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from analysis.attribution.relp_model import RelPReplacementModel

    if max_new_tokens <= 0:
        logger.info("Continuation baking disabled (max_new_tokens<=0)")
        return 0

    overlay_files = [
        p for p in sorted(glob.glob(str(overlay_dir / "*.json")))
        if os.path.basename(p) not in {"graph-metadata.json", "run_attribution_args.json"}
    ]
    if not overlay_files:
        logger.warning("No overlay graphs to bake continuations for in %s", overlay_dir)
        return 0

    _patch_hf_model_info()
    dtype = torch.bfloat16
    try:
        tok = AutoTokenizer.from_pretrained(base_model)
        base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype, device_map=device)
        base.eval()
        adapter = RelPReplacementModel.from_pretrained(adapter_checkpoint, device=device, dtype=dtype)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Continuation baking skipped -- model load failed (%s: %s)", type(exc).__name__, exc)
        return 0

    def base_logits(ids):
        return base(ids).logits

    def adapter_logits(ids):
        out = adapter(ids)
        return out.logits if hasattr(out, "logits") else out

    by_base_slug: dict[str, dict[str, Any]] = {}
    baked = 0
    for f in overlay_files:
        g = json.load(open(f))
        m = g.get("metadata", {})
        comp = m.setdefault("comparison", {})
        prompt_tokens = m.get("prompt_tokens")
        if not prompt_tokens:
            continue
        ids0 = torch.tensor([prompt_tokens], device=device)
        try:
            bc = _greedy(base_logits, ids0, tok, max_new_tokens)
            ac = _greedy(adapter_logits, ids0, tok, max_new_tokens)
        except Exception as exc:  # noqa: BLE001
            logger.warning("  continuation gen failed for %s (%s)", os.path.basename(f), exc)
            continue
        comp["base_continuation"] = bc
        comp["adapter_continuation"] = ac
        json.dump(g, open(f, "w"))
        by_base_slug[comp.get("base_slug", m.get("slug"))] = {"base_continuation": bc, "adapter_continuation": ac}
        baked += 1
        logger.info("  baked continuations: %s", os.path.basename(f))

    for cdir in compact_dirs or []:
        for f in sorted(glob.glob(str(cdir / "*.json"))):
            if os.path.basename(f) in {"graph-metadata.json", "run_attribution_args.json"}:
                continue
            g = json.load(open(f))
            comp = g.get("metadata", {}).setdefault("comparison", {})
            conts = by_base_slug.get(comp.get("base_slug"))
            if conts:
                comp["base_continuation"] = conts["base_continuation"]
                comp["adapter_continuation"] = conts["adapter_continuation"]
                json.dump(g, open(f, "w"))

    del base, adapter
    torch.cuda.empty_cache()
    logger.info("Baked continuations into %d overlay graphs (+ compact variants)", baked)
    return baked


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--overlay_dir", type=Path, required=True, help="Dir of overlay graph JSONs.")
    ap.add_argument("--compact_dir", type=Path, nargs="*", default=None, help="Optional compact dir(s) to also fill.")
    ap.add_argument("--base_model", default="google/gemma-2-2b", help="Base model for the base-side continuation.")
    ap.add_argument("--adapter_checkpoint", required=True, help="HF repo or path of the transcoder-adapter model.")
    ap.add_argument("--max_new_tokens", type=int, default=DEFAULT_CONTINUATION_MAX_NEW_TOKENS, help="Greedy tokens per side (<=0 disables).")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    n = bake_continuations(
        overlay_dir=args.overlay_dir,
        compact_dirs=list(args.compact_dir) if args.compact_dir else None,
        base_model=args.base_model,
        adapter_checkpoint=args.adapter_checkpoint,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
    logger.info("Done: baked %d.", n)


if __name__ == "__main__":
    main()
