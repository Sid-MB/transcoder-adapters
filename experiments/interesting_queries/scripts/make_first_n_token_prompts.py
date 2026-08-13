#!/usr/bin/env python
"""Emit growing-prefix prompt files so run_base_adapter_comparison can trace the FIRST N refusal tokens.

Session "[refusal-token-tracing]" (fd1f0d19-f5d1-48cc-9002-3f06f2abe2fd), 2026-08-10.

The base-vs-adapter attribution overlay traces the LAST token of each prompt file. To build a graph at
each of the first N positions of the instruct model's refusal ("I", "cannot", "provide", ...), we greedily
generate that refusal once, then write N prompt files per request whose assistant prefix grows by one
token each: file pos{k} ends at refusal token k (so run_base_adapter_comparison traces exactly that token,
with tokens 0..k-1 as context). Files use the DeepSeek-marker authoring format the loader re-renders with
the real gemma chat template (see analysis.attribution.run_attribution._load_chat_formatted_prompt), so
pass ``--prompt_format chat`` to run_base_adapter_comparison -- identical to the existing *_strict_refusal
overlays, which trace only position 0 ("I").

Round-trip check: each assistant prefix is decoded from token ids then, as the loader will, re-encoded; if
re-encoding doesn't reproduce the intended target token the file is still written but a warning is logged
(rare for these plain-English openings, but it would silently trace the wrong token).

Outputs ``--output_dir``/<id>__pos{k:02d}.txt for k=0..N-1, plus openings.json (the harvested refusal per
id: text + token ids + the per-position target tokens) for the write-up.

Example:
    PYTHONPATH=. uv run --extra viz python experiments/interesting_queries/scripts/make_first_n_token_prompts.py \\
        --prompt_ids harm_125 harm_139 harm_116 --n_tokens 10 \\
        --output_dir analysis/attribution/prompts/refusal_first10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from helpers.log import logger, setup_logging  # noqa: E402
from analysis.attribution.run_attribution import (  # noqa: E402
    DEEPSEEK_ASSISTANT_TOKEN,
    DEEPSEEK_BOS_TOKEN,
    DEEPSEEK_USER_TOKEN,
)

INSTRUCT_MODEL = "google/gemma-2-2b-it"
STRICT_RESULTS = "experiments/interesting_queries/results/strict_compliance_refusal/results.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompt_ids", nargs="+", default=["harm_125", "harm_139", "harm_116"],
                   help="Prompt ids to trace. Read from --strict_results (selected records) unless --requests_json is given, in which case they are its keys.")
    p.add_argument("--n_tokens", type=int, default=10, help="How many leading response-token positions to emit files for (k=0..n_tokens-1).")
    p.add_argument("--source_model", default=INSTRUCT_MODEL, help="Model whose greedy completion defines the tokens to trace. Instruct (default) = the refusal we dissect; pass an adapter checkpoint with --source_is_adapter to trace the ADAPTER's own completion (e.g. its COMPLIANT opening, for a compliance-vs-refusal contrast).")
    p.add_argument("--source_is_adapter", action="store_true", help="Load --source_model via AutoModelForCausalLMWithTranscoder (a transcoder adapter) instead of a plain HF model.")
    p.add_argument("--tokenizer_model", default=INSTRUCT_MODEL, help="Tokenizer / chat template (adapters reuse the gemma-2-2b-it template they were trained with).")
    p.add_argument("--strict_results", default=STRICT_RESULTS, help="results.json from find_strict_compliance_refusal; supplies the request text per id when --requests_json is not set.")
    p.add_argument("--requests_json", default=None, help="Optional path to a JSON dict {id: request_text}. Use for prompts not in strict_results (e.g. benign controls or hand-picked jailbreaks). Overrides --strict_results as the request source.")
    p.add_argument("--skip_refusals", action="store_true", help="Skip (and log) any prompt whose greedy completion OPENS with a refusal ('I cannot'/'I can't'/'I'm sorry'/...). Use when harvesting COMPLIANT completions for the contrast, so refused prompts don't sneak in.")
    p.add_argument("--output_dir", required=True, type=Path, help="Directory to write <id>__pos{k}.txt prompt files + openings.json.")
    p.add_argument("--device", default="cuda", help="Torch device for the one greedy generation per prompt.")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Model dtype.")
    return p.parse_args()


_REFUSAL_OPENERS = ("i cannot", "i can't", "i can not", "i'm sorry", "i am sorry", "i will not", "i won't", "i'm unable", "i am unable", "i'm not able", "i apologize")


def _looks_like_refusal(text: str) -> bool:
    t = text.strip().lower()
    return any(t.startswith(m) for m in _REFUSAL_OPENERS)


def _dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def main() -> None:
    setup_logging()
    args = parse_args()
    logging.getLogger().setLevel(logging.INFO)
    logger.info("Invocation: make_first_n_token_prompts.py %s", " ".join(f"--{k} {v}" for k, v in vars(args).items()))

    if args.requests_json:
        reqs = json.loads(Path(args.requests_json).read_text())
        cats = {pid: "" for pid in reqs}
    else:
        data = json.loads(Path(args.strict_results).read_text())
        by_id = {r["id"]: r for r in data["records"] if r.get("selected")}
        reqs = {pid: by_id[pid]["prompt"] for pid in args.prompt_ids if pid in by_id}
        cats = {pid: by_id[pid].get("category", "") for pid in reqs}
        missing = [pid for pid in args.prompt_ids if pid not in by_id]
        if missing:
            raise SystemExit(f"prompt ids not among selected records in {args.strict_results}: {missing}")
    ids = args.prompt_ids if not args.requests_json else list(reqs.keys())

    tok = AutoTokenizer.from_pretrained(args.tokenizer_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if args.source_is_adapter:
        from models.auto import AutoModelForCausalLMWithTranscoder

        model = AutoModelForCausalLMWithTranscoder.from_pretrained(args.source_model, dtype=_dtype(args.dtype))
    else:
        model = AutoModelForCausalLM.from_pretrained(args.source_model, dtype=_dtype(args.dtype))
    model = model.to(args.device).eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    openings: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    for pid in ids:
        request = reqs[pid]
        prompt_ids = tok.apply_chat_template([{"role": "user", "content": request}], tokenize=True, add_generation_prompt=True)
        enc = torch.tensor([prompt_ids], device=args.device)
        gen = model.generate(enc, max_new_tokens=max(args.n_tokens + 4, 16), do_sample=False, use_cache=True, pad_token_id=tok.pad_token_id)
        resp_ids = gen[0][enc.shape[1]:].tolist()[: args.n_tokens]
        resp_text = tok.decode(resp_ids, skip_special_tokens=True)
        is_refusal = _looks_like_refusal(resp_text)
        logger.info("%s opening (%d tok)%s: %r", pid, len(resp_ids), " [REFUSAL]" if is_refusal else " [comply]", resp_text)
        if args.skip_refusals and is_refusal:
            skipped[pid] = resp_text
            logger.info("  -> skipped (%s opens with a refusal; --skip_refusals set)", pid)
            continue

        per_pos = []
        for k in range(len(resp_ids)):
            assistant_content = tok.decode(resp_ids[: k + 1], skip_special_tokens=True)
            # Mirror the loader: it re-encodes assistant_content (add_special_tokens=False) and traces the LAST token.
            re_ids = tok.encode(assistant_content, add_special_tokens=False)
            target_ok = bool(re_ids) and re_ids[-1] == resp_ids[k]
            if not target_ok:
                logger.warning("%s pos%02d: re-encode target %s != intended %s (assistant=%r) -- graph would trace a different token",
                               pid, k, re_ids[-1] if re_ids else None, resp_ids[k], assistant_content)
            text = f"{DEEPSEEK_BOS_TOKEN}{DEEPSEEK_USER_TOKEN}{request}{DEEPSEEK_ASSISTANT_TOKEN}{assistant_content}"
            (args.output_dir / f"{pid}__pos{k:02d}.txt").write_text(text)
            per_pos.append({"k": k, "target_token_id": resp_ids[k],
                            "target_token": tok.decode([resp_ids[k]]), "assistant_prefix": assistant_content,
                            "roundtrip_ok": target_ok})
        openings[pid] = {"request": request, "category": cats.get(pid, ""), "is_refusal": is_refusal,
                         "response_text": resp_text, "response_token_ids": resp_ids, "positions": per_pos}
    if skipped:
        openings["_skipped_refusals"] = skipped

    (args.output_dir / "openings.json").write_text(json.dumps(openings, indent=2) + "\n")
    n_files = sum(len(v["positions"]) for k, v in openings.items() if k != "_skipped_refusals")
    logger.info("Wrote %d prompt files (%d prompts kept, %d skipped) + openings.json to %s",
                n_files, len(openings) - (1 if skipped else 0), len(skipped), args.output_dir)


if __name__ == "__main__":
    main()
