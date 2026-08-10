"""Measure how much a model's behavior depends on the prompt TEMPLATE, not the prompt content.

[08-10-26] (session 1fee7826-a308-4d81-b53c-9cf6bf8512ea)

Motivation: base-vs-adapter refusal overlays render every prompt through the gemma chat
template (``<start_of_turn>user ... <start_of_turn>model``). That is in-distribution for a
transcoder adapter (trained on lmsys via ``apply_chat_template``, see
``training/dataset/gemma2/lmsys_chat.py``) but OUT-of-distribution for base gemma-2-2b, which
never saw those markers in that role -- it degenerates into echoing the prompt and looping on
pretraining boilerplate. So "base does not refuse" is confounded with "base cannot parse the
prompt", and the two are indistinguishable in the current graphs.

This probe separates them: for each (model x prompt_format) cell it records

  * the greedy continuation (is it coherent, or looping/echoing?),
  * p(target_token) -- for refusal prompts the target is "I", so this is a cheap refusal proxy,
  * the top-k next-token distribution and its entropy (flat == confused/OOD),
  * a ``looping`` flag from n-gram repetition in the continuation.

Generation only -- no attribution -- so it is minutes, not hours, and is meant to be run BEFORE
committing GPU to a full retrace. Every continuation is saved to the output JSON so specific
examples can be revisited later.

Example:
    uv run --extra viz python -m analysis.evals.template_sensitivity_probe \
        --prompts <dir-of-marked-prompt-txt-files> \
        --base_model google/gemma-2-2b \
        --adapter_checkpoint siddharthmb/2026.TA.gemma2_2b_huge_... \
        --instruct_model google/gemma-2-2b-it \
        --prompt_formats chat plain --max_prompts 12 --max_new_tokens 120 \
        --output_json <dir>/template_probe.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from helpers.log import logger, setup_logging


def _looping(text: str, n: int = 5, threshold: int = 3) -> bool:
    """True if any n-gram of words repeats >= threshold times (degenerate repetition)."""
    words = text.split()
    if len(words) < n * threshold:
        return False
    grams = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    return max(grams.values()) >= threshold


def _entropy(probs) -> float:
    import torch

    p = probs[probs > 0]
    return float(-(p * torch.log(p)).sum())


def _greedy_with_stats(logits_fn, ids, tokenizer, *, target_token: int, max_new_tokens: int, top_k: int) -> dict[str, Any]:
    """Greedy-decode, and report the FIRST-step distribution (where the target token lives)."""
    import torch

    prompt_len = ids.shape[1]
    first_probs = None
    stopped = False
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = logits_fn(ids)[0, -1].float()
            if first_probs is None:
                first_probs = torch.softmax(logits, dim=-1)
            nxt = logits.argmax().view(1, 1)
            ids = torch.cat([ids, nxt.to(ids.device)], dim=1)
            if nxt.item() == tokenizer.eos_token_id:
                stopped = True
                break

    text = tokenizer.decode(ids[0, prompt_len:], skip_special_tokens=False)
    top_p, top_i = first_probs.topk(top_k)
    return {
        "continuation": text,
        "n_tokens": int(ids.shape[1] - prompt_len),
        "stopped_at_eos": stopped,
        "looping": _looping(text),
        "target_token": tokenizer.decode([target_token]),
        "target_prob": float(first_probs[target_token]),
        "target_rank": int((first_probs > first_probs[target_token]).sum()) + 1,
        "top_k": [
            {"token": tokenizer.decode([int(i)]), "prob": round(float(p), 4)}
            for p, i in zip(top_p.tolist(), top_i.tolist())
        ],
        "entropy": round(_entropy(first_probs), 3),
    }


def _load_model(spec: str, kind: str, device: str, dtype):
    from transformers import AutoModelForCausalLM

    if kind == "adapter":
        from analysis.attribution.relp_model import RelPReplacementModel

        model = RelPReplacementModel.from_pretrained(spec, device=device, dtype=dtype)

        def logits_fn(ids):
            out = model(ids)
            return out.logits if hasattr(out, "logits") else out

        return model, logits_fn

    model = AutoModelForCausalLM.from_pretrained(spec, torch_dtype=dtype, device_map=device)
    model.eval()
    return model, (lambda ids: model(ids).logits)


def run_probe(
    *,
    prompts_dir: str,
    models: list[tuple[str, str, str]],
    prompt_formats: list[str],
    tokenizer_model: str,
    max_prompts: int,
    max_new_tokens: int,
    top_k: int,
    device: str,
) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    from analysis.attribution.run_attribution import load_prompt_file, _list_prompt_files

    dtype = torch.bfloat16
    # One tokenizer for ALL cells: gemma-2-2b and -it share a vocabulary, and only the -it
    # tokenizer carries a chat_template, so using it keeps token ids comparable across models
    # while still allowing prompt_format="chat" to render.
    tok = AutoTokenizer.from_pretrained(tokenizer_model)

    files = _list_prompt_files(prompts_dir)[:max_prompts]
    logger.info("Probing %d prompts x %d formats x %d models", len(files), len(prompt_formats), len(models))

    rendered: dict[str, dict[str, Any]] = {}
    for fmt in prompt_formats:
        for f in files:
            toks, target, text = load_prompt_file(f, tok, prompt_format=fmt, model_type="gemma2")
            rendered[f"{fmt}|{f.stem}"] = {"tokens": toks, "target": target, "prompt": text}

    results: list[dict[str, Any]] = []
    for name, spec, kind in models:
        logger.info("Loading %s (%s): %s", name, kind, spec)
        model, logits_fn = _load_model(spec, kind, device, dtype)
        for key, r in rendered.items():
            fmt, slug = key.split("|", 1)
            ids = torch.tensor([r["tokens"]], device=device)
            stats = _greedy_with_stats(
                logits_fn, ids, tok, target_token=r["target"], max_new_tokens=max_new_tokens, top_k=top_k
            )
            results.append({"model": name, "prompt_format": fmt, "slug": slug, "prompt": r["prompt"], **stats})
            logger.info(
                "  %-9s %-6s %-12s p(target)=%.3f rank=%-4d entropy=%.2f loop=%s",
                name, fmt, slug, stats["target_prob"], stats["target_rank"], stats["entropy"], stats["looping"],
            )
        del model
        torch.cuda.empty_cache()

    return {
        "config": {
            "prompts_dir": prompts_dir,
            "models": [{"name": n, "spec": s, "kind": k} for n, s, k in models],
            "prompt_formats": prompt_formats,
            "tokenizer_model": tokenizer_model,
            "max_prompts": max_prompts,
            "max_new_tokens": max_new_tokens,
        },
        "results": results,
        "summary": summarize(results),
    }


def summarize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per (model, prompt_format) cell -- the table you actually read."""
    cells: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in results:
        cells.setdefault((r["model"], r["prompt_format"]), []).append(r)
    out = []
    for (model, fmt), rs in sorted(cells.items()):
        n = len(rs)
        out.append({
            "model": model,
            "prompt_format": fmt,
            "n_prompts": n,
            "mean_target_prob": round(sum(r["target_prob"] for r in rs) / n, 4),
            "frac_target_is_argmax": round(sum(r["target_rank"] == 1 for r in rs) / n, 3),
            "mean_entropy": round(sum(r["entropy"] for r in rs) / n, 3),
            "frac_looping": round(sum(r["looping"] for r in rs) / n, 3),
            "frac_stopped_at_eos": round(sum(r["stopped_at_eos"] for r in rs) / n, 3),
        })
    return out


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", required=True, help="Directory of marked prompt .txt files (DeepSeek/Qwen markers; last assistant token = target).")
    ap.add_argument("--base_model", default="google/gemma-2-2b", help="Plain base model -- the side currently producing OOD gibberish under the chat template.")
    ap.add_argument("--adapter_checkpoint", default=None, help="Transcoder-adapter checkpoint (HF repo or path). The instruct-bridged side.")
    ap.add_argument("--instruct_model", default=None, help="Optional real instruct model (e.g. google/gemma-2-2b-it) as the upper-bound reference column.")
    ap.add_argument("--prompt_formats", nargs="+", default=["chat", "plain"], choices=["chat", "plain", "raw"], help="Template renderings to compare. 'chat' = gemma <start_of_turn> template; 'plain' = neutral 'User:/Assistant:' dialogue.")
    ap.add_argument("--tokenizer_model", default="google/gemma-2-2b-it", help="Tokenizer used for ALL cells (must have a chat_template for --prompt_formats chat). gemma base/-it share a vocab.")
    ap.add_argument("--max_prompts", type=int, default=12, help="Prompts to probe. Keep small -- this is a fast pre-check, not the full run.")
    ap.add_argument("--max_new_tokens", type=int, default=120, help="Greedy tokens per cell. Enough to see whether output is coherent or looping.")
    ap.add_argument("--top_k", type=int, default=8, help="Next-token candidates to record at the target position.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output_json", type=Path, required=True, help="Where to write full results INCLUDING every continuation (kept for later inspection).")
    args = ap.parse_args()

    models = [("base", args.base_model, "base")]
    if args.adapter_checkpoint:
        models.append(("adapter", args.adapter_checkpoint, "adapter"))
    if args.instruct_model:
        models.append(("instruct", args.instruct_model, "base"))

    payload = run_probe(
        prompts_dir=args.prompts,
        models=models,
        prompt_formats=args.prompt_formats,
        tokenizer_model=args.tokenizer_model,
        max_prompts=args.max_prompts,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        device=args.device,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2))

    logger.info("\n%-10s %-7s %6s %8s %9s %8s %8s", "MODEL", "FORMAT", "N", "p(tgt)", "argmax%", "entropy", "loop%")
    for s in payload["summary"]:
        logger.info(
            "%-10s %-7s %6d %8.3f %9.2f %8.2f %8.2f",
            s["model"], s["prompt_format"], s["n_prompts"], s["mean_target_prob"],
            s["frac_target_is_argmax"], s["mean_entropy"], s["frac_looping"],
        )
    logger.info("Wrote %s", args.output_json)


if __name__ == "__main__":
    main()
