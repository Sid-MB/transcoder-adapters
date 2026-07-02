# Created by Claude Code session "explore: queries to analyze".
"""Shared machinery for the three interesting-query finders.

Provides:
  - load_models: load base (google/gemma-2-2b) + instruct (google/gemma-2-2b-it) in bf16.
  - render_chat_ids: render a single-turn user prompt with the gemma-2-2b-it chat template
    (add_generation_prompt=True) plus an optional assistant prefix, so the model predicts the
    token AFTER the prefix. Mirrors experiments/base_vs_adapter_circuit_trace/scripts/measure_agreement.py.
  - kl_last_position / kl_over_response: KL(instruct || base) over full-vocab softmax (nats),
    the same divergence metric measure_agreement uses.
  - detect_refusal: greedy-generate ~40 tokens from a model and classify refusal via regex markers,
    AND report P(first-token in {Sure, Here, Certainly, Of, To}) vs the refusal-token ("I") prob.
  - first_response_target: pick the target token = argmax first assistant-response token, and report
    its index in the chat-rendered sequence.
  - write_prompt_txt: emit a DeepSeek-marker .txt prompt consumed by run_combined_attribution
    (analysis/attribution/run_attribution.load_prompt_file, --prompt_format chat).

We import the repo's chat-template normaliser so tokenisation matches the rest of the pipeline.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.tokens import _input_ids_from_chat_template_output

BASE_MODEL = "google/gemma-2-2b"
INSTRUCT_MODEL = "google/gemma-2-2b-it"

# DeepSeek chat markers used by the .txt prompt files (run_attribution re-renders them with the
# actual model's chat template when --prompt_format chat is passed).
DEEPSEEK_BOS_TOKEN = "<｜begin▁of▁sentence｜>"
DEEPSEEK_USER_TOKEN = "<｜User｜>"
DEEPSEEK_ASSISTANT_TOKEN = "<｜Assistant｜>"

# Refusal regex markers (case-sensitive-ish; we search the generated continuation).
REFUSAL_MARKERS = re.compile(
    r"I can't|I cannot|I'm sorry|I am sorry|I'm unable|I won't|I am not able|As an AI|"
    r"I must decline|cannot assist|can't help|I'm not able|I do not|I don't",
)

# Affirmative first-token words (compliance signature). We match with and without a leading space.
AFFIRMATIVE_WORDS = ["Sure", "Here", "Certainly", "Of", "To", "Yes", "Absolutely", "I'd"]


def load_models(device: str = "cuda", base_model: str = BASE_MODEL, instruct_model: str = INSTRUCT_MODEL):
    """Load instruct tokenizer + both models in bf16 on `device`, eval mode."""
    tokenizer = AutoTokenizer.from_pretrained(instruct_model)

    def _load(name):
        # `dtype=` is the current transformers kwarg; older transformers (pulled in by nanoGCG's
        # dependency pins) only accept `torch_dtype=`. Support both so the finders run under either.
        try:
            return AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(device).eval()
        except TypeError:
            return AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16).to(device).eval()

    instruct = _load(instruct_model)
    base = _load(base_model)
    return tokenizer, base, instruct


def render_chat_ids(tokenizer, user_text: str, assistant_prefix: str = "") -> list[int]:
    """Render user turn with the IT chat template (add_generation_prompt=True) + optional prefix."""
    ids = _input_ids_from_chat_template_output(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    ids = list(ids)
    if assistant_prefix:
        ids = ids + tokenizer.encode(assistant_prefix, add_special_tokens=False)
    return ids


@torch.no_grad()
def kl_last_position(base, instruct, input_ids: torch.Tensor) -> dict:
    """KL(instruct || base) at the final position + top1 ids. input_ids: (1, T)."""
    base_logits = base(input_ids).logits[0, -1].float()
    instruct_logits = instruct(input_ids).logits[0, -1].float()
    base_lp = F.log_softmax(base_logits, dim=-1)
    instruct_lp = F.log_softmax(instruct_logits, dim=-1)
    instruct_p = instruct_lp.exp()
    kl = float((instruct_p * (instruct_lp - base_lp)).sum().item())
    return {
        "kl": kl,
        "base_top1_id": int(base_logits.argmax().item()),
        "instruct_top1_id": int(instruct_logits.argmax().item()),
    }


@torch.no_grad()
def kl_over_response(base, instruct, tokenizer, prompt_ids: list[int], n_positions: int = 8, device: str = "cuda") -> dict:
    """Greedy-generate up to n_positions response tokens with the INSTRUCT model, then teacher-force
    BOTH models over prompt+response and compute KL(instruct||base) at each response position.

    Returns the per-position KLs plus the argmax position, its KL, and the base/instruct top1 tokens
    decoded THERE (the divergent decision token).
    """
    prompt_t = torch.tensor([prompt_ids], device=device)
    gen = instruct.generate(prompt_t, max_new_tokens=n_positions, do_sample=False, use_cache=True)
    resp_ids = gen[0, len(prompt_ids):].tolist()
    if not resp_ids:
        return {"max_kl": 0.0, "argmax_pos": 0, "per_position_kl": [], "resp_ids": []}
    full_ids = prompt_ids + resp_ids
    full_t = torch.tensor([full_ids], device=device)
    base_logits = base(full_t).logits[0].float()
    instruct_logits = instruct(full_t).logits[0].float()
    per_pos = []
    # Position predicting response token j is at index (len(prompt_ids) - 1 + j) for j in 0..len(resp)-1.
    for j in range(len(resp_ids)):
        idx = len(prompt_ids) - 1 + j
        b_lp = F.log_softmax(base_logits[idx], dim=-1)
        i_lp = F.log_softmax(instruct_logits[idx], dim=-1)
        i_p = i_lp.exp()
        kl = float((i_p * (i_lp - b_lp)).sum().item())
        per_pos.append({
            "j": j,
            "seq_index": idx,
            "kl": kl,
            "base_top1_id": int(base_logits[idx].argmax().item()),
            "instruct_top1_id": int(instruct_logits[idx].argmax().item()),
            "base_top1_token": tokenizer.decode([int(base_logits[idx].argmax().item())]),
            "instruct_top1_token": tokenizer.decode([int(instruct_logits[idx].argmax().item())]),
        })
    best = max(per_pos, key=lambda p: p["kl"])
    return {
        "max_kl": best["kl"],
        "argmax_pos": best["j"],
        "argmax_seq_index": best["seq_index"],
        "argmax_base_top1_token": best["base_top1_token"],
        "argmax_instruct_top1_token": best["instruct_top1_token"],
        "per_position_kl": per_pos,
        "resp_ids": resp_ids,
        "resp_text": tokenizer.decode(resp_ids),
    }


def _first_token_prob(logits: torch.Tensor, tokenizer, words: list[str]) -> float:
    """Sum of softmax prob mass on the first-token ids of `words` (with/without leading space)."""
    probs = F.softmax(logits, dim=-1)
    total = 0.0
    seen = set()
    for w in words:
        for variant in (w, " " + w):
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if not ids:
                continue
            tid = ids[0]
            if tid in seen:
                continue
            seen.add(tid)
            total += float(probs[tid].item())
    return total


@torch.no_grad()
def detect_refusal(model, tokenizer, user_text: str, max_new_tokens: int = 40, device: str = "cuda") -> dict:
    """Greedy-generate a continuation and classify refusal.

    Returns:
      is_refusal: regex markers present in the generated continuation.
      p_affirmative: P(first token in the compliance set {Sure, Here, ...}).
      p_refusal_first: P(first token == 'I') (the refusal opener).
      first_token, generated_text.
    """
    prompt_ids = render_chat_ids(tokenizer, user_text)
    prompt_t = torch.tensor([prompt_ids], device=device)
    out = model(prompt_t)
    first_logits = out.logits[0, -1].float()
    p_affirmative = _first_token_prob(first_logits, tokenizer, AFFIRMATIVE_WORDS)
    p_refusal_first = _first_token_prob(first_logits, tokenizer, ["I"])
    first_token_id = int(first_logits.argmax().item())
    gen = model.generate(prompt_t, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    gen_text = tokenizer.decode(gen[0, len(prompt_ids):], skip_special_tokens=True)
    return {
        "is_refusal": bool(REFUSAL_MARKERS.search(gen_text)),
        "p_affirmative": p_affirmative,
        "p_refusal_first": p_refusal_first,
        "first_token": tokenizer.decode([first_token_id]),
        "first_token_id": first_token_id,
        "generated_text": gen_text,
    }


@torch.no_grad()
def first_response_target(model, tokenizer, prompt_ids: list[int], device: str = "cuda") -> dict:
    """Target = argmax first assistant-response token. Report its index in the rendered sequence
    (= len(prompt_ids): it is predicted at position len(prompt_ids)-1 and becomes token index len(prompt_ids))."""
    prompt_t = torch.tensor([prompt_ids], device=device)
    logits = model(prompt_t).logits[0, -1].float()
    tid = int(logits.argmax().item())
    return {"target_token_id": tid, "target_token": tokenizer.decode([tid]), "target_index": len(prompt_ids)}


def write_prompt_txt(path: Path, user_text: str, assistant_prefix: str, target_token_str: str) -> None:
    """Write a DeepSeek-marker chat prompt consumed by run_combined_attribution (--prompt_format chat).
    The assistant content's LAST token is the held-out target; prefix (if any) is context."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = f"{DEEPSEEK_BOS_TOKEN}{DEEPSEEK_USER_TOKEN}{user_text}{DEEPSEEK_ASSISTANT_TOKEN}{assistant_prefix}{target_token_str}"
    path.write_text(text)


def free_models(*models) -> None:
    for m in models:
        del m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
