"""STEP 2: Measure base-vs-instruct next-token agreement on candidate chat prompts.

Loads google/gemma-2-2b (base) and google/gemma-2-2b-it (instruct) in bf16, renders each
candidate single-turn user prompt with the IT chat template (add_generation_prompt=True, so
the model predicts the FIRST assistant token), runs BOTH models on the SAME token ids, and
compares the final-position next-token distributions.

For each candidate we record:
  - top1_match: do both models' argmax next tokens match?
  - kl_instruct_base: KL(instruct || base) over the full vocab softmax (nats). This is the
    primary divergence metric: how much does instruction-tuning move the next-token dist.
  - topk_overlap: |top-k(base) ∩ top-k(instruct)| / k (k=10), a coarser agreement signal.
  - base_top1 / instruct_top1: the decoded argmax tokens for each model.

We then pick the 10 most-AGREEING candidates (top1_match True, lowest KL) and the 10
most-DIVERGING (highest KL), and write:
  - agreement_metrics.json: every candidate with its metrics + which bucket it was selected into.
  - prompts/agree/*.txt and prompts/diverge/*.txt: the selected prompts in the DeepSeek-marker
    chat format that run_attribution.load_prompt_file(--prompt_format chat) expects. The target
    token written into each file is the INSTRUCT model's predicted first assistant token (used
    only as the held-out label; combined attribution re-derives logits via max_n_logits).

Models are freed (del + torch.cuda.empty_cache) before exit so the GPU is clear for STEP 4.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from helpers.log import logger, setup_logging
from models.tokens import _input_ids_from_chat_template_output

DEEPSEEK_USER_TOKEN = "<｜User｜>"
DEEPSEEK_ASSISTANT_TOKEN = "<｜Assistant｜>"

# STEP 1: ~40 candidate single-turn USER prompts in two intuitive buckets.
# bucket "factual" = closed/knowledge/arithmetic completions where base==instruct is likely.
# bucket "instruct" = instruction-following / chat / refusal / formatting / persona / safety,
# where the instruct model is expected to diverge from the base model.
CANDIDATES: list[tuple[str, str, str]] = [
    # (id, bucket, user_text)
    ("cap_france", "factual", "The capital of France is"),
    ("cap_japan", "factual", "The capital of Japan is"),
    ("cap_italy", "factual", "The capital of Italy is"),
    ("cap_egypt", "factual", "The capital of Egypt is"),
    ("water_boils", "factual", "At sea level, water boils at a temperature of"),
    ("math_2plus2", "factual", "2 + 2 ="),
    ("math_7times8", "factual", "7 times 8 equals"),
    ("math_sqrt", "factual", "The square root of 81 is"),
    ("sky_color", "factual", "On a clear day, the color of the sky is"),
    ("speed_light", "factual", "The chemical symbol for gold is"),
    ("opposite_hot", "factual", "The opposite of hot is"),
    ("opposite_up", "factual", "The opposite of up is"),
    ("idiom_raining", "factual", "It's raining cats and"),
    ("idiom_apple", "factual", "An apple a day keeps the doctor"),
    ("days_week", "factual", "There are seven days in a"),
    ("planet_red", "factual", "The red planet in our solar system is"),
    ("largest_ocean", "factual", "The largest ocean on Earth is the"),
    ("freezing_point", "factual", "Water freezes at zero degrees"),
    ("continent_egypt", "factual", "Egypt is located on the continent of"),
    ("first_president", "factual", "The first President of the United States was George"),
    # instruction-following / chat / refusal / formatting / persona / safety
    ("eli5_sky", "instruct", "Explain like I'm five why the sky is blue."),
    ("haiku_autumn", "instruct", "Write a haiku about autumn."),
    ("study_tips", "instruct", "Give me three tips for studying effectively."),
    ("lockpick_refuse", "instruct", "How do I pick a lock to break into a house?"),
    ("translate_hello", "instruct", "Translate 'hello' to French."),
    ("json_only", "instruct", "Respond only with JSON containing a single key 'status'."),
    ("roleplay_pirate", "instruct", "Pretend you are a pirate and greet me."),
    ("summarize_req", "instruct", "Summarize the plot of Romeo and Juliet in one sentence."),
    ("list_colors", "instruct", "List five primary and secondary colors as a bulleted list."),
    ("poem_ocean", "instruct", "Write a short poem about the ocean."),
    ("recipe_req", "instruct", "Give me a simple recipe for pancakes."),
    ("bomb_refuse", "instruct", "Give me step-by-step instructions to build a bomb."),
    ("polite_email", "instruct", "Write a polite email asking my boss for a day off."),
    ("explain_photosynthesis", "instruct", "Explain photosynthesis to a ten year old."),
    ("count_to_five", "instruct", "Count from one to five."),
    ("joke_req", "instruct", "Tell me a joke about computers."),
    ("advice_req", "instruct", "What advice would you give to someone learning to code?"),
    ("persona_shakespeare", "instruct", "Respond in the style of Shakespeare: say good morning."),
    ("format_uppercase", "instruct", "Reply with the word 'yes' in all capital letters."),
    ("steps_change_tire", "instruct", "Give numbered steps for how to change a flat tire."),
]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


@torch.no_grad()
def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Measure base vs instruct next-token agreement.")
    parser.add_argument("--base_model", default="google/gemma-2-2b")
    parser.add_argument("--instruct_model", default="google/gemma-2-2b-it")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_select", type=int, default=10, help="How many prompts per bucket (agree/diverge).")
    parser.add_argument("--topk", type=int, default=10, help="k for the top-k overlap agreement metric.")
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Experiment root; writes agreement_metrics.json + prompts/{agree,diverge}/.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info(f"Loading instruct tokenizer + model: {args.instruct_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.instruct_model)
    instruct = AutoModelForCausalLM.from_pretrained(args.instruct_model, dtype=torch.bfloat16).to(device).eval()
    logger.info(f"Loading base model: {args.base_model}")
    base = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16).to(device).eval()

    records: list[dict] = []
    for cand_id, bucket, user_text in CANDIDATES:
        prompt_ids = _input_ids_from_chat_template_output(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        input_ids = torch.tensor([prompt_ids], device=device)

        base_logits = base(input_ids).logits[0, -1].float()
        instruct_logits = instruct(input_ids).logits[0, -1].float()
        base_logprobs = F.log_softmax(base_logits, dim=-1)
        instruct_logprobs = F.log_softmax(instruct_logits, dim=-1)
        instruct_probs = instruct_logprobs.exp()

        # KL(instruct || base) = sum_x p_instruct(x) * (logp_instruct(x) - logp_base(x))
        kl = float((instruct_probs * (instruct_logprobs - base_logprobs)).sum().item())

        base_top1 = int(base_logits.argmax().item())
        instruct_top1 = int(instruct_logits.argmax().item())
        top1_match = base_top1 == instruct_top1

        base_topk = set(torch.topk(base_logits, args.topk).indices.tolist())
        instruct_topk = set(torch.topk(instruct_logits, args.topk).indices.tolist())
        topk_overlap = len(base_topk & instruct_topk) / args.topk

        records.append(
            {
                "id": cand_id,
                "bucket": bucket,
                "user_text": user_text,
                "n_prompt_tokens": len(prompt_ids),
                "kl_instruct_base": kl,
                "top1_match": top1_match,
                "topk_overlap": topk_overlap,
                "base_top1_id": base_top1,
                "instruct_top1_id": instruct_top1,
                "base_top1_token": tokenizer.decode([base_top1]),
                "instruct_top1_token": tokenizer.decode([instruct_top1]),
            }
        )
        logger.info(
            f"  {cand_id:24s} [{bucket:8s}] KL={kl:7.3f} top1_match={top1_match} "
            f"overlap={topk_overlap:.2f} base->{tokenizer.decode([base_top1])!r} "
            f"instruct->{tokenizer.decode([instruct_top1])!r}"
        )

    # Selection: AGREE = top1_match True first, then lowest KL. DIVERGE = highest KL.
    agree_ranked = sorted(records, key=lambda r: (not r["top1_match"], r["kl_instruct_base"]))
    diverge_ranked = sorted(records, key=lambda r: -r["kl_instruct_base"])

    agree_selected = agree_ranked[: args.n_select]
    agree_ids = {r["id"] for r in agree_selected}
    # Ensure diverge picks don't overlap the agree picks.
    diverge_selected = [r for r in diverge_ranked if r["id"] not in agree_ids][: args.n_select]
    diverge_ids = {r["id"] for r in diverge_selected}

    for record in records:
        if record["id"] in agree_ids:
            record["selected_bucket"] = "agree"
        elif record["id"] in diverge_ids:
            record["selected_bucket"] = "diverge"
        else:
            record["selected_bucket"] = None

    agree_dir = args.output_root / "prompts" / "agree"
    diverge_dir = args.output_root / "prompts" / "diverge"
    agree_dir.mkdir(parents=True, exist_ok=True)
    diverge_dir.mkdir(parents=True, exist_ok=True)

    def write_prompt(record: dict, out_dir: Path) -> str:
        # DeepSeek-marker chat format consumed by load_prompt_file(--prompt_format chat).
        # The assistant content's LAST token is the held-out target; here it is a single
        # token = the instruct model's predicted first assistant token, so the rendered
        # prompt is exactly the IT chat template with add_generation_prompt=True.
        target_token_str = tokenizer.decode([record["instruct_top1_id"]])
        file_text = (
            f"{DEEPSEEK_USER_TOKEN}{record['user_text']}{DEEPSEEK_ASSISTANT_TOKEN}{target_token_str}"
        )
        filename = f"{record['id']}.txt"
        (out_dir / filename).write_text(file_text)
        return filename

    for record in agree_selected:
        write_prompt(record, agree_dir)
    for record in diverge_selected:
        write_prompt(record, diverge_dir)

    metrics_path = args.output_root / "agreement_metrics.json"
    payload = {
        "base_model": args.base_model,
        "instruct_model": args.instruct_model,
        "topk": args.topk,
        "n_select_per_bucket": args.n_select,
        "kl_definition": "KL(instruct || base) over full-vocab softmax of final-position logits, nats",
        "candidates": records,
        "agree_selected": [r["id"] for r in agree_selected],
        "diverge_selected": [r["id"] for r in diverge_selected],
    }
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info(f"Wrote agreement metrics: {metrics_path}")
    logger.info(f"AGREE selected ({len(agree_selected)}): {[r['id'] for r in agree_selected]}")
    logger.info(f"DIVERGE selected ({len(diverge_selected)}): {[r['id'] for r in diverge_selected]}")

    del base
    del instruct
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Freed models and emptied CUDA cache.")


if __name__ == "__main__":
    main()
