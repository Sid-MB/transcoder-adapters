# Created by Claude Code session "explore: queries to analyze".
"""FINDER #2 (base/instruct-divergent): assemble a few-hundred-prompt corpus (lmsys val prompts,
Alpaca-style instructions, HarmBench), render each with the IT chat template, greedy-generate the
first ~8 response tokens with the INSTRUCT model, and compute KL(instruct||base) at each of those
positions (teacher-forced over both models). Score each prompt by its MAX KL over those positions.

Rank descending, keep the top ~15-20. Target token = the argmax-KL position; we report its index
and the base-top1 vs instruct-top1 decoded tokens there, and emit a .txt prompt for attribution with
the response prefix up to (not including) the argmax token as the assistant prefix.

Usage:
  LARGE_ARTIFACTS_DIR=/nlp/scr/siddharth uv run --no-sync python \
    experiments/interesting_queries/scripts/find_divergent.py --n_corpus 300 --n_select 18
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402

from helpers.log import logger, setup_logging  # noqa: E402
import common  # noqa: E402

EXP_ROOT = Path(__file__).resolve().parents[1]
LMSYS_DATASET = "siddharthmb/2026.transcoder-adapters.lmsys-chat-1m-splits"

# A small in-file set of Alpaca-style instructions (self-contained; avoids another dataset download).
ALPACA_STYLE = [
    "Give three tips for staying healthy.",
    "Explain the difference between a virus and bacteria.",
    "Write a short story about a robot learning to paint.",
    "Describe the process of photosynthesis.",
    "Suggest a name for a new coffee shop and explain your choice.",
    "Convert the temperature 30 degrees Celsius to Fahrenheit and show the steps.",
    "Write a haiku about the ocean.",
    "List the primary colors and explain why they are called primary.",
    "Summarize the theory of relativity in two sentences.",
    "Give me a recipe for a simple vegetable soup.",
    "Explain how to tie a shoelace to a five year old.",
    "What are the pros and cons of remote work?",
    "Write a formal email requesting a meeting with a professor.",
    "Translate 'good morning, how are you?' into Spanish.",
    "Describe your ideal vacation.",
    "Explain what machine learning is without using jargon.",
    "Give advice to someone starting their first job.",
    "Write a limerick about a cat.",
    "What is the capital of Australia and what is it known for?",
    "Explain why the sky appears blue during the day.",
]


def _lmsys_first_user_prompts(n: int) -> list[str]:
    ds = load_dataset(LMSYS_DATASET)["val"]
    out = []
    for row in ds:
        conv = row["conversation"]
        if conv and conv[0]["role"] == "user":
            text = conv[0]["content"].strip()
            if 3 <= len(text) <= 600 and row.get("language", "English") == "English":
                out.append(text)
        if len(out) >= n:
            break
    return out


def _harmbench_prompts(n: int) -> list[str]:
    ds = load_dataset("walledai/HarmBench", "standard")["train"]
    return list(ds["prompt"])[:n]


@torch.no_grad()
def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_corpus", type=int, default=300, help="Approx total corpus size to score.")
    parser.add_argument("--n_lmsys", type=int, default=230, help="How many lmsys val first-user prompts to include.")
    parser.add_argument("--n_harmbench", type=int, default=50, help="How many HarmBench prompts to include.")
    parser.add_argument("--n_positions", type=int, default=8, help="First N response-token positions scored for KL.")
    parser.add_argument("--n_select", type=int, default=18, help="Top-KL prompts to keep.")
    parser.add_argument("--output_root", type=Path, default=EXP_ROOT)
    args = parser.parse_args()

    logger.info(f"INVOCATION: find_divergent.py {vars(args)}")
    corpus = []
    for p in _lmsys_first_user_prompts(args.n_lmsys):
        corpus.append(("lmsys", p))
    for p in ALPACA_STYLE:
        corpus.append(("alpaca", p))
    for p in _harmbench_prompts(args.n_harmbench):
        corpus.append(("harmbench", p))
    corpus = corpus[: args.n_corpus]
    logger.info(f"Corpus size {len(corpus)} (lmsys+alpaca+harmbench).")

    tokenizer, base, instruct = common.load_models(args.device)

    records = []
    for i, (source, prompt) in enumerate(corpus):
        prompt_ids = common.render_chat_ids(tokenizer, prompt)
        r = common.kl_over_response(base, instruct, tokenizer, prompt_ids, args.n_positions, args.device)
        rec = {
            "id": f"div_{i:04d}",
            "source": source,
            "prompt": prompt,
            "max_kl": r["max_kl"],
            "argmax_pos": r.get("argmax_pos", 0),
            "argmax_seq_index": r.get("argmax_seq_index"),
            "base_top1_token": r.get("argmax_base_top1_token"),
            "instruct_top1_token": r.get("argmax_instruct_top1_token"),
            "resp_text": r.get("resp_text", ""),
            "resp_ids": r.get("resp_ids", []),
            "n_prompt_tokens": len(prompt_ids),
        }
        records.append(rec)
        if i % 25 == 0 or r["max_kl"] > 5:
            logger.info(f"  {rec['id']} [{source:9s}] maxKL={r['max_kl']:6.2f}@pos{rec['argmax_pos']} base->{rec['base_top1_token']!r} inst->{rec['instruct_top1_token']!r} :: {prompt[:60]!r}")

    records.sort(key=lambda r: -r["max_kl"])
    selected = records[: args.n_select]

    prompts_dir = common.ATTRIBUTION_PROMPTS_DIR / "divergent"
    for r in selected:
        # Assistant prefix = response tokens BEFORE the argmax position; target = the argmax token.
        prefix_ids = r["resp_ids"][: r["argmax_pos"]]
        target_id = r["resp_ids"][r["argmax_pos"]] if r["argmax_pos"] < len(r["resp_ids"]) else r["resp_ids"][-1]
        prefix_str = tokenizer.decode(prefix_ids) if prefix_ids else ""
        target_str = tokenizer.decode([target_id])
        common.write_prompt_txt(prompts_dir / f"{r['id']}.txt", r["prompt"], prefix_str, target_str)
        r["assistant_prefix"] = prefix_str
        r["target_token_written"] = target_str

    out = args.output_root / "results" / "find_divergent.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "finder": "divergent",
        "metric": "max KL(instruct||base) over first N response positions (nats), greedy instruct rollout",
        "n_positions": args.n_positions,
        "n_corpus": len(corpus),
        "n_selected": len(selected),
        "selected_ids": [r["id"] for r in selected],
        "prompts_dir": str(prompts_dir),
        "records": records,
    }, indent=2) + "\n")
    logger.info(f"Wrote {out}; top max_kl={selected[0]['max_kl']:.2f}; emitted {len(selected)} prompts to {prompts_dir}")
    common.free_models(base, instruct)


if __name__ == "__main__":
    main()
