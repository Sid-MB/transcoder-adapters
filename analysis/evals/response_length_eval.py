#!/usr/bin/env python3
#%%
"""
Eval: compare response lengths between a trained model and the base instruction-tuned model.

Generates responses for 100 prompts from the LMSYS chat validation set and reports
length statistics (tokens) for each model.  A large divergence in length indicates the
trained model's generation behaviour has drifted from the base model.

Usage:
    python -m analysis.evals.response_length_eval --model <trained_model_path>
    python -m analysis.evals.response_length_eval \\
        --model /path/to/checkpoint \\
        --base_model google/gemma-2-2b-it \\
        --transcoder
"""

from helpers.log import logger, setup_logging
from helpers.paths import PRODUCTS_DIR

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
import random
import numpy as np
import torch
from dataclasses import dataclass, asdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LMSYS_DATASET = "siddharthmb/2026.transcoder-adapters.lmsys-chat-1m-splits"
N_SAMPLES = 100
MAX_PROMPT_TOKENS = 512   # truncate long prompts so generation stays tractable
MAX_NEW_TOKENS = 512
SEED = 42
OUTPUT_DIR = PRODUCTS_DIR / "response_length_evals"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Using device: {DEVICE}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_lmsys_prompts(tokenizer, n_samples: int, seed: int) -> list[dict]:
    """Return up to `n_samples` prompts from the LMSYS val split.

    Each element has:
        prompt_ids  – list[int]   tokenised prompt (with generation prompt appended)
        question_id – str         unique identifier for deduplication
    """
    from datasets import load_dataset, DatasetDict

    ds = load_dataset(LMSYS_DATASET)
    assert isinstance(ds, DatasetDict)

    split = None
    for candidate in ("val", "validation", "test"):
        if candidate in ds:
            split = ds[candidate]
            break
    if split is None:
        raise ValueError(f"No val split in {LMSYS_DATASET}. Available: {list(ds.keys())}")

    logger.info(f"LMSYS val split: {len(split)} conversations")

    # Collect first assistant turn from each conversation so prompts are short
    # and diverse.  Stop early once we have enough.
    prompts: list[dict] = []
    rng = random.Random(seed)
    indices = list(range(len(split)))
    rng.shuffle(indices)

    for conv_idx in indices:
        row = split[conv_idx]
        conversation = row["conversation"]

        for turn_idx, msg in enumerate(conversation):
            if msg["role"] not in ("assistant", "model"):
                continue
            prefix = conversation[:turn_idx]
            if not prefix:
                continue

            prompt_ids = tokenizer.apply_chat_template(
                prefix, tokenize=True, add_generation_prompt=True,
            )

            # Skip very long prompts to keep generation tractable
            if len(prompt_ids) > MAX_PROMPT_TOKENS:
                continue

            prompts.append({
                "prompt_ids": prompt_ids,
                "question_id": f"lmsys_{conv_idx}_{turn_idx}",
            })
            break  # one prompt per conversation

        if len(prompts) >= n_samples:
            break

    logger.info(f"Collected {len(prompts)} prompts")
    return prompts


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate_response_lengths(
    model,
    prompts: list[dict],
    max_new_tokens: int = MAX_NEW_TOKENS,
    device: str = DEVICE,
) -> list[int]:
    """Generate one response per prompt and return token counts."""
    lengths: list[int] = []

    for ex in tqdm(prompts, desc="Generating"):
        input_ids = torch.tensor([ex["prompt_ids"]], dtype=torch.long, device=device)

        # Gemma2 exposes eos_token_id as a list; model.generate requires a scalar.
        eos_id = model.config.eos_token_id
        pad_id = eos_id[0] if isinstance(eos_id, list) else eos_id

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy – deterministic & reproducible
                pad_token_id=pad_id,
            )

        n_new = output_ids.shape[1] - input_ids.shape[1]
        lengths.append(int(n_new))

    return lengths


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
@dataclass
class LengthStats:
    n: int
    mean: float
    median: float
    std: float
    p10: float
    p25: float
    p75: float
    p90: float
    min: int
    max: int
    n_truncated: int   # responses that hit max_new_tokens


def compute_stats(lengths: list[int], max_new_tokens: int) -> LengthStats:
    a = np.array(lengths)
    return LengthStats(
        n=len(a),
        mean=float(np.mean(a)),
        median=float(np.median(a)),
        std=float(np.std(a)),
        p10=float(np.percentile(a, 10)),
        p25=float(np.percentile(a, 25)),
        p75=float(np.percentile(a, 75)),
        p90=float(np.percentile(a, 90)),
        min=int(np.min(a)),
        max=int(np.max(a)),
        n_truncated=int(np.sum(a >= max_new_tokens)),
    )


def log_stats(label: str, stats: LengthStats, max_new_tokens: int) -> None:
    logger.info(f"\n{'='*60}")
    logger.info(f"Response length stats: {label}")
    logger.info(f"{'='*60}")
    logger.info(f"  n         : {stats.n}")
    logger.info(f"  mean      : {stats.mean:.1f}")
    logger.info(f"  median    : {stats.median:.1f}")
    logger.info(f"  std       : {stats.std:.1f}")
    logger.info(f"  [p10, p90]: [{stats.p10:.0f}, {stats.p90:.0f}]")
    logger.info(f"  [min, max]: [{stats.min}, {stats.max}]")
    logger.info(f"  truncated : {stats.n_truncated} (hit max_new_tokens={max_new_tokens})")


def log_comparison(base_stats: LengthStats, eval_stats: LengthStats) -> None:
    logger.info(f"\n{'='*60}")
    logger.info("Comparison summary (trained / base)")
    logger.info(f"{'='*60}")
    ratio_mean   = eval_stats.mean   / base_stats.mean   if base_stats.mean   else float('nan')
    ratio_median = eval_stats.median / base_stats.median if base_stats.median else float('nan')
    logger.info(f"  mean ratio   : {ratio_mean:.3f}  ({eval_stats.mean:.1f} vs {base_stats.mean:.1f})")
    logger.info(f"  median ratio : {ratio_median:.3f}  ({eval_stats.median:.1f} vs {base_stats.median:.1f})")

    # Simple sanity band: warn if mean length differs by more than 50 %
    if not (0.5 <= ratio_mean <= 2.0):
        logger.warning(
            f"Mean length ratio {ratio_mean:.2f} is outside [0.5, 2.0] – "
            "the trained model's generation length has drifted significantly."
        )
    else:
        logger.info("  -> Length ratio within expected range [0.5, 2.0].")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_path: str, use_transcoder: bool = False):
    if use_transcoder:
        from models.auto import AutoModelForCausalLMWithTranscoder
        model = AutoModelForCausalLMWithTranscoder.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        logger.info("  Loaded as transcoder model")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        logger.info("  Loaded as AutoModelForCausalLM")
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Compare response lengths of a trained model vs the base instruction-tuned model "
                    "on 100 prompts from the LMSYS chat validation set."
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Path or HF id of the trained model to evaluate.")
    parser.add_argument("--base_model", type=str, default="google/gemma-2-2b-it",
                        help="Path or HF id of the base instruction-tuned model (default: google/gemma-2-2b-it).")
    parser.add_argument("--n_samples", type=int, default=N_SAMPLES)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--transcoder", action="store_true",
                        help="Load the trained model as a transcoder checkpoint.")
    args = parser.parse_args()

    # Tokenizer from the base model
    logger.info(f"Loading tokenizer from {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prompts
    prompts = load_lmsys_prompts(tokenizer, n_samples=args.n_samples, seed=args.seed)
    if len(prompts) < args.n_samples:
        logger.warning(f"Only {len(prompts)} prompts collected (requested {args.n_samples}).")

    # ---- Base model ----
    logger.info(f"\nLoading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    base_model.eval()

    logger.info("Generating responses with base model …")
    base_lengths = generate_response_lengths(base_model, prompts, args.max_new_tokens)
    base_stats = compute_stats(base_lengths, args.max_new_tokens)
    log_stats(f"base ({args.base_model})", base_stats, args.max_new_tokens)

    del base_model
    torch.cuda.empty_cache()

    # ---- Trained model ----
    logger.info(f"\nLoading trained model: {args.model}")
    eval_model = load_model(args.model, use_transcoder=args.transcoder)
    eval_model.eval()

    logger.info("Generating responses with trained model …")
    eval_lengths = generate_response_lengths(eval_model, prompts, args.max_new_tokens)
    eval_stats = compute_stats(eval_lengths, args.max_new_tokens)
    log_stats(f"trained ({args.model})", eval_stats, args.max_new_tokens)

    del eval_model
    torch.cuda.empty_cache()

    # ---- Comparison ----
    log_comparison(base_stats, eval_stats)

    # ---- Save ----
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    clean_name = args.model.rstrip("/").replace("/", "__")
    output_path = OUTPUT_DIR / f"response_length__{clean_name}.json"
    output = {
        "model": args.model,
        "base_model": args.base_model,
        "n_samples": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "base": asdict(base_stats),
        "trained": asdict(eval_stats),
        "mean_ratio": eval_stats.mean / base_stats.mean if base_stats.mean else None,
        "median_ratio": eval_stats.median / base_stats.median if base_stats.median else None,
        "base_lengths": base_lengths,
        "trained_lengths": eval_lengths,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nSaved results to {output_path}")


#%%
if __name__ == "__main__":
    main()
