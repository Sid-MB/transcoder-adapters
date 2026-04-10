#!/usr/bin/env python3
#%%
"""
Compute token-level metrics (KL divergence, top-1 agreement) for a model
against R1-Distill reference using on-policy rollouts from evalchemy.

Usage:
    python -m analysis.evals.compute_token_metrics_onpolicy --model <model_path>
    python -m analysis.evals.compute_token_metrics_onpolicy --model Qwen/Qwen2.5-Math-7B
"""

from helpers.log import log_group, logger, setup_logging
from pathlib import Path

import os

from helpers.paths import PRODUCTS_DIR
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
import glob
import torch
import numpy as np
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


@dataclass(frozen=True)
class TokenMetrics:
    n_tokens: int
    kl_mean: float
    top1_agreement: float
    n_interesting: int
    kl_mean_interesting: float
    top1_agreement_interesting: float


@dataclass(frozen=True)
class BenchmarkMetrics(TokenMetrics):
    n_samples: int = 0


@dataclass(frozen=True)
class EvalResults(TokenMetrics):
    per_benchmark: dict[str, BenchmarkMetrics] = field(default_factory=dict)



# Output directory
OUTPUT_DIR = PRODUCTS_DIR / "token_recon_evals"

#%%
# ============================================================================
# CONFIG
# ============================================================================

EVALCHEMY_DIR = Path("/nlp/scr/nathu/sparse-adaptation/evalchemy_v5")
LMSYS_DATASET = "siddharthmb/2026.transcoder-adapters.lmsys-chat-1m-splits"

# Data source configurations
DATA_SOURCES = ["evalchemy_qwen", "lmsys_chat"]

# Prompt templates (matching evalchemy)
MATH_PROMPT = """Problem: {problem}
Mark your solution with \\boxed
Answer:"""

GPQA_PROMPT = """Return your final response within \\boxed{{}} and only include the letter choice (A, B, C, or D) as your final response.
Problem: {problem}
Options: {options}
Answer:"""

LIVECODE_STDIN_PROMPT = """Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output. Simply call the function after the definition."""

LIVECODE_NONSTDIN_PROMPT = """Generate an executable Python function generated from the given prompt. Return the function body without invoking it at the final solution."""

# Sampling config
N_SAMPLES = 200
MAX_SEQ_LENGTH = 10000
SEED = 42
DEBUG_SAMPLES = 2  # Number of samples to show in debug mode

# Interesting token threshold: tokens where ref model's top prob <= this
INTERESTING_THRESHOLD = 0.8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Log device
logger.info(f"Using device: {DEVICE}")

#%%
# ============================================================================
# DATA LOADING (no filtering)
# ============================================================================
def load_livecode_is_stdin_map():
    """Load pre-computed task_id -> is_stdin mapping."""
    mapping_path = "/juice2/u/nathu/sparse_adaptation/data/livecode_is_stdin_map.json"
    with open(mapping_path) as f:
        return json.load(f)


def load_all_rollouts(eval_dir: str):
    """Load all rollouts from evalchemy results (no filtering).

    Each example includes a 'question_id' for deduplication during sampling.
    """
    examples = []

    # Find newest file for each benchmark
    json_files = glob.glob(os.path.join(eval_dir, "*.json"))

    if not json_files:
        logger.error(f"No JSON files found in {eval_dir}")
        raise FileNotFoundError(f"No JSON files found in {eval_dir}")

    benchmark_files = {}
    for json_path in json_files:
        filename = os.path.basename(json_path)
        parts = filename.split('_')
        if len(parts) >= 2:
            bm_name = parts[1]
            if bm_name not in benchmark_files:
                benchmark_files[bm_name] = []
            benchmark_files[bm_name].append(json_path)

    for bm_name in benchmark_files:
        benchmark_files[bm_name] = sorted(benchmark_files[bm_name])[-1]

    logger.info(f"Loading rollouts from {eval_dir}:")
    for bm_name, path in sorted(benchmark_files.items()):
        logger.info(f"  {bm_name}: {os.path.basename(path)}")

    # Load LiveCodeBench is_stdin mapping
    lcb_is_stdin = load_livecode_is_stdin_map()

    for json_path in benchmark_files.values():
        with open(json_path) as f:
            data = json.load(f)

        results = data.get('results', {})

        # MATH500 (single sample)
        if 'MATH500' in results:
            for i, ex in enumerate(results['MATH500'].get('examples', [])):
                problem = ex.get('problem', '')
                model_output = ex.get('model_output', '')
                if model_output:
                    examples.append({
                        'prompt': MATH_PROMPT.format(problem=problem),
                        'response': model_output,
                        'benchmark': 'MATH500',
                        'question_id': f"MATH500_{i}",
                    })

        # AIME25, AMC23 (multi-sample)
        for bm in ['AIME25', 'AMC23']:
            if bm in results:
                for i, ex in enumerate(results[bm].get('examples', [])):
                    problem = ex.get('problem', ex.get('question', ''))
                    for j, out in enumerate(ex.get('model_outputs', [])):
                        if out:
                            examples.append({
                                'prompt': MATH_PROMPT.format(problem=problem),
                                'response': out,
                                'benchmark': bm,
                                'question_id': f"{bm}_{i}",
                            })

        # GPQADiamond (multi-sample)
        if 'GPQADiamond' in results:
            for i, ex in enumerate(results['GPQADiamond'].get('examples', [])):
                question = ex.get('Question', '')
                options = ex.get('multiple_choice_string', '')
                for j, out in enumerate(ex.get('model_outputs', [])):
                    if out:
                        examples.append({
                            'prompt': GPQA_PROMPT.format(problem=question, options=options),
                            'response': out,
                            'benchmark': 'GPQADiamond',
                            'question_id': f"GPQADiamond_{i}",
                        })

        # LiveCodeBench
        if 'LiveCodeBenchv5_official' in results:
            for ex in results['LiveCodeBenchv5_official'].get('examples', []):
                task_id = ex.get('task_id', '')
                prompt_text = ex.get('prompt', '')
                model_output = ex.get('model_output', '')

                if model_output and task_id in lcb_is_stdin:
                    is_stdin = lcb_is_stdin[task_id]
                    if is_stdin:
                        full_prompt = LIVECODE_STDIN_PROMPT + prompt_text
                    else:
                        full_prompt = LIVECODE_NONSTDIN_PROMPT + prompt_text

                    examples.append({
                        'prompt': full_prompt,
                        'response': model_output,
                        'benchmark': 'LiveCodeBench',
                        'question_id': f"LiveCodeBench_{task_id}",
                    })

    logger.info(f"\nLoaded {len(examples)} total rollouts")

    # Print per-benchmark counts
    from collections import Counter
    bm_counts = Counter(ex['benchmark'] for ex in examples)
    for bm, count in sorted(bm_counts.items()):
        logger.info(f"  {bm}: {count}")

    return examples


def load_lmsys_examples(tokenizer, max_examples: int | None = None) -> list[dict]:
    """Load assistant turns from the LMSYS chat val split as prompt/response examples.

    Each multi-turn conversation produces one example per assistant turn, using
    the preceding messages as the prompt context.

    Args:
        tokenizer: Tokenizer for encoding prompt/response pairs.
        max_examples: If set, stop after collecting this many examples to avoid
            tokenizing the entire dataset. Use ~2x n_samples to leave room for
            deduplicated sampling.
    """
    from datasets import load_dataset, DatasetDict

    ds = load_dataset(LMSYS_DATASET, trust_remote_code=True) # pyright: ignore[reportAssignmentType]
    assert isinstance(ds, DatasetDict), f'Expected a DatasetDict with splits, got {type(ds)} for "{LMSYS_DATASET}"'

    split = None
    for candidate in ("val", "validation", "test"):
        if candidate in ds:
            split = ds[candidate]
            break
    if split is None:
        raise ValueError(f"No val split found in {LMSYS_DATASET}. Available: {list(ds.keys())}")

    logger.info(f"Loading LMSYS chat val split: {len(split)} conversations")

    examples = []
    conv_idx = -1
    for conv_idx, row in enumerate(split):
        conversation = row["conversation"] # pyright: ignore[reportArgumentType, reportCallIssue]
        for turn_idx, msg in enumerate(conversation):
            if msg["role"] not in ("assistant", "model"):
                continue
            # Prompt = all messages before this assistant turn
            prefix_messages = conversation[:turn_idx]
            if not prefix_messages:
                continue

            # Tokenize prefix with generation prompt to get the prompt boundary
            prompt_ids = tokenizer.apply_chat_template(
                prefix_messages, tokenize=True, add_generation_prompt=True,
            )
            response_ids = tokenizer.encode(msg["content"], add_special_tokens=False)

            examples.append({
                'prompt_ids': prompt_ids,
                'response_ids': response_ids,
                'benchmark': 'lmsys_chat',
                'question_id': f"lmsys_{conv_idx}_{turn_idx}",
            })
            if max_examples is not None and len(examples) >= max_examples:
                break
        if max_examples is not None and len(examples) >= max_examples:
            break

    logger.info(f"Extracted {len(examples)} assistant turns from {conv_idx + 1} conversations")
    return examples


def sample_deduplicated(examples: list, n_samples: int, seed: int):
    """Sample examples, prioritizing one rollout per question before duplicates."""
    import random
    random.seed(seed)

    # Group by question_id
    from collections import defaultdict
    by_question = defaultdict(list)
    for ex in examples:
        by_question[ex['question_id']].append(ex)

    # Shuffle rollouts within each question
    for qid in by_question:
        random.shuffle(by_question[qid])

    # Sample round-robin: one from each question, then repeat
    sampled = []
    question_ids = list(by_question.keys())
    random.shuffle(question_ids)

    round_idx = 0
    while len(sampled) < n_samples:
        added_this_round = False
        for qid in question_ids:
            if round_idx < len(by_question[qid]):
                sampled.append(by_question[qid][round_idx])
                added_this_round = True
                if len(sampled) >= n_samples:
                    break
        if not added_this_round:
            break  # No more examples to add
        round_idx += 1

    return sampled


#%%
# ============================================================================
# TOKENIZATION
# ============================================================================
def tokenize_example(ex: dict, tokenizer, max_length: int):
    """Tokenize a single example into input_ids and labels.

    Supports two formats:
    - evalchemy: has 'prompt' and 'response' strings
    - lmsys_chat: has pre-tokenized 'prompt_ids' and 'response_ids' lists
    """
    if 'prompt_ids' in ex:
        prompt_ids = ex['prompt_ids']
        response_ids = ex['response_ids']
    else:
        messages = [{"role": "user", "content": ex['prompt']}]
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        response_ids = tokenizer.encode(ex['response'], add_special_tokens=False)

    input_ids = prompt_ids + response_ids

    # Truncate if needed
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]

    # Labels: -100 for prompt, actual ids for response
    labels = [-100] * len(prompt_ids) + response_ids
    if len(labels) > max_length:
        labels = labels[:max_length]

    return {
        'input_ids': torch.tensor(input_ids),
        'labels': torch.tensor(labels),
    }


#%%
# ============================================================================
# METRICS
# ============================================================================
def compute_token_metrics(logits_model, logits_ref, labels):
    """
    Compute per-token KL divergence and top-1 agreement.
    Returns metrics only for response tokens (labels != -100).
    Also returns ref model's max prob for filtering "interesting" tokens.
    """
    # Shift: logits[i] predicts token[i+1]
    logits_model = logits_model[:, :-1, :]
    logits_ref = logits_ref[:, :-1, :]
    labels_shifted = labels[:, 1:]

    # Slice to response positions before expensive softmax/log_softmax
    response_mask = labels_shifted[0] != -100
    logits_ref_resp = logits_ref[0, response_mask].float()      # (n_resp, vocab)
    logits_model_resp = logits_model[0, response_mask].float()  # (n_resp, vocab)

    # Softmax / log softmax
    p_ref = torch.softmax(logits_ref_resp, dim=-1)
    log_p_ref = torch.log_softmax(logits_ref_resp, dim=-1)
    log_p_model = torch.log_softmax(logits_model_resp, dim=-1)

    # Reference model's max probability (for filtering interesting tokens)
    max_prob_ref = p_ref.max(dim=-1).values

    # KL divergence: KL(ref || model)
    kl_per_token = (p_ref * (log_p_ref - log_p_model)).sum(dim=-1)

    # Top-1 agreement
    top1_ref = logits_ref_resp.argmax(dim=-1)
    top1_model = logits_model_resp.argmax(dim=-1)
    top1_match = (top1_ref == top1_model)

    return {
        'kl': kl_per_token.cpu().numpy(),
        'top1_match': top1_match.cpu().numpy(),
        'ref_max_prob': max_prob_ref.cpu().numpy(),
    }


#%%
# ============================================================================
# MAIN EVALUATION
# ============================================================================
def load_model(model_path: str, use_transcoder: bool = False):
    """Load model, optionally as transcoder (auto-detects architecture)."""
    if use_transcoder:
        from models.auto import AutoModelForCausalLMWithTranscoder
        model = AutoModelForCausalLMWithTranscoder.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        logger.info(f"  Loaded as transcoder model (auto-detected arch)")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        logger.info(f"  Loaded as AutoModelForCausalLM")

    return model


@log_group("Token-level evaluation")
def run_token_metrics_eval(
    eval_model,
    ref_model,
    tokenizer,
    examples: list[dict],
    max_length: int = MAX_SEQ_LENGTH,
    device: str = DEVICE,
    debug: bool = False,
    save_output: bool = False,
) -> EvalResults:
    """Compute token-level KL divergence and top-1 agreement between two models.

    This is the main callable API. Models must already be loaded and in eval mode.

    Args:
        eval_model: The model being evaluated.
        ref_model: The reference model whose distribution is treated as ground truth.
        tokenizer: Tokenizer (used for tokenizing examples and debug decoding).
        examples: List of example dicts from load_all_rollouts() or load_lmsys_examples().
        max_length: Max sequence length for tokenization.
        device: Device string (e.g. "cuda").
        debug: If True, log decoded tokens for the first few samples.
        save_output: If True, save results to a JSON file in OUTPUT_DIR.

    Returns:
        EvalResults with aggregate and per-benchmark metrics.
    """
    model_name = getattr(eval_model.config, '_name_or_path', 'unknown_model')
    reference_model_name = getattr(ref_model.config, '_name_or_path', 'unknown_ref')
    # Running sums for memory efficiency (no storing all tokens)
    total_kl = 0.0
    total_match = 0.0
    n_tokens = 0

    # For interesting tokens
    total_kl_interesting = 0.0
    total_match_interesting = 0.0
    n_interesting = 0

    # Per-benchmark stats
    from collections import defaultdict
    bm_stats = defaultdict(lambda: {
        'n_samples': 0, 'n_tokens': 0, 'total_kl': 0.0, 'total_match': 0.0,
        'n_interesting': 0, 'total_kl_interesting': 0.0, 'total_match_interesting': 0.0
    })

    for i, ex in enumerate(tqdm(examples, desc="Evaluating")):
        tokens = tokenize_example(ex, tokenizer, max_length)
        input_ids = tokens['input_ids'].unsqueeze(0).to(device)
        labels = tokens['labels'].unsqueeze(0).to(device)
        benchmark = ex['benchmark']

        with torch.no_grad():
            logits_ref = ref_model(input_ids=input_ids).logits
            logits_model = eval_model(input_ids=input_ids).logits

            metrics = compute_token_metrics(logits_model, logits_ref, labels)
            del logits_ref, logits_model

            # Update running sums
            total_kl += metrics['kl'].sum()
            total_match += metrics['top1_match'].sum()
            n_tokens += len(metrics['kl'])

            # Interesting tokens
            int_mask = metrics['ref_max_prob'] <= INTERESTING_THRESHOLD
            total_kl_interesting += metrics['kl'][int_mask].sum()
            total_match_interesting += metrics['top1_match'][int_mask].sum()
            n_interesting += int_mask.sum()

            # Per-benchmark stats
            bm_stats[benchmark]['n_samples'] += 1
            bm_stats[benchmark]['n_tokens'] += len(metrics['kl'])
            bm_stats[benchmark]['total_kl'] += metrics['kl'].sum()
            bm_stats[benchmark]['total_match'] += metrics['top1_match'].sum()
            bm_stats[benchmark]['n_interesting'] += int_mask.sum()
            bm_stats[benchmark]['total_kl_interesting'] += metrics['kl'][int_mask].sum()
            bm_stats[benchmark]['total_match_interesting'] += metrics['top1_match'][int_mask].sum()

        # Debug: show first few samples
        if debug and i < DEBUG_SAMPLES:
            logger.info(f"\n{'='*60}")
            logger.info(f"DEBUG Sample {i+1} (benchmark: {ex['benchmark']})")
            logger.info(f"{'='*60}")

            # Decode prompt and response
            prompt_len = (labels[0] == -100).sum().item()
            prompt_token_ids = input_ids[0, :prompt_len].tolist()
            response_token_ids = input_ids[0, prompt_len:].tolist()

            # Show last few prompt tokens (should include <think>\n)
            logger.info(f"Prompt ({prompt_len} tokens) - last 5 tokens:")
            for tid in prompt_token_ids[-5:]:
                logger.info(f"  {tid}: {repr(tokenizer.decode([tid]))}")

            logger.info(f"\nResponse ({len(response_token_ids)} tokens) - first 5 tokens:")
            for tid in response_token_ids[:5]:
                logger.info(f"  {tid}: {repr(tokenizer.decode([tid]))}")

            logger.info(f"\nResponse text preview:")
            response_text = tokenizer.decode(response_token_ids)
            logger.info(response_text[:500] + "..." if len(response_text) > 500 else response_text)

            # Per-sample metrics
            n_int_sample = int_mask.sum()
            logger.info(f"\nMetrics for this sample ({len(metrics['kl'])} tokens, {n_int_sample} interesting):")
            logger.info(f"  All tokens:         KL={np.mean(metrics['kl']):.4f}, Top-1 agree={np.mean(metrics['top1_match']):.4f}")
            if n_int_sample > 0:
                logger.info(f"  Interesting tokens: KL={np.mean(metrics['kl'][int_mask]):.4f}, Top-1 agree={np.mean(metrics['top1_match'][int_mask]):.4f}")

    # Compute per-benchmark summaries
    per_benchmark = {}
    for bm, stats in bm_stats.items():
        n = stats['n_tokens']
        n_int = stats['n_interesting']
        per_benchmark[bm] = BenchmarkMetrics(
            n_samples=stats['n_samples'], # pyright: ignore[reportArgumentType]
            n_tokens=n, # pyright: ignore[reportArgumentType]
            kl_mean=stats['total_kl'] / n if n > 0 else float('nan'),
            top1_agreement=stats['total_match'] / n if n > 0 else float('nan'),
            n_interesting=int(n_int),
            kl_mean_interesting=stats['total_kl_interesting'] / n_int if n_int > 0 else float('nan'),
            top1_agreement_interesting=stats['total_match_interesting'] / n_int if n_int > 0 else float('nan'),
        )

    results = EvalResults(
        n_tokens=n_tokens,
        kl_mean=total_kl / n_tokens if n_tokens > 0 else float('nan'),
        top1_agreement=total_match / n_tokens if n_tokens > 0 else float('nan'),
        n_interesting=int(n_interesting),
        kl_mean_interesting=total_kl_interesting / n_interesting if n_interesting > 0 else float('nan'),
        top1_agreement_interesting=total_match_interesting / n_interesting if n_interesting > 0 else float('nan'),
        per_benchmark=per_benchmark,
    )

    # Log results
    logger.info("\n" + "="*70)
    logger.info(f"RESULTS: {model_name}")
    logger.info("="*70)
    logger.info(f"  Samples:           {len(examples)}")
    logger.info(f"  Total tokens:      {results.n_tokens:,}")
    logger.info("")
    logger.info("  ALL TOKENS:")
    logger.info(f"    KL divergence:   {results.kl_mean:.4f}")
    logger.info(f"    Top-1 agreement: {results.top1_agreement:.4f} ({results.top1_agreement*100:.2f}%)")
    logger.info("")
    frac_interesting = results.n_interesting / results.n_tokens * 100
    logger.info(f"  INTERESTING TOKENS (ref max_prob <= {INTERESTING_THRESHOLD}):")
    logger.info(f"    Count:           {results.n_interesting:,} ({frac_interesting:.1f}%)")
    logger.info(f"    KL divergence:   {results.kl_mean_interesting:.4f}")
    logger.info(f"    Top-1 agreement: {results.top1_agreement_interesting:.4f} ({results.top1_agreement_interesting*100:.2f}%)")

    # Per-benchmark stats
    logger.info("\n" + "-"*70)
    logger.info("PER-BENCHMARK STATS:")
    logger.info("-"*70)
    logger.info(f"{'Benchmark':<20} {'Samples':>8} {'Tokens':>10} {'KL':>8} {'Top1':>8} {'KL_int':>8} {'Top1_int':>8}")
    logger.info("-"*70)
    for bm in sorted(results.per_benchmark.keys()):
        stats = results.per_benchmark[bm]
        logger.info(f"{bm:<20} {stats.n_samples:>8} {stats.n_tokens:>10,} {stats.kl_mean:>8.4f} {stats.top1_agreement:>8.2%} {stats.kl_mean_interesting:>8.4f} {stats.top1_agreement_interesting:>8.2%}")
    logger.info("="*70)

    # Save results to JSON
    if save_output:
        from dataclasses import asdict

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        clean_name = model_name.rstrip('/').replace('/', '__')
        output_path = os.path.join(OUTPUT_DIR, f"{clean_name}.json")

        per_benchmark_json = {bm: asdict(m) for bm, m in results.per_benchmark.items()}
        output_data = {
            'model': model_name,
            'reference_model': reference_model_name,
            'n_samples': len(examples),
            'n_tokens': results.n_tokens,
            'interesting_threshold': INTERESTING_THRESHOLD,
            'all_tokens': {
                'kl_mean': float(results.kl_mean),
                'top1_agreement': float(results.top1_agreement),
            },
            'interesting_tokens': {
                'n_tokens': int(results.n_interesting),
                'frac_tokens': frac_interesting / 100,
                'kl_mean': float(results.kl_mean_interesting),
                'top1_agreement': float(results.top1_agreement_interesting),
            },
            'per_benchmark': per_benchmark_json,
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"\nSaved token-level evaluation results to {output_path}")

    return results


#%%
def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Compute token-level metrics vs reference model")
    parser.add_argument("--model", type=str, required=True,
                        help="HF model path or local checkpoint to evaluate. "
                             "Token-level KL and top-1 agreement are computed for this model's "
                             "predictions against the reference model. Use --transcoder if this "
                             "is a transcoder-adapted checkpoint.")
    parser.add_argument("--reference_model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        help="HF model path for the reference model whose output distribution "
                             "is treated as ground truth. KL(reference || model) is computed "
                             "per token. Also used as the tokenizer source. "
                             "(e.g. deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, google/gemma-2-2b-it)")
    parser.add_argument("--data_source", type=str, default="evalchemy_qwen", choices=DATA_SOURCES,
                        help="Validation data source: evalchemy_qwen or lmsys_chat")
    parser.add_argument("--n_samples", type=int, default=N_SAMPLES, help="Number of rollouts to sample")
    parser.add_argument("--max_length", type=int, default=MAX_SEQ_LENGTH, help="Max sequence length")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for sampling")
    parser.add_argument("--debug", action="store_true", help="Show de-tokenized samples for debugging")
    parser.add_argument("--transcoder", action="store_true", help="Load model as transcoder (auto-detects arch)")
    parser.add_argument("--hybrid", action="store_true", help="Disable transcoders (use hybrid model: ref attention + base MLP)")
    args = parser.parse_args()

    reference_model = args.reference_model

    # Load tokenizer
    logger.info(f"Loading tokenizer from {reference_model}")
    tokenizer = AutoTokenizer.from_pretrained(reference_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load examples from chosen data source
    if args.data_source == "evalchemy_qwen":
        eval_dir = EVALCHEMY_DIR / "deepseek-ai__DeepSeek-R1-Distill-Qwen-7B"
        all_examples = load_all_rollouts(str(eval_dir))
    elif args.data_source == "lmsys_chat":
        all_examples = load_lmsys_examples(tokenizer, max_examples=args.n_samples * 2)
    else:
        raise ValueError(f"Unknown data source: {args.data_source}")

    # Sample with deduplication (one rollout per question first)
    if args.n_samples < len(all_examples):
        examples = sample_deduplicated(all_examples, args.n_samples, args.seed)
        n_unique_questions = len(set(ex['question_id'] for ex in examples))
        logger.info(f"\nSampled {len(examples)} examples from {n_unique_questions} unique questions (seed={args.seed})")
    else:
        examples = all_examples
        logger.info(f"\nUsing all {len(examples)} examples")

    # Load models
    logger.info(f"\nLoading reference model: {reference_model}")
    ref_model = AutoModelForCausalLM.from_pretrained(
        reference_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    ref_model.eval()

    logger.info(f"\nLoading eval model: {args.model}")
    eval_model = load_model(args.model, use_transcoder=args.transcoder or args.hybrid)
    
    if args.hybrid:
        logger.info("Disabling transcoders for hybrid model evaluation")
        # Handle both Qwen2 and Gemma2 architectures via their common transcoder interface
        if hasattr(eval_model, "model") and hasattr(eval_model.model, "layers"):
            for layer in eval_model.model.layers:
                if hasattr(layer, "mlp") and hasattr(layer.mlp, "disable_transcoder"):
                    layer.mlp.disable_transcoder = True
        else:
            logger.warning("Could not find layers to disable transcoders. Is this a transcoder model?")

    eval_model.eval()

    # Evaluate
    run_token_metrics_eval(
        eval_model, ref_model, tokenizer, examples, args.max_length, DEVICE,
        debug=args.debug, save_output=True,
    )


#%%
if __name__ == "__main__":
    main()
