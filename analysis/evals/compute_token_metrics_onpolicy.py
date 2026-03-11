#!/usr/bin/env python3
#%%
"""
Compute token-level metrics (KL divergence, top-1 agreement) for a model
against R1-Distill reference using on-policy rollouts from evalchemy.

Usage:
    python claude_scripts/compute_token_metrics_onpolicy.py --model <model_path>
    python claude_scripts/compute_token_metrics_onpolicy.py --model Qwen/Qwen2.5-Math-7B
"""
from pathlib import Path
import sys
sys.path.insert(0, '/juice2/u/nathu/sparse_adaptation')

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
import glob
import re
import torch
import numpy as np
from dataclasses import dataclass
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
    per_benchmark: dict[str, BenchmarkMetrics] = None

# Output directory
OUTPUT_DIR = "/nlp/scr/nathu/sparse-adaptation/token_recon_evals"

#%%
# ============================================================================
# CONFIG
# ============================================================================

EVALCHEMY_DIR = Path("/nlp/scr/nathu/sparse-adaptation/evalchemy_v5")

# Per-family configuration: reference model, eval directory, transcoder loader
MODEL_FAMILIES = {
    "qwen": {
        "reference_model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "eval_dir": EVALCHEMY_DIR / "deepseek-ai__DeepSeek-R1-Distill-Qwen-7B",
    },
    "gemma2": {
        "reference_model": "google/gemma-2-2b-it",
        "eval_dir": EVALCHEMY_DIR / "google__gemma-2-2b-it",
    },
}

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

    print(f"Loading rollouts from {eval_dir}:")
    for bm_name, path in sorted(benchmark_files.items()):
        print(f"  {bm_name}: {os.path.basename(path)}")

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

    print(f"\nLoaded {len(examples)} total rollouts")

    # Print per-benchmark counts
    from collections import Counter
    bm_counts = Counter(ex['benchmark'] for ex in examples)
    for bm, count in sorted(bm_counts.items()):
        print(f"  {bm}: {count}")

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
    """Tokenize a single example using DeepSeek chat format."""
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
def compute_token_metrics(logits_model, logits_ref, input_ids, labels):
    """
    Compute per-token KL divergence and top-1 agreement.
    Returns metrics only for response tokens (labels != -100).
    Also returns ref model's max prob for filtering "interesting" tokens.
    """
    # Shift: logits[i] predicts token[i+1]
    logits_model = logits_model[:, :-1, :]
    logits_ref = logits_ref[:, :-1, :]
    labels_shifted = labels[:, 1:]

    # Response token mask
    response_mask = labels_shifted != -100

    # Softmax / log softmax
    p_ref = torch.softmax(logits_ref.float(), dim=-1)
    log_p_ref = torch.log_softmax(logits_ref.float(), dim=-1)
    log_p_model = torch.log_softmax(logits_model.float(), dim=-1)

    # Reference model's max probability (for filtering interesting tokens)
    max_prob_ref = p_ref.max(dim=-1).values

    # KL divergence: KL(ref || model)
    kl_per_token = (p_ref * (log_p_ref - log_p_model)).sum(dim=-1)

    # Top-1 agreement
    top1_ref = logits_ref.argmax(dim=-1)
    top1_model = logits_model.argmax(dim=-1)
    top1_match = (top1_ref == top1_model)

    # Filter to response tokens
    mask = response_mask[0]
    return {
        'kl': kl_per_token[0, mask].cpu().numpy(),
        'top1_match': top1_match[0, mask].cpu().numpy(),
        'ref_max_prob': max_prob_ref[0, mask].cpu().numpy(),
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
        print(f"  Loaded as transcoder model (auto-detected arch)")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        print(f"  Loaded as AutoModelForCausalLM")

    return model


def evaluate_model(model_path: str, ref_model, tokenizer, examples, max_length: int, device: str, debug: bool = False, use_transcoder: bool = False):
    """Evaluate a model against the reference."""
    print(f"\nLoading model: {model_path}")
    model = load_model(model_path, use_transcoder=use_transcoder)
    model.eval()

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
            logits_model = model(input_ids=input_ids).logits

            metrics = compute_token_metrics(logits_model, logits_ref, input_ids, labels)

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
            print(f"\n{'='*60}")
            print(f"DEBUG Sample {i+1} (benchmark: {ex['benchmark']})")
            print(f"{'='*60}")

            # Decode prompt and response
            prompt_len = (labels[0] == -100).sum().item()
            prompt_token_ids = input_ids[0, :prompt_len].tolist()
            response_token_ids = input_ids[0, prompt_len:].tolist()

            # Show last few prompt tokens (should include <think>\n)
            print(f"Prompt ({prompt_len} tokens) - last 5 tokens:")
            for tid in prompt_token_ids[-5:]:
                print(f"  {tid}: {repr(tokenizer.decode([tid]))}")

            print(f"\nResponse ({len(response_token_ids)} tokens) - first 5 tokens:")
            for tid in response_token_ids[:5]:
                print(f"  {tid}: {repr(tokenizer.decode([tid]))}")

            print(f"\nResponse text preview:")
            response_text = tokenizer.decode(response_token_ids)
            print(response_text[:500] + "..." if len(response_text) > 500 else response_text)

            # Per-sample metrics
            n_int_sample = int_mask.sum()
            print(f"\nMetrics for this sample ({len(metrics['kl'])} tokens, {n_int_sample} interesting):")
            print(f"  All tokens:         KL={np.mean(metrics['kl']):.4f}, Top-1 agree={np.mean(metrics['top1_match']):.4f}")
            if n_int_sample > 0:
                print(f"  Interesting tokens: KL={np.mean(metrics['kl'][int_mask]):.4f}, Top-1 agree={np.mean(metrics['top1_match'][int_mask]):.4f}")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    # Compute per-benchmark summaries
    per_benchmark = {}
    for bm, stats in bm_stats.items():
        n = stats['n_tokens']
        n_int = stats['n_interesting']
        per_benchmark[bm] = BenchmarkMetrics(
            n_samples=stats['n_samples'],
            n_tokens=n,
            kl_mean=stats['total_kl'] / n if n > 0 else float('nan'),
            top1_agreement=stats['total_match'] / n if n > 0 else float('nan'),
            n_interesting=int(n_int),
            kl_mean_interesting=stats['total_kl_interesting'] / n_int if n_int > 0 else float('nan'),
            top1_agreement_interesting=stats['total_match_interesting'] / n_int if n_int > 0 else float('nan'),
        )

    # Return summary stats
    return EvalResults(
        n_tokens=n_tokens,
        kl_mean=total_kl / n_tokens if n_tokens > 0 else float('nan'),
        top1_agreement=total_match / n_tokens if n_tokens > 0 else float('nan'),
        n_interesting=int(n_interesting),
        kl_mean_interesting=total_kl_interesting / n_interesting if n_interesting > 0 else float('nan'),
        top1_agreement_interesting=total_match_interesting / n_interesting if n_interesting > 0 else float('nan'),
        per_benchmark=per_benchmark,
    )


#%%
def main():
    parser = argparse.ArgumentParser(description="Compute token-level metrics vs reference model")
    parser.add_argument("--model", type=str, required=True, help="Model path to evaluate")
    parser.add_argument("--model_family", type=str, default="qwen", choices=list(MODEL_FAMILIES.keys()),
                        help="Model family (determines reference model and eval dir)")
    parser.add_argument("--n_samples", type=int, default=N_SAMPLES, help="Number of rollouts to sample")
    parser.add_argument("--max_length", type=int, default=MAX_SEQ_LENGTH, help="Max sequence length")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for sampling")
    parser.add_argument("--debug", action="store_true", help="Show de-tokenized samples for debugging")
    parser.add_argument("--transcoder", action="store_true", help="Load model as transcoder (auto-detects arch)")
    args = parser.parse_args()

    family_config = MODEL_FAMILIES[args.model_family]
    reference_model = family_config["reference_model"]
    eval_dir = family_config["eval_dir"]

    # Load tokenizer
    print(f"Loading tokenizer from {reference_model}")
    tokenizer = AutoTokenizer.from_pretrained(reference_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load all rollouts
    all_examples = load_all_rollouts(eval_dir)

    # Sample with deduplication (one rollout per question first)
    if args.n_samples < len(all_examples):
        examples = sample_deduplicated(all_examples, args.n_samples, args.seed)
        n_unique_questions = len(set(ex['question_id'] for ex in examples))
        print(f"\nSampled {len(examples)} examples from {n_unique_questions} unique questions (seed={args.seed})")
    else:
        examples = all_examples
        print(f"\nUsing all {len(examples)} examples")

    # Load reference model
    print(f"\nLoading reference model: {reference_model}")
    ref_model = AutoModelForCausalLM.from_pretrained(
        reference_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    ref_model.eval()

    # Evaluate
    results = evaluate_model(
        args.model, ref_model, tokenizer, examples, args.max_length, DEVICE,
        debug=args.debug, use_transcoder=args.transcoder
    )

    # Print results
    print("\n" + "="*70)
    print(f"RESULTS: {args.model}")
    print("="*70)
    print(f"  Samples:           {len(examples)}")
    print(f"  Total tokens:      {results.n_tokens:,}")
    print()
    print(f"  ALL TOKENS:")
    print(f"    KL divergence:   {results.kl_mean:.4f}")
    print(f"    Top-1 agreement: {results.top1_agreement:.4f} ({results.top1_agreement*100:.2f}%)")
    print()
    frac_interesting = results.n_interesting / results.n_tokens * 100
    print(f"  INTERESTING TOKENS (ref max_prob <= {INTERESTING_THRESHOLD}):")
    print(f"    Count:           {results.n_interesting:,} ({frac_interesting:.1f}%)")
    print(f"    KL divergence:   {results.kl_mean_interesting:.4f}")
    print(f"    Top-1 agreement: {results.top1_agreement_interesting:.4f} ({results.top1_agreement_interesting*100:.2f}%)")

    # Per-benchmark stats
    print("\n" + "-"*70)
    print("PER-BENCHMARK STATS:")
    print("-"*70)
    print(f"{'Benchmark':<20} {'Samples':>8} {'Tokens':>10} {'KL':>8} {'Top1':>8} {'KL_int':>8} {'Top1_int':>8}")
    print("-"*70)
    for bm in sorted(results.per_benchmark.keys()):
        stats = results.per_benchmark[bm]
        print(f"{bm:<20} {stats.n_samples:>8} {stats.n_tokens:>10,} {stats.kl_mean:>8.4f} {stats.top1_agreement:>8.2%} {stats.kl_mean_interesting:>8.4f} {stats.top1_agreement_interesting:>8.2%}")
    print("="*70)

    # Save results to JSON
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Clean model name for filename
    model_name = args.model.rstrip('/').replace('/', '__')
    suffix = "_debug" if args.debug else ""
    output_path = os.path.join(OUTPUT_DIR, f"{model_name}{suffix}.json")

    # Convert per_benchmark to JSON-serializable dicts
    from dataclasses import asdict
    per_benchmark_json = {bm: asdict(m) for bm, m in results.per_benchmark.items()}

    output_data = {
        'model': args.model,
        'reference_model': reference_model,
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
    print(f"\nSaved results to {output_path}")


#%%
if __name__ == "__main__":
    main()
