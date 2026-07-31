"""Compare Gemma2 transcoder generation through HF Transformers and vLLM."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from helpers.log import logger, setup_logging
from models.auto import AutoModelForCausalLMWithTranscoder, load_tokenizer
from models.steering import FeatureSteeringSpec
from models.vllm_steering import feature_steering_hf_overrides


@dataclass
class GenerationResult:
    texts: list[str]
    token_ids: list[list[int]]
    load_seconds: float
    generate_seconds: float

    @property
    def n_tokens(self) -> int:
        return sum(len(ids) for ids in self.token_ids)

    @property
    def tokens_per_second(self) -> float:
        if self.generate_seconds == 0:
            return float("inf")
        return self.n_tokens / self.generate_seconds


def parse_feature_steering_spec(value: str) -> FeatureSteeringSpec:
    try:
        cantor_id_text, strength_text = value.split(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected CANTOR_ID:STRENGTH") from exc

    try:
        return FeatureSteeringSpec(int(cantor_id_text), float(strength_text))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def load_prompts(prompt_args: list[str], prompt_file: str | None) -> list[str]:
    prompts = list(prompt_args)
    if prompt_file is not None:
        path = Path(prompt_file)
        prompts.extend(line.rstrip("\n") for line in path.read_text().splitlines() if line.strip())
    if not prompts:
        prompts = ["What is the capital of France?"]
    return prompts


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def run_hf(args: argparse.Namespace, prompts: list[str]) -> GenerationResult:
    load_start = time.perf_counter()
    tokenizer = load_tokenizer(args.model_path, tokenizer_path=args.tokenizer)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLMWithTranscoder.from_pretrained(
        args.model_path,
        torch_dtype=_dtype_from_name(args.dtype),
        device_map=args.hf_device_map,
    )
    model.eval()
    if args.steer:
        model.set_feature_steering(args.steer, mode=args.steering_mode)
    load_seconds = time.perf_counter() - load_start

    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.temperature == 0:
        generate_kwargs["do_sample"] = False
    else:
        generate_kwargs.update(
            {
                "do_sample": True,
                "temperature": args.temperature,
                "top_p": args.top_p,
            }
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    generate_start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generate_kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    generate_seconds = time.perf_counter() - generate_start

    prompt_len = inputs["input_ids"].shape[1]
    new_token_ids = [row[prompt_len:].tolist() for row in output_ids]
    texts = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)
    return GenerationResult(texts, new_token_ids, load_seconds, generate_seconds)


def run_vllm(args: argparse.Namespace, prompts: list[str]) -> GenerationResult:
    from models.gemma2_transcoder_vllm import register_gemma2_vllm_transcoder

    register_gemma2_vllm_transcoder()

    from vllm import LLM, SamplingParams

    hf_overrides = feature_steering_hf_overrides(args.steer, args.steering_mode) if args.steer else {}

    load_start = time.perf_counter()
    llm = LLM(
        model=args.model_path,
        tokenizer=args.tokenizer,
        trust_remote_code=True,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=args.seed,
        hf_overrides=hf_overrides,
    )
    load_seconds = time.perf_counter() - load_start

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    generate_start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    generate_seconds = time.perf_counter() - generate_start

    texts = [output.outputs[0].text for output in outputs]
    token_ids = [list(output.outputs[0].token_ids) for output in outputs]
    return GenerationResult(texts, token_ids, load_seconds, generate_seconds)


def log_result(name: str, result: GenerationResult) -> None:
    logger.info(
        "%s: load=%.2fs generate=%.2fs tokens=%s throughput=%.2f tok/s",
        name,
        result.load_seconds,
        result.generate_seconds,
        result.n_tokens,
        result.tokens_per_second,
    )


def compare_results(hf: GenerationResult, vllm: GenerationResult) -> None:
    exact_tokens = hf.token_ids == vllm.token_ids
    exact_text = hf.texts == vllm.texts
    logger.info("Exact token match: %s", exact_tokens)
    logger.info("Exact text match: %s", exact_text)

    for idx, (hf_ids, vllm_ids, hf_text, vllm_text) in enumerate(
        zip(hf.token_ids, vllm.token_ids, hf.texts, vllm.texts, strict=True)
    ):
        if hf_ids == vllm_ids and hf_text == vllm_text:
            continue
        first_diff = next(
            (i for i, (left, right) in enumerate(zip(hf_ids, vllm_ids)) if left != right),
            min(len(hf_ids), len(vllm_ids)),
        )
        logger.warning(
            "Prompt %s differs at generated token %s (hf_len=%s, vllm_len=%s)",
            idx,
            first_diff,
            len(hf_ids),
            len(vllm_ids),
        )
        logger.warning("HF text: %r", hf_text)
        logger.warning("vLLM text: %r", vllm_text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", help="Gemma2 transcoder checkpoint path or HF repo ID")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path; defaults to model_path/base fallback")
    parser.add_argument("--prompt", action="append", default=[], help="Prompt to run; repeat for batches")
    parser.add_argument("--prompt_file", default=None, help="One prompt per non-empty line")
    parser.add_argument("--engine", choices=["both", "hf", "vllm"], default="both")
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hf_device_map", default="auto")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--steer", action="append", default=[], type=parse_feature_steering_spec,
                        metavar="CANTOR_ID:STRENGTH")
    parser.add_argument("--steering_mode", choices=["min", "add", "set"], default="min")
    args = parser.parse_args()

    setup_logging()
    prompts = load_prompts(args.prompt, args.prompt_file)
    logger.info("Loaded %s prompt(s)", len(prompts))
    if args.temperature != 0:
        logger.warning("Nonzero temperature can make HF/vLLM exact-match comparisons fail by design")

    hf_result = run_hf(args, prompts) if args.engine in ("both", "hf") else None
    vllm_result = run_vllm(args, prompts) if args.engine in ("both", "vllm") else None

    if hf_result is not None:
        log_result("HF", hf_result)
    if vllm_result is not None:
        log_result("vLLM", vllm_result)
    if hf_result is not None and vllm_result is not None:
        compare_results(hf_result, vllm_result)


if __name__ == "__main__":
    main()
