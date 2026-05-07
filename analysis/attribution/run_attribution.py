#%%
"""
Batch attribution script for RelP models.

Takes a directory of prompt files, runs attribution on each, and saves
graph files for frontend visualization.

Prompt files should include the target token at the end (it will be dropped
to form the prompt, and used to verify the target).

Output structure (for running multiple models on same prompts):
    output_dir/
    ├── graph-metadata.json
    ├── {run_name}__{prompt_name}.json
    └── features/{scan}/...

Usage:
    python -m analysis.attribution.run_attribution \
        --checkpoint /path/to/model/folded_v2/ \
        --run_name r1_distil \
        --prompts_dir /path/to/prompts/ \
        --output_dir /path/to/output/

After running, start the frontend with:
    circuit-tracer start-server --graph_file_dir {output_dir} --port 8042
"""

import os
import argparse
import subprocess
import sys
from pathlib import Path
from collections.abc import Sequence
from typing import Literal

import torch

from helpers.log import logger, setup_logging

from analysis.attribution.relp_model import RelPReplacementModel

DEEPSEEK_BOS_TOKEN = "<｜begin▁of▁sentence｜>"
DEEPSEEK_USER_TOKEN = "<｜User｜>"
DEEPSEEK_ASSISTANT_TOKEN = "<｜Assistant｜>"
QWEN_IM_START = "<|im_start|>"
QWEN_IM_END = "<|im_end|>"
PromptFormat = Literal["auto", "raw", "chat"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run RelP attribution on multiple prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required args - explicit for accounting
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--run_name", type=str, required=True, help="Name for this run (used in output: {run_name}__{prompt}.json)")
    parser.add_argument("--prompts", type=str, required=True, help="Directory with .txt prompt files, or path to a single .txt file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for graphs")

    # Optional: override scan name (defaults to run_name)
    parser.add_argument("--scan", type=str, default=None, help="Scan name for features (default: run_name)")
    parser.add_argument(
        "--prompt_format",
        type=str,
        choices=["auto", "raw", "chat"],
        default="auto",
        help=(
            "How to tokenize prompt files. 'raw' preserves file text exactly; "
            "'chat' parses marked prompt files and applies the tokenizer chat template; "
            "'auto' uses chat formatting for Gemma2 checkpoints with marked prompt files."
        ),
    )

    # Attribution parameters
    parser.add_argument("--max_n_logits", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_feature_nodes", type=int, default=10000)
    parser.add_argument("--node_threshold", type=float, default=0.8)
    parser.add_argument("--edge_threshold", type=float, default=0.98)

    parser.add_argument("--device", type=str, default="cuda", help="Device (ignored if --device_map is set)")
    parser.add_argument("--device_map", type=str, default=None, help="Device map for multi-GPU. Use 'auto' to split layers across GPUs.")
    parser.add_argument(
        "--auto_shard_gpus",
        action="store_true",
        help=(
            "Detect visible CUDA GPUs and launch one attribution worker per GPU, "
            "with each worker processing a disjoint prompt shard."
        ),
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Total number of prompt shards. Used internally by --auto_shard_gpus, but can also be set manually.",
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="Zero-based prompt shard index for this process.",
    )

    return parser


def _validate_shard_args(num_shards: int, shard_index: int):
    if num_shards < 1:
        raise ValueError(f"--num_shards must be >= 1, got {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(
            f"--shard_index must be in [0, {num_shards}), got {shard_index}"
        )


def _select_shard_items[T](
    items: Sequence[T],
    *,
    num_shards: int,
    shard_index: int,
) -> list[T]:
    _validate_shard_args(num_shards, shard_index)
    if num_shards == 1:
        return list(items)
    return [
        item
        for item_index, item in enumerate(items)
        if item_index % num_shards == shard_index
    ]


def _list_prompt_files(prompts_dir: str | Path) -> list[Path]:
    prompts_path = Path(prompts_dir)
    if not prompts_path.exists():
        raise ValueError(f"Prompts path does not exist: {prompts_dir}")
    if prompts_path.is_file():
        if prompts_path.suffix != ".txt":
            raise ValueError(f"Prompt file must be a .txt file: {prompts_dir}")
        return [prompts_path]
    return sorted(prompts_path.glob("*.txt"))


def _parse_visible_cuda_devices(
    cuda_visible_devices: str | None,
    *,
    device_count: int,
) -> list[str]:
    if device_count <= 0:
        return []
    if cuda_visible_devices is not None and cuda_visible_devices.strip():
        devices = [
            device.strip()
            for device in cuda_visible_devices.split(",")
            if device.strip()
        ]
        if devices == ["-1"]:
            return []
        return devices[:device_count]

    return [str(device_index) for device_index in range(device_count)]


def _build_auto_shard_worker_command(
    args: argparse.Namespace,
    *,
    num_shards: int,
    shard_index: int,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "analysis.attribution.run_attribution",
        "--checkpoint",
        args.checkpoint,
        "--run_name",
        args.run_name,
        "--prompts",
        args.prompts,
        "--output_dir",
        args.output_dir,
        "--prompt_format",
        args.prompt_format,
        "--max_n_logits",
        str(args.max_n_logits),
        "--batch_size",
        str(args.batch_size),
        "--max_feature_nodes",
        str(args.max_feature_nodes),
        "--node_threshold",
        str(args.node_threshold),
        "--edge_threshold",
        str(args.edge_threshold),
        "--device",
        "cuda",
        "--num_shards",
        str(num_shards),
        "--shard_index",
        str(shard_index),
    ]
    if args.scan is not None:
        command.extend(["--scan", args.scan])
    return command


def _run_auto_sharded_attribution(args: argparse.Namespace) -> bool:
    if not args.auto_shard_gpus:
        return False
    if args.num_shards != 1 or args.shard_index != 0:
        raise ValueError("--auto_shard_gpus cannot be combined with manual shard args")
    if args.device_map is not None:
        raise ValueError("--auto_shard_gpus cannot be combined with --device_map")
    if args.device not in {"cuda", "cuda:0"}:
        raise ValueError("--auto_shard_gpus requires --device cuda or --device cuda:0")

    prompt_count = len(_list_prompt_files(args.prompts))
    if prompt_count == 0:
        raise ValueError(f"No .txt prompt files found in {args.prompts}")

    visible_devices = _parse_visible_cuda_devices(
        os.environ.get("CUDA_VISIBLE_DEVICES"),
        device_count=torch.cuda.device_count(),
    )
    if not visible_devices:
        raise RuntimeError("No visible CUDA GPUs found for --auto_shard_gpus")

    num_workers = min(len(visible_devices), prompt_count)
    if num_workers == 1:
        logger.info("--auto_shard_gpus found one usable GPU/prompt; running in this process")
        return False

    logger.info(
        f"Launching {num_workers} attribution workers across {len(visible_devices)} visible GPU(s)"
    )
    processes: list[tuple[int, subprocess.Popen]] = []
    for worker_index, cuda_device in enumerate(visible_devices[:num_workers]):
        command = _build_auto_shard_worker_command(
            args,
            num_shards=num_workers,
            shard_index=worker_index,
        )
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = cuda_device
        logger.info(
            f"Worker {worker_index}: CUDA_VISIBLE_DEVICES={cuda_device}, "
            f"shard {worker_index}/{num_workers}"
        )
        processes.append((worker_index, subprocess.Popen(command, env=env)))

    failures = []
    for worker_index, process in processes:
        returncode = process.wait()
        if returncode != 0:
            failures.append((worker_index, returncode))

    if failures:
        formatted_failures = ", ".join(
            f"worker {worker_index} exited {returncode}"
            for worker_index, returncode in failures
        )
        raise RuntimeError(f"Attribution worker failure(s): {formatted_failures}")

    logger.info("All attribution workers completed successfully")
    logger.info(f"\nTo view graphs run:")
    logger.info(f"  circuit-tracer start-server --graph_file_dir {args.output_dir}")
    return True

#%%
def _parse_chat_prompt_text(text: str) -> tuple[str, str] | None:
    """Parse a raw marked prompt into user and assistant text."""
    if DEEPSEEK_USER_TOKEN in text and DEEPSEEK_ASSISTANT_TOKEN in text:
        if text.startswith(DEEPSEEK_BOS_TOKEN):
            text = text[len(DEEPSEEK_BOS_TOKEN):]
        user_part, assistant_part = text.split(DEEPSEEK_ASSISTANT_TOKEN, 1)
        if DEEPSEEK_USER_TOKEN not in user_part:
            return None
        user_content = user_part.split(DEEPSEEK_USER_TOKEN, 1)[1]
        return user_content, assistant_part

    user_header = f"{QWEN_IM_START}user\n"
    assistant_header = f"{QWEN_IM_START}assistant\n"
    if user_header in text and assistant_header in text:
        user_part, assistant_part = text.split(assistant_header, 1)
        user_content = user_part.split(user_header, 1)[1]
        if user_content.endswith(QWEN_IM_END + "\n"):
            user_content = user_content[:-(len(QWEN_IM_END) + 1)]
        elif user_content.endswith(QWEN_IM_END):
            user_content = user_content[:-len(QWEN_IM_END)]
        if assistant_part.endswith(QWEN_IM_END + "\n"):
            assistant_part = assistant_part[:-(len(QWEN_IM_END) + 1)]
        elif assistant_part.endswith(QWEN_IM_END):
            assistant_part = assistant_part[:-len(QWEN_IM_END)]
        return user_content, assistant_part

    return None


def _should_chat_format_prompt(
    text: str,
    prompt_format: PromptFormat,
    model_type: str | None,
) -> bool:
    if prompt_format == "raw":
        return False
    if prompt_format == "chat":
        return True
    return model_type == "gemma2" and _parse_chat_prompt_text(text) is not None


def _load_chat_formatted_prompt(
    text: str,
    tokenizer,
) -> tuple[list[int], int, str]:
    parsed = _parse_chat_prompt_text(text)
    if parsed is None:
        raise ValueError(
            "Prompt format 'chat' requires DeepSeek/Qwen-style user and assistant markers"
        )

    user_content, assistant_content = parsed
    assistant_ids = tokenizer.encode(assistant_content, add_special_tokens=False)
    if not assistant_ids:
        raise ValueError("Assistant prompt content must include a target token")

    target_token = assistant_ids[-1]
    assistant_prefix = tokenizer.decode(assistant_ids[:-1])
    prompt_tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=True,
        add_generation_prompt=True,
    )
    prompt_tokens = list(prompt_tokens) + tokenizer.encode(
        assistant_prefix,
        add_special_tokens=False,
    )
    prompt_str = tokenizer.decode(prompt_tokens)
    return prompt_tokens, target_token, prompt_str


def load_prompt_file(
    path: Path,
    tokenizer,
    prompt_format: PromptFormat = "auto",
    model_type: str | None = None,
) -> tuple[list[int], int, str]:
    """
    Load a prompt file where the last token is the target.

    Returns:
        prompt_tokens: Token IDs for the prompt (excluding target)
        target_token: The target token ID
        prompt_str: Decoded prompt string
    """
    text = path.read_text()
    if _should_chat_format_prompt(text, prompt_format, model_type):
        return _load_chat_formatted_prompt(text, tokenizer)
    if prompt_format == "auto":
        logger.warning(
            "Prompt format auto did not detect chat markers in %s; falling back to raw tokenization",
            path,
        )

    full_tokens = tokenizer.encode(text, add_special_tokens=False)
    if not full_tokens:
        raise ValueError(f"Prompt file is empty after tokenization: {path}")
    prompt_tokens = full_tokens[:-1]
    target_token = full_tokens[-1]
    prompt_str = tokenizer.decode(prompt_tokens)
    return prompt_tokens, target_token, prompt_str


def load_prompts(
    prompts_dir: str,
    tokenizer,
    prompt_format: PromptFormat = "auto",
    model_type: str | None = None,
    num_shards: int = 1,
    shard_index: int = 0,
) -> dict[str, dict]:
    """
    Load prompts from a directory of .txt files.

    Returns dict mapping slug to {tokens, target, text}.
    """
    prompts = {}
    txt_files = _list_prompt_files(prompts_dir)
    if not txt_files:
        raise ValueError(f"No .txt files found in {prompts_dir}")

    for txt_file in _select_shard_items(
        txt_files,
        num_shards=num_shards,
        shard_index=shard_index,
    ):
        slug = txt_file.stem
        tokens, target, text = load_prompt_file(
            txt_file,
            tokenizer,
            prompt_format=prompt_format,
            model_type=model_type,
        )
        prompts[slug] = {
            "tokens": tokens,
            "target": target,
            "text": text,
        }
        target_str = tokenizer.decode([target])
        logger.info(f"  {slug}: {len(tokens)} tokens, target={target_str!r}")

    return prompts


def run_attribution_for_prompt(
    prompt_tokens: list[int],
    slug: str,
    model: RelPReplacementModel,
    scan: str,
    output_dir: str,
    max_n_logits: int,
    batch_size: int,
    max_feature_nodes: int,
    node_threshold: float,
    edge_threshold: float,
):
    """Run attribution for a single prompt and save graph files."""
    import gc
    from analysis.attribution.attribute import attribute
    from circuit_tracer.utils.create_graph_files import create_graph_files

    logger.info(f"  Running attribution (batch_size={batch_size})...")
    raw_graph = attribute(
        prompt_tokens,
        model,
        max_n_logits=max_n_logits,
        max_feature_nodes=max_feature_nodes,
        batch_size=batch_size,
        verbose=True,
    )

    logger.info(f"  Graph: {raw_graph.active_features.shape[0]} active, "
               f"{raw_graph.selected_features.shape[0]} selected")

    # BOS zeroing is done inline in attribute.py
    graph = raw_graph

    # Free up GPU memory before pruning (prune_graph needs to sort large edge matrix)
    graph.to("cpu")
    # Clear model's cached features and gradients
    for layer in model.model.model.layers:
        if hasattr(layer.mlp, 'cached_features'):
            layer.mlp.cached_features = None
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f"  GPU memory after cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Save graph files - if pruning OOMs, save raw graph instead
    try:
        create_graph_files(
            graph_or_path=graph,
            slug=slug,
            scan=scan,
            output_path=output_dir,
            node_threshold=node_threshold,
            edge_threshold=edge_threshold,
        )
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" not in str(e).lower() and "CUDA" not in str(e):
            raise
        # Save raw graph so we don't lose the expensive backward passes
        raw_graph_path = Path(output_dir) / f"{slug}_raw.pt"
        logger.info(f"  OOM during pruning, saving raw graph to {raw_graph_path}")
        graph.to_pt(str(raw_graph_path))
        logger.info(f"  Raw graph saved. Prune later with: create_graph_files('{raw_graph_path}', ...)")

    return graph


#%%
def main():
    parser = build_parser()
    args = parser.parse_args()
    setup_logging()
    try:
        _validate_shard_args(args.num_shards, args.shard_index)
        if _run_auto_sharded_attribution(args):
            return
    except Exception as e:
        parser.error(str(e))

    # Default scan to run_name
    if args.scan is None:
        args.scan = args.run_name

    # Print config
    logger.info("=" * 60)
    logger.info("RelP Attribution")
    logger.info("=" * 60)
    logger.info(f"Run name:    {args.run_name}")
    logger.info(f"Checkpoint:  {args.checkpoint}")
    logger.info(f"Prompts:     {args.prompts}")
    logger.info(f"Prompt fmt:  {args.prompt_format}")
    logger.info(f"Output:      {args.output_dir}")
    logger.info(f"Scan:        {args.scan}")
    if args.num_shards > 1:
        logger.info(f"Shard:       {args.shard_index}/{args.num_shards}")
    logger.info("=" * 60)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    logger.info("\nLoading model...")
    model = RelPReplacementModel.from_pretrained(
        args.checkpoint,
        device=args.device,
        device_map=args.device_map,
        dtype=torch.bfloat16,
    )
    logger.info(f"Loaded: {model.cfg.n_layers} layers, {model.cfg.d_model} d_model, "
               f"{model.cfg.n_features} features")
    model_type = getattr(model.hf_config, "model_type", None)

    # Load prompts (single file or directory)
    logger.info("\nLoading prompts...")
    prompts_path = Path(args.prompts)
    if prompts_path.is_file():
        if _select_shard_items(
            [prompts_path],
            num_shards=args.num_shards,
            shard_index=args.shard_index,
        ):
            slug = prompts_path.stem
            tokens, target, text = load_prompt_file(
                prompts_path,
                model.tokenizer,
                prompt_format=args.prompt_format,
                model_type=model_type,
            )
            prompts = {slug: {"tokens": tokens, "target": target, "text": text}}
            target_str = model.tokenizer.decode([target])
            logger.info(f"  {slug}: {len(tokens)} tokens, target={target_str!r}")
        else:
            prompts = {}
    else:
        prompts = load_prompts(
            args.prompts,
            model.tokenizer,
            prompt_format=args.prompt_format,
            model_type=model_type,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
        )
    prompt_items = list(prompts.items())
    logger.info(f"Found {len(prompt_items)} prompt(s)")

    # Run attribution
    logger.info("\n" + "=" * 60)
    logger.info("Running Attribution")
    logger.info("=" * 60)

    results = {}
    for i, (prompt_name, data) in enumerate(prompt_items, 1):
        slug = f"{args.run_name}__{prompt_name}"
        graph_path = Path(args.output_dir) / f"{slug}.json"

        # Skip if graph already exists
        if graph_path.exists():
            logger.info(f"\n[{i}/{len(prompt_items)}] {slug} - SKIPPED (already exists)")
            results[prompt_name] = "skipped"
            continue

        logger.info(f"\n[{i}/{len(prompt_items)}] {slug}")
        prompt_text: str = data['text']  # type: ignore[assignment]
        prompt_tokens: list[int] = data["tokens"]  # type: ignore[assignment]
        logger.info(f"  Last 60 chars: ...{prompt_text[-60:]!r}")
        logger.info(f"  Target: {model.tokenizer.decode([data['target']])!r}")

        try:
            graph = run_attribution_for_prompt(
                prompt_tokens=prompt_tokens,
                slug=slug,
                model=model,
                scan=args.scan,
                output_dir=args.output_dir,
                max_n_logits=args.max_n_logits,
                batch_size=args.batch_size,
                max_feature_nodes=args.max_feature_nodes,
                node_threshold=args.node_threshold,
                edge_threshold=args.edge_threshold,
            )
            results[prompt_name] = "success"
            logger.info(f"  Done: {slug}.json")
        except Exception as e:
            results[prompt_name] = f"error: {e}"
            logger.exception("  Error while running attribution for %s: %s", slug, e)

            # Clean up GPU memory after OOM to allow recovery
            if "CUDA" in str(e) or "out of memory" in str(e).lower():
                logger.info("  Attempting CUDA memory cleanup...")
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                logger.info(f"  GPU memory after cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    success = sum(1 for v in results.values() if v == "success")
    skipped = sum(1 for v in results.values() if v == "skipped")
    errors = len(prompt_items) - success - skipped
    logger.info(f"Success: {success}, Skipped: {skipped}, Errors: {errors} (total: {len(prompt_items)})")

    logger.info(f"\nTo view graphs run:")
    logger.info(f"  circuit-tracer start-server --graph_file_dir {args.output_dir}")


#%%
if __name__ == "__main__":
    main()
