"""Load a trained transcoder model and chat with it interactively.

Usage:
    # From a HuggingFace repo:
    python -m analysis.simple_load.simple_load siddharthmb/2026.TA.gemma2_2b_tc8192_...

    # From a local checkpoint:
    python -m analysis.simple_load.simple_load /path/to/checkpoint

    # Specify tokenizer separately (if not saved with checkpoint):
    python -m analysis.simple_load.simple_load siddharthmb/2026.TA.gemma2_2b_... \
        --tokenizer google/gemma-2-2b
"""

import argparse

import torch
from transformers import AutoTokenizer, TextStreamer

from helpers.log import logger, setup_logging
from models import get_transcoder_classes, detect_architecture


def load_model(model_path: str, tokenizer_path: str | None = None, arch: str | None = None):
    """Load a transcoder model and tokenizer.

    Args:
        model_path: HF repo ID or local path to checkpoint.
        tokenizer_path: HF repo ID or local path for the tokenizer.
            If None, tries loading from model_path first, then falls back
            to the base model name from the config.
        arch: Architecture name (e.g. "gemma2", "qwen2"). Auto-detected if None.
    """
    if arch is None:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        arch = detect_architecture(config._name_or_path or model_path)

    _, ModelCls = get_transcoder_classes(arch)

    model = ModelCls.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # Try loading tokenizer from checkpoint, fall back to base model
    if tokenizer_path is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except OSError:
            base = model.config._name_or_path
            logger.info(f"No tokenizer in checkpoint, loading from base model: {base}")
            tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    return model, tokenizer


def chat(model, tokenizer, use_chat_template: bool = True, show_special_tokens: bool = False):
    """Interactive multi-turn chat loop.

    Args:
        use_chat_template: If True, wraps input in the tokenizer's chat template
            (user/assistant turns). If False, sends raw text directly.
        show_special_tokens: If True, prints special tokens (e.g. <bos>, <start_of_turn>)
            in both the formatted prompt and the model output.
    """
    skip_special = not show_special_tokens
    streamer = TextStreamer(tokenizer, skip_special_tokens=skip_special, skip_prompt=True)
    messages = []

    mode = "chat" if use_chat_template else "raw"
    logger.info(f"\nModel loaded (mode={mode}, show_special_tokens={show_special_tokens}).")
    logger.info("Type your message (Ctrl+C to exit, /clear to reset).\n")

    while True:
        try:
            user_input = input(">>> ")
        except (KeyboardInterrupt, EOFError):
            logger.info("")
            break

        if not user_input.strip():
            continue

        if user_input.strip() == "/clear":
            messages.clear()
            logger.info("Conversation cleared.\n")
            continue

        if use_chat_template:
            messages.append({"role": "user", "content": user_input})
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt",
            ).to(model.device)
        else:
            input_ids = tokenizer(user_input, return_tensors="pt").input_ids.to(model.device)

        if show_special_tokens:
            logger.info(f"\n[PROMPT] {tokenizer.decode(input_ids[0], skip_special_tokens=False)}")
            logger.info("[OUTPUT] ")

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=512,
                streamer=streamer,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        # Extract only the new tokens for conversation history
        new_tokens = output_ids[0][input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        if use_chat_template:
            messages.append({"role": "assistant", "content": response})

        logger.info("")


def main():
    parser = argparse.ArgumentParser(description="Load and chat with a transcoder model")
    parser.add_argument("model_path", help="HF repo ID or local path to checkpoint")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path (defaults to model_path)")
    parser.add_argument("--arch", default=None, choices=["gemma2", "qwen2"],
                        help="Architecture (auto-detected if not set)")
    parser.add_argument("--raw", action="store_true",
                        help="Send raw text without chat template")
    parser.add_argument("--show_special_tokens", action="store_true", default=True,
                        help="Show special tokens in prompt and output")
    args = parser.parse_args()

    setup_logging()
    model, tokenizer = load_model(args.model_path, args.tokenizer, args.arch)
    chat(model, tokenizer, use_chat_template=not args.raw, show_special_tokens=args.show_special_tokens)


if __name__ == "__main__":
    main()
