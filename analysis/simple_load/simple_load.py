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
import sys

import torch
from transformers import TextStreamer

from helpers.log import logger, setup_logging
from models import checkpoint_load_kwargs_for_model_type, get_transcoder_classes, detect_architecture
from models.auto import load_tokenizer
from models.steering import FeatureSteeringSpec, cantor_unpair


def load_model(model_path: str, tokenizer_path: str | None = None, arch: str | None = None):
    """Load a transcoder model and tokenizer.

    Args:
        model_path: HF repo ID or local path to checkpoint.
        tokenizer_path: HF repo ID or local path for the tokenizer.
            If None, resolved from model_type via the canonical base tokenizer.
        arch: Architecture name (e.g. "gemma2", "qwen2"). Auto-detected if None.
    """
    if arch is None:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        arch = detect_architecture(config._name_or_path or model_path)

    _, ModelCls = get_transcoder_classes(arch)
    load_kwargs = checkpoint_load_kwargs_for_model_type(arch)

    model = ModelCls.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        **load_kwargs,
    )
    model.eval()

    tokenizer = load_tokenizer(model_path, tokenizer_path=tokenizer_path)

    return model, tokenizer


def extract_input_ids(encoded) -> torch.Tensor:
    """Return the input_ids tensor from tokenizer outputs."""
    if isinstance(encoded, torch.Tensor):
        return encoded
    if hasattr(encoded, "input_ids"):
        return encoded.input_ids
    if isinstance(encoded, dict) and "input_ids" in encoded:
        return encoded["input_ids"]
    raise TypeError(f"Could not extract input_ids from {type(encoded).__name__}")


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
            input_ids = extract_input_ids(tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt",
            )).to(model.device)
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


def run_prompt(
    model,
    tokenizer,
    prompt: str,
    use_chat_template: bool = True,
    show_special_tokens: bool = False,
) -> str:
    """Generate a single response for a prompt.

    Args:
        prompt: Prompt text to send to the model.
        use_chat_template: If True, wraps input as a single user turn with an
            assistant generation prompt. If False, sends raw text directly.
        show_special_tokens: If True, includes special tokens in the decoded
            model output.
    """
    if use_chat_template:
        input_ids = extract_input_ids(tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        )).to(model.device)
    else:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    if show_special_tokens:
        logger.info(f"\n[PROMPT] {tokenizer.decode(input_ids[0], skip_special_tokens=False)}")
        logger.info("[OUTPUT]")

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    new_tokens = output_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=not show_special_tokens)


def parse_feature_steering_spec(value: str) -> FeatureSteeringSpec:
    """Parse a CANTOR_ID:STRENGTH steering CLI value."""
    try:
        cantor_id_text, strength_text = value.split(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected CANTOR_ID:STRENGTH") from exc

    try:
        return FeatureSteeringSpec(int(cantor_id_text), float(strength_text))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def log_feature_steering_targets(model, specs: list[FeatureSteeringSpec]) -> None:
    """Log decoded steering targets and warn for targets with no decoder effect."""
    if not specs or not hasattr(model, "_transcoder_mlps"):
        return

    try:
        mlps = tuple(model._transcoder_mlps())
    except Exception as exc:
        logger.warning(f"Could not inspect feature steering targets: {exc}")
        return

    for spec in specs:
        layer_idx, feature_idx = cantor_unpair(spec.cantor_id)
        if layer_idx >= len(mlps):
            continue

        mlp = mlps[layer_idx]
        n_features = getattr(mlp, "n_features", None)
        decoder = getattr(mlp, "transcoder_dec", None)
        weight = getattr(decoder, "weight", None)
        if weight is None or n_features is None or feature_idx >= n_features:
            continue

        column_norm = weight[:, feature_idx].detach().float().norm().item()
        logger.info(
            f"Feature steering target {spec.cantor_id} -> layer {layer_idx}, "
            f"feature {feature_idx}/{n_features}, strength={spec.strength:g}, "
            f"decoder_column_norm={column_norm:.6g}"
        )
        if column_norm <= 1e-8:
            logger.warning(
                f"Feature steering target {spec.cantor_id} has near-zero decoder column norm; "
                "steering it may have no visible effect."
            )


def main():
    parser = argparse.ArgumentParser(description="Load and chat with a transcoder model")
    parser.add_argument("model_path", help="HF repo ID or local path to checkpoint")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path (defaults to model_path)")
    parser.add_argument("--arch", default=None, choices=["gemma2", "gemma4", "qwen2"],
                        help="Architecture (auto-detected if not set)")
    parser.add_argument("--raw", action="store_true",
                        help="Send raw text without chat template")
    parser.add_argument("--prompt", default=None,
                        help="Run this prompt once, write the generated response to stdout, and exit")
    parser.add_argument("--show_special_tokens", action="store_true", default=True,
                        help="Show special tokens in prompt and output")
    parser.add_argument("--hybrid", action="store_true",
                        help="Disable transcoders (use hybrid model: ref attention + base MLP)")
    parser.add_argument("--steer", action="append", default=[], type=parse_feature_steering_spec,
                        metavar="CANTOR_ID:STRENGTH",
                        help="Steer a transcoder feature; repeat for multiple targets")
    parser.add_argument("--feature-data", default=None,
                        help="Feature-data run directory; with --steer, print top activation snippets")
    parser.add_argument("--show-steered-activations", action="store_true",
                        help="Color top activated tokens for each steered feature")
    parser.add_argument("--steering_mode", default="min", choices=["min", "add", "set"],
                        help="How to apply steering strengths")
    args = parser.parse_args()

    if args.hybrid and args.steer:
        parser.error("--hybrid cannot be used with --steer")
    if args.feature_data and not args.steer:
        parser.error("--feature-data requires at least one --steer target")
    if args.show_steered_activations and not args.steer:
        parser.error("--show-steered-activations requires at least one --steer target")

    setup_logging()
    model, tokenizer = load_model(args.model_path, args.tokenizer, args.arch)

    if args.hybrid:
        logger.info("Disabling transcoders for hybrid model generation")
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            for layer in model.model.layers:
                if hasattr(layer, "mlp") and hasattr(layer.mlp, "disable_transcoder"):
                    layer.mlp.disable_transcoder = True
        else:
            logger.warning("Could not find layers to disable transcoders. Is this a transcoder model?")

    if args.steer:
        if not hasattr(model, "set_feature_steering"):
            parser.error("Loaded model does not support feature steering")
        try:
            model.set_feature_steering(args.steer, mode=args.steering_mode)
        except ValueError as exc:
            parser.error(str(exc))
        logger.info(f"Feature steering enabled for {len(args.steer)} target(s), mode={args.steering_mode}")
        log_feature_steering_targets(model, args.steer)

    if args.feature_data:
        from analysis.simple_load.feature_data import format_steered_feature_snippets

        snippets = format_steered_feature_snippets(args.feature_data, args.steer)
        if snippets:
            sys.stdout.write(snippets)
            if not snippets.endswith("\n"):
                sys.stdout.write("\n")

    if args.prompt is not None:
        if args.show_steered_activations:
            from analysis.simple_load.activation_highlights import run_prompt_with_activation_highlights

            run_prompt_with_activation_highlights(
                model,
                tokenizer,
                args.prompt,
                args.steer,
                use_chat_template=not args.raw,
                show_special_tokens=args.show_special_tokens,
            )
            return

        response = run_prompt(
            model,
            tokenizer,
            args.prompt,
            use_chat_template=not args.raw,
            show_special_tokens=args.show_special_tokens,
        )
        sys.stdout.write(response)
        if response and not response.endswith("\n"):
            sys.stdout.write("\n")
        return

    if args.show_steered_activations:
        from analysis.simple_load.activation_highlights import chat_with_activation_highlights

        chat_with_activation_highlights(
            model,
            tokenizer,
            args.steer,
            use_chat_template=not args.raw,
            show_special_tokens=args.show_special_tokens,
        )
        return

    chat(model, tokenizer, use_chat_template=not args.raw, show_special_tokens=args.show_special_tokens)


if __name__ == "__main__":
    main()
