"""ANSI activation highlights for simple_load steering targets."""

from __future__ import annotations

import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import torch

from helpers.log import logger
from models.steering import FeatureSteeringSpec, cantor_unpair


_RESET = "\x1b[0m"


@dataclass(frozen=True)
class _Target:
    spec: FeatureSteeringSpec
    layer_idx: int
    feature_idx: int


def _extract_input_ids(encoded) -> torch.Tensor:
    if isinstance(encoded, torch.Tensor):
        return encoded
    if hasattr(encoded, "input_ids"):
        return encoded.input_ids
    if isinstance(encoded, dict) and "input_ids" in encoded:
        return encoded["input_ids"]
    raise TypeError(f"Could not extract input_ids from {type(encoded).__name__}")


def _targets(specs: Iterable[FeatureSteeringSpec]) -> list[_Target]:
    return [
        _Target(spec=spec, layer_idx=layer_idx, feature_idx=feature_idx)
        for spec in specs
        for layer_idx, feature_idx in [cantor_unpair(spec.cantor_id)]
    ]


def _input_ids_for_prompt(model, tokenizer, prompt: str, use_chat_template: bool) -> torch.Tensor:
    if use_chat_template:
        input_ids = _extract_input_ids(tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        ))
    else:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    return input_ids.to(model.device)


def _input_ids_for_messages(model, tokenizer, messages: list[dict[str, str]], use_chat_template: bool) -> torch.Tensor:
    if use_chat_template:
        input_ids = _extract_input_ids(tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ))
    else:
        input_ids = tokenizer(messages[-1]["content"], return_tensors="pt").input_ids
    return input_ids.to(model.device)


def _generate(model, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model.generate(
            input_ids,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )


def _collect_activations(
    model,
    input_ids: torch.Tensor,
    targets: list[_Target],
) -> dict[int, list[float]]:
    """Collect pre-steering ReLU activations for target features."""
    if not hasattr(model, "_transcoder_mlps"):
        raise ValueError("Loaded model does not expose transcoder MLPs for activation highlighting")

    mlps = tuple(model._transcoder_mlps())
    by_layer: dict[int, list[_Target]] = defaultdict(list)
    for target in targets:
        if target.layer_idx >= len(mlps):
            raise ValueError(
                f"Feature steering layer {target.layer_idx} out of range for {len(mlps)} layers"
            )
        by_layer[target.layer_idx].append(target)

    chunks: dict[int, list[torch.Tensor]] = {target.spec.cantor_id: [] for target in targets}
    handles = []

    def make_hook(layer_targets: list[_Target]):
        def hook(module, inputs, output):
            features = torch.relu(output.detach()).float().cpu()
            for target in layer_targets:
                if target.feature_idx >= features.shape[-1]:
                    raise ValueError(
                        f"Feature steering feature {target.feature_idx} out of range for layer "
                        f"{target.layer_idx} with {features.shape[-1]} features"
                    )
                chunks[target.spec.cantor_id].append(features[0, :, target.feature_idx])

        return hook

    try:
        for layer_idx, layer_targets in by_layer.items():
            handles.append(mlps[layer_idx].transcoder_enc.register_forward_hook(make_hook(layer_targets)))
        with torch.no_grad():
            model(input_ids)
    finally:
        for handle in handles:
            handle.remove()

    return {
        cantor_id: torch.cat(parts).tolist() if parts else []
        for cantor_id, parts in chunks.items()
    }


def _ansi_orange(value: float) -> str:
    value = max(0.0, min(1.0, value))
    red = int(80 + 175 * value)
    green = int(24 + 116 * value)
    return f"\x1b[48;2;{red};{green};0m"


def _decode_token(tokenizer, token_id: int, show_special_tokens: bool) -> str:
    return tokenizer.decode([token_id], skip_special_tokens=not show_special_tokens)


def _highlighted_text(
    tokenizer,
    token_ids: list[int],
    activations: list[float],
    show_special_tokens: bool,
    top_k: int = 20,
) -> str:
    if not activations:
        return "".join(_decode_token(tokenizer, token_id, show_special_tokens) for token_id in token_ids)

    n_tokens = min(len(token_ids), len(activations))
    positive = [(idx, act) for idx, act in enumerate(activations[:n_tokens]) if act > 0]
    if not positive:
        return "".join(_decode_token(tokenizer, token_id, show_special_tokens) for token_id in token_ids)

    top = sorted(positive, key=lambda item: item[1], reverse=True)[:top_k]
    top_indices = {idx for idx, _ in top}
    max_activation = max(act for _, act in top)

    parts: list[str] = []
    for idx, token_id in enumerate(token_ids):
        token = _decode_token(tokenizer, token_id, show_special_tokens)
        if idx in top_indices and max_activation > 0:
            intensity = activations[idx] / max_activation
            parts.append(f"{_ansi_orange(intensity)}{token}{_RESET}")
        else:
            parts.append(token)
    return "".join(parts)


def _write_activation_views(
    model,
    tokenizer,
    output_ids: torch.Tensor,
    specs: list[FeatureSteeringSpec],
    show_special_tokens: bool,
) -> None:
    targets = _targets(specs)
    activations = _collect_activations(model, output_ids.to(model.device), targets)
    token_ids = output_ids[0].detach().cpu().tolist()

    sys.stdout.write("\n[STEERED FEATURE ACTIVATIONS]\n")
    for target in targets:
        values = activations.get(target.spec.cantor_id, [])
        max_activation = max(values) if values else 0.0
        sys.stdout.write(
            f"cantor={target.spec.cantor_id} layer={target.layer_idx} "
            f"feature={target.feature_idx} pre_steering_max_activation={max_activation:.6g}\n"
        )
        if max_activation <= 0:
            sys.stdout.write("No positive pre-steering activations in this chat turn.\n")
        sys.stdout.write(_highlighted_text(tokenizer, token_ids, values, show_special_tokens))
        sys.stdout.write("\n")


def run_prompt_with_activation_highlights(
    model,
    tokenizer,
    prompt: str,
    specs: list[FeatureSteeringSpec],
    use_chat_template: bool = True,
    show_special_tokens: bool = False,
) -> str:
    input_ids = _input_ids_for_prompt(model, tokenizer, prompt, use_chat_template)
    if show_special_tokens:
        logger.info(f"\n[PROMPT] {tokenizer.decode(input_ids[0], skip_special_tokens=False)}")
        logger.info("[OUTPUT]")

    output_ids = _generate(model, input_ids)
    new_tokens = output_ids[0][input_ids.shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=not show_special_tokens)

    sys.stdout.write(response)
    if response and not response.endswith("\n"):
        sys.stdout.write("\n")
    _write_activation_views(model, tokenizer, output_ids, specs, show_special_tokens)
    return response


def chat_with_activation_highlights(
    model,
    tokenizer,
    specs: list[FeatureSteeringSpec],
    use_chat_template: bool = True,
    show_special_tokens: bool = False,
) -> None:
    messages: list[dict[str, str]] = []
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
            input_ids = _input_ids_for_messages(model, tokenizer, messages, use_chat_template=True)
        else:
            messages = [{"role": "user", "content": user_input}]
            input_ids = _input_ids_for_messages(model, tokenizer, messages, use_chat_template=False)

        if show_special_tokens:
            logger.info(f"\n[PROMPT] {tokenizer.decode(input_ids[0], skip_special_tokens=False)}")
            logger.info("[OUTPUT]")

        output_ids = _generate(model, input_ids)
        new_tokens = output_ids[0][input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=not show_special_tokens)
        sys.stdout.write(response)
        if response and not response.endswith("\n"):
            sys.stdout.write("\n")

        if use_chat_template:
            messages.append({"role": "assistant", "content": response})

        _write_activation_views(model, tokenizer, output_ids, specs, show_special_tokens)
        logger.info("")
