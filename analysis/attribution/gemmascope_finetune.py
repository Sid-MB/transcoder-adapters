"""Shared helpers to apply fine-tuned GemmaScope transcoder weights at load time.

<!-- Created by Claude Code session "implement: new 07/02 gemmascope transcoder experiments". 2026-07-10. -->

The fine-tune (analysis/features/finetune_transcoder_shift.py) writes per-layer
``finetuned_layer_{L}.safetensors`` (W_enc/W_dec/b_enc/b_dec; the JumpReLU threshold is frozen so
it is not overwritten). "Applying" the fine-tune anywhere = load the pretrained GemmaScope
``TranscoderSet`` as usual, then copy those layers' weights in place. This one implementation is
shared by every GemmaScope-loading entry point (feature collection, FVU/L0 eval, circuit-tracer
attribution/serving, graph comparison) so the fine-tune drops in with a single ``--finetuned_*``
flag and behaves identically everywhere.
"""

from __future__ import annotations

from pathlib import Path

from helpers.log import logger

_PATCH_KEYS = ("W_enc", "W_dec", "b_enc", "b_dec")


def patch_finetuned_layers(transcoders, finetune_dir, layers, device, dtype) -> None:
    """Copy fine-tuned W_enc/W_dec/b_enc/b_dec into ``transcoders[L]`` in place, for each L in layers.

    Args:
        transcoders: a circuit-tracer ``TranscoderSet`` (indexable by layer).
        finetune_dir: dir containing ``finetuned_layer_{L}.safetensors``.
        layers: layer indices to patch.
        device, dtype: unused directly (weights are cast to each param's own dtype/device); kept
            for a uniform call signature next to ``_load_gemmascope_transcoders``.
    """
    from safetensors.torch import load_file

    finetune_dir = Path(finetune_dir)
    for layer in layers:
        sd = load_file(str(finetune_dir / f"finetuned_layer_{layer}.safetensors"))
        t = transcoders[layer]
        for key in _PATCH_KEYS:
            param = getattr(t, key)
            param.data.copy_(sd[key].to(device=param.device, dtype=param.dtype))
        logger.info("Patched fine-tuned weights into layer %d from %s", layer, finetune_dir)


def maybe_patch(transcoders, finetune_dir, finetune_layers, device, dtype) -> None:
    """Patch only if both ``finetune_dir`` and ``finetune_layers`` are provided (else no-op)."""
    if finetune_dir and finetune_layers:
        patch_finetuned_layers(transcoders, finetune_dir, finetune_layers, device, dtype)
