"""Combine a GemmaScope base transcoder and our adapter transcoder per layer.

Full-replacement attribution (Nathan 5/28 notes, ``MLP(x) = T_base(x) + T_finetune(x) + Err``):
replacing each MLP with the SUM of the base GemmaScope transcoder ``T_base`` and our
trained adapter transcoder ``T_adapter`` removes the base-MLP nonlinearity that forced
RelP. With the MLP fully replaced by transcoders, circuit-tracer's standard linear
attribution applies directly and produces real reconstruction-error nodes
(``error = MLP_true - T_base - T_adapter``) for free — exactly the error triangles the
notes ask for.

Construction: per layer, a single ``SingleLayerTranscoder`` whose feature bank is the
concatenation of the two transcoders' banks:

    W_enc = [W_enc_base ; W_enc_adapter]      shape (n_base + n_adapter, d_model)
    W_dec = [W_dec_base ; W_dec_adapter]      shape (n_base + n_adapter, d_model)
    b_enc = [b_enc_base ; b_enc_adapter]      shape (n_base + n_adapter,)
    b_dec = b_dec_base + b_dec_adapter        shape (d_model,)
    activation = blockwise: base activation (JumpReLU) on [:n_base],
                            adapter activation (ReLU) on [n_base:]

so ``combined.forward(x) == T_base(x) + T_adapter(x)`` exactly (see verify_combination).
Base features occupy combined indices ``[0, n_base)``; adapter features
``[n_base, n_base + n_adapter)``. Use feature_source()/split_combined_feature_index() to
map a combined feature node back to its source ("base"/"adapter") and that source's local
feature index — needed for node tagging in the overlay and for routing feature examples to
``/base_features`` vs ``/adapter_features``.
"""

from __future__ import annotations

import torch
from torch import nn

from circuit_tracer.transcoder.single_layer_transcoder import (
    SingleLayerTranscoder,
    TranscoderSet,
)

SOURCE_BASE = "base"
SOURCE_ADAPTER = "adapter"


class BlockwiseActivation(nn.Module):
    """Apply ``base_activation`` to the first ``n_base`` features, ``adapter_activation`` to the rest.

    GemmaScope uses a per-feature JumpReLU; our adapter uses ReLU. A combined transcoder
    therefore needs different activations on the two feature blocks; this module splits the
    pre-activations on the last dim and applies each block's own activation function (which
    already carries its own parameters, e.g. the JumpReLU threshold).
    """

    def __init__(self, n_base: int, base_activation: nn.Module, adapter_activation: nn.Module):
        super().__init__()
        self.n_base = n_base
        self.base_activation = base_activation
        self.adapter_activation = adapter_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_activation(x[..., : self.n_base])
        adapter = self.adapter_activation(x[..., self.n_base :])
        return torch.cat([base, adapter], dim=-1)


def build_combined_layer_transcoder(
    base_tc: SingleLayerTranscoder,
    adapter_tc: SingleLayerTranscoder,
    *,
    layer_idx: int,
    device: torch.device,
    dtype: torch.dtype,
) -> SingleLayerTranscoder:
    """Concatenate one base and one adapter SingleLayerTranscoder into a combined one."""
    if base_tc.d_model != adapter_tc.d_model:
        raise ValueError(
            f"d_model mismatch at layer {layer_idx}: base {base_tc.d_model} vs adapter {adapter_tc.d_model}"
        )
    if base_tc.W_skip is not None or adapter_tc.W_skip is not None:
        raise NotImplementedError("Combined transcoders with skip connections are not supported yet.")

    n_base = base_tc.d_transcoder
    n_adapter = adapter_tc.d_transcoder
    d_model = base_tc.d_model

    activation = BlockwiseActivation(
        n_base,
        base_tc.activation_function,
        adapter_tc.activation_function,
    )
    combined = SingleLayerTranscoder(
        d_model=d_model,
        d_transcoder=n_base + n_adapter,
        activation_function=activation,
        layer_idx=layer_idx,
        skip_connection=False,
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        combined.W_enc.copy_(
            torch.cat([base_tc.W_enc.to(device, dtype), adapter_tc.W_enc.to(device, dtype)], dim=0)
        )
        combined.W_dec.copy_(
            torch.cat([base_tc.W_dec.to(device, dtype), adapter_tc.W_dec.to(device, dtype)], dim=0)
        )
        combined.b_enc.copy_(
            torch.cat([base_tc.b_enc.to(device, dtype), adapter_tc.b_enc.to(device, dtype)], dim=0)
        )
        combined.b_dec.copy_(base_tc.b_dec.to(device, dtype) + adapter_tc.b_dec.to(device, dtype))
    return combined


def build_combined_transcoder_set(
    base_set: TranscoderSet,
    adapter_set: TranscoderSet,
    *,
    scan_name: str | list[str] | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> TranscoderSet:
    """Build a per-layer combined transcoder set (T_base + T_adapter) from two sets."""
    if base_set.n_layers != adapter_set.n_layers:
        raise ValueError(
            f"Layer count mismatch: base {base_set.n_layers} vs adapter {adapter_set.n_layers}"
        )
    for hook_name in ("feature_input_hook", "feature_output_hook"):
        base_hook = getattr(base_set, hook_name)
        adapter_hook = getattr(adapter_set, hook_name)
        if base_hook != adapter_hook:
            raise ValueError(
                f"{hook_name} mismatch: base {base_hook!r} vs adapter {adapter_hook!r}; "
                "both transcoders must read/write the same hooks to be summed."
            )

    transcoders: dict[int, SingleLayerTranscoder] = {}
    for layer in range(base_set.n_layers):
        base_tc = base_set[layer]
        adapter_tc = adapter_set[layer]
        layer_device = device or base_tc.device
        layer_dtype = dtype or base_tc.dtype
        transcoders[layer] = build_combined_layer_transcoder(
            base_tc,
            adapter_tc,
            layer_idx=layer,
            device=layer_device,
            dtype=layer_dtype,
        )
    return TranscoderSet(
        transcoders,
        feature_input_hook=base_set.feature_input_hook,
        feature_output_hook=base_set.feature_output_hook,
        scan_name=scan_name,
    )


def extract_adapter_transcoder_set(
    adapter_model,
    *,
    feature_input_hook: str = "ln2.hook_normalized",
    feature_output_hook: str = "hook_mlp_out",
    scan_name: str | list[str] | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> TranscoderSet:
    """Extract our trained adapter's per-layer transcoders into a circuit-tracer TranscoderSet.

    The adapter MLP branch is ``out = transcoder_dec(relu(transcoder_enc(x)))``. Mapped to
    circuit-tracer's convention (``encode``: ``F.linear(x, W_enc, b_enc)``; ``decode``:
    ``acts @ W_dec + b_dec``):

        W_enc <- mlp.transcoder_enc.weight              [n_features, d_model]
        b_enc <- mlp.transcoder_enc.bias
        W_dec <- mlp.transcoder_dec.weight.T            [n_features, d_model]
        b_dec <- mlp.transcoder_dec.bias                (the checkpoint has a decoder bias)

    This orientation was validated algebraically exact (fp32-vs-fp64 relative error <= 2.6e-7
    across all 26 layers) in misc_scripts/validate_combined_transcoder.py.
    """
    from circuit_tracer.transcoder.single_layer_transcoder import SingleLayerTranscoder

    layers = adapter_model.model.layers
    transcoders: dict[int, SingleLayerTranscoder] = {}
    for layer_idx, layer in enumerate(layers):
        enc = layer.mlp.transcoder_enc
        dec = layer.mlp.transcoder_dec
        n_features, d_model = enc.weight.shape
        layer_device = device or enc.weight.device
        layer_dtype = dtype or enc.weight.dtype
        tc = SingleLayerTranscoder(
            d_model=d_model,
            d_transcoder=n_features,
            activation_function=torch.nn.ReLU(),
            layer_idx=layer_idx,
            skip_connection=False,
            device=layer_device,
            dtype=layer_dtype,
        )
        with torch.no_grad():
            tc.W_enc.copy_(enc.weight.to(layer_device, layer_dtype))
            tc.W_dec.copy_(dec.weight.T.contiguous().to(layer_device, layer_dtype))
            if enc.bias is not None:
                tc.b_enc.copy_(enc.bias.to(layer_device, layer_dtype))
            if dec.bias is not None:
                tc.b_dec.copy_(dec.bias.to(layer_device, layer_dtype))
        transcoders[layer_idx] = tc
    return TranscoderSet(
        transcoders,
        feature_input_hook=feature_input_hook,
        feature_output_hook=feature_output_hook,
        scan_name=scan_name,
    )


def n_base_features(base_set: TranscoderSet) -> int:
    """Number of base (GemmaScope) features per layer = the base/adapter split point."""
    return base_set[0].d_transcoder


def feature_source(feature_idx: int, n_base: int) -> str:
    """Return SOURCE_BASE or SOURCE_ADAPTER for a combined feature index."""
    return SOURCE_BASE if feature_idx < n_base else SOURCE_ADAPTER


def split_combined_feature_index(feature_idx: int, n_base: int) -> tuple[str, int]:
    """Map a combined feature index to (source, source-local feature index)."""
    if feature_idx < n_base:
        return SOURCE_BASE, feature_idx
    return SOURCE_ADAPTER, feature_idx - n_base


@torch.no_grad()
def verify_combination(
    base_tc: SingleLayerTranscoder,
    adapter_tc: SingleLayerTranscoder,
    combined_tc: SingleLayerTranscoder,
    *,
    n_samples: int = 4,
    seed_offset: int = 0,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> dict[str, float]:
    """Check combined(x) == base(x) + adapter(x) on random inputs (forward-equivalence)."""
    device = combined_tc.device
    dtype = combined_tc.dtype
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1234 + seed_offset)
    x = torch.randn(n_samples, base_tc.d_model, generator=generator).to(device, dtype)

    combined_out = combined_tc(x)
    expected = base_tc(x.clone()).to(device, dtype) + adapter_tc(x.clone()).to(device, dtype)
    diff = (combined_out - expected).abs()
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "allclose": bool(torch.allclose(combined_out, expected, atol=atol, rtol=rtol)),
    }
