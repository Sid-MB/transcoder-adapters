"""Shared building blocks for RelP transcoder attribution models."""

from __future__ import annotations

import glob
import os
from collections.abc import Callable, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F


def resolve_checkpoint_path(path: str) -> str:
    """Resolve a local path or HuggingFace repo ID to a local checkpoint path."""
    if os.path.exists(path):
        return path

    from huggingface_hub import snapshot_download

    return snapshot_download(path)


def load_checkpoint_state_dict(path: str) -> dict[str, torch.Tensor]:
    """Load a safetensors or PyTorch checkpoint state dict from path."""
    if os.path.isdir(path):
        weight_files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        if weight_files:
            from safetensors.torch import load_file

            state_dict: dict[str, torch.Tensor] = {}
            for filename in weight_files:
                state_dict.update(load_file(filename))
            return state_dict

        weight_file = os.path.join(path, "pytorch_model.bin")
        return torch.load(weight_file, map_location="cpu")

    return torch.load(path, map_location="cpu")


class RelPRMSNorm(nn.Module):
    """RMSNorm with RelP-aware backward through the normalization scale."""

    def __init__(self, hidden_size: int, eps: float = 1e-6, *, gemma_offset: bool = False):
        super().__init__()
        init = torch.zeros(hidden_size) if gemma_offset else torch.ones(hidden_size)
        self.weight = nn.Parameter(init)
        self.variance_epsilon = eps
        self.gemma_offset = gemma_offset
        self.relp_enabled = True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        if input_dtype in (torch.float16, torch.bfloat16):
            hidden_states = hidden_states.to(torch.float32)

        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        scale = torch.rsqrt(variance + self.variance_epsilon)
        if self.relp_enabled:
            scale = scale.detach()

        hidden_states = hidden_states * scale
        weight = (1.0 + self.weight.float()) if self.gemma_offset else self.weight.float()
        return (weight * hidden_states).to(input_dtype)


class RelPMLPWithTranscoder(nn.Module):
    """Gated MLP plus transcoder branch with RelP-linearized nonlinearities."""

    def __init__(
        self,
        config,
        *,
        act_fn: Callable[[torch.Tensor], torch.Tensor],
        bias: bool = False,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = act_fn

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)

        self.n_features = getattr(config, "transcoder_n_features", 512)
        self.dec_bias = getattr(config, "transcoder_dec_bias", False)
        self.transcoder_enc = nn.Linear(self.hidden_size, self.n_features, bias=True)
        self.transcoder_dec = nn.Linear(self.n_features, self.hidden_size, bias=self.dec_bias)

        self.relp_enabled = True
        self.stop_grad_at: set[str] = set()
        self.disable_transcoder = False
        self.cached_features: torch.Tensor | None = None
        self.feature_mask: torch.Tensor | None = None

    def _relp_gate_activation(self, gate: torch.Tensor) -> torch.Tensor:
        """Preserve activation value while detaching the nonlinear gate factor."""
        activated = self.act_fn(gate)
        safe_gate = torch.where(gate.abs() > 1e-6, gate, torch.ones_like(gate))
        scale = torch.where(gate.abs() > 1e-6, activated / safe_gate, torch.ones_like(gate))
        return gate * scale.detach()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)

        if self.relp_enabled:
            gate_act = self._relp_gate_activation(gate)
            combined = gate_act * up
            combined = 0.5 * combined + 0.5 * combined.detach()
        else:
            combined = self.act_fn(gate) * up

        base_output = self.down_proj(combined)
        if "base_mlp_output" in self.stop_grad_at:
            base_output = base_output.detach()

        if self.disable_transcoder:
            self.cached_features = None
            return base_output

        features = F.relu(self.transcoder_enc(hidden_states))
        if self.feature_mask is not None:
            features = features * self.feature_mask

        if "transcoder_features" in self.stop_grad_at:
            features = features.detach().requires_grad_(True)
        elif torch.is_grad_enabled() and features.requires_grad:
            features.retain_grad()

        self.cached_features = features
        transcoder_output = self.transcoder_dec(features)
        if "transcoder_output" in self.stop_grad_at:
            transcoder_output = transcoder_output.detach()

        return base_output + transcoder_output


class RelPCausalLMWithTranscoderMixin:
    """Shared control and feature-cache API for RelP causal LM classes."""

    relp_decoder_layer_cls: type

    def set_relp_enabled(self, enabled: bool):
        self.relp_enabled = enabled
        self._propagate_relp_settings()

    def set_stop_grad_at(self, points: set[str]):
        self.stop_grad_at = points
        self._propagate_relp_settings()

    def _decoder_layers(self) -> Iterator:
        for layer in self.model.layers:
            yield layer

    def _propagate_relp_settings(self):
        for layer in self._decoder_layers():
            for norm_name in getattr(layer, "relp_norm_names", ()):
                getattr(layer, norm_name).relp_enabled = self.relp_enabled
            layer.self_attn.relp_enabled = self.relp_enabled
            layer.self_attn.stop_grad_at = self.stop_grad_at
            layer.mlp.relp_enabled = self.relp_enabled
            layer.mlp.stop_grad_at = self.stop_grad_at
        self.model.norm.relp_enabled = self.relp_enabled

    def set_chunked_attention(self, enabled: bool, chunk_size: int = 512):
        for layer in self._decoder_layers():
            layer.self_attn.use_chunked_attention = enabled
            layer.self_attn.attention_chunk_size = chunk_size

    def set_feature_mask(self, mask: torch.Tensor | None):
        for i, layer in enumerate(self._decoder_layers()):
            layer.mlp.feature_mask = mask[i] if mask is not None else None

    def get_cached_features(self) -> dict:
        return {
            i: layer.mlp.cached_features
            for i, layer in enumerate(self._decoder_layers())
            if layer.mlp.cached_features is not None
        }

    def get_feature_attributions(self) -> dict:
        attrs = {}
        for i, layer in enumerate(self._decoder_layers()):
            features = layer.mlp.cached_features
            if features is not None and features.grad is not None:
                attrs[i] = features * features.grad
        return attrs

    def get_feature_attribution_summary(self) -> dict:
        return {i: attr.sum().item() for i, attr in self.get_feature_attributions().items()}
