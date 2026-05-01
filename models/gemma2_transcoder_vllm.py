"""vLLM-compatible Gemma2 model with integrated transcoder adapters."""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.steering import FeatureSteeringMixin, apply_feature_steering
from models.vllm_steering import get_feature_steering_from_config

from vllm.config import VllmConfig
from vllm.model_executor.models.gemma2 import (
    Gemma2ForCausalLM as VLLMGemma2ForCausalLM,
    Gemma2MLP,
)


def _join_prefix(*parts: str) -> str:
    return ".".join(part for part in parts if part)


class Gemma2MLPWithTranscoder(Gemma2MLP):
    """vLLM Gemma2 MLP with an added transcoder branch."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        hidden_activation: str,
        quant_config: Any | None = None,
        prefix: str = "",
        transcoder_n_features: int = 512,
        transcoder_dec_bias: bool = False,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_activation=hidden_activation,
            quant_config=quant_config,
            prefix=prefix,
        )

        self.hidden_size = hidden_size
        self.n_features = transcoder_n_features
        self.dec_bias = transcoder_dec_bias
        self.transcoder_enc = nn.Linear(self.hidden_size, self.n_features, bias=True)
        self.transcoder_dec = nn.Linear(self.n_features, self.hidden_size, bias=transcoder_dec_bias)
        self.disable_transcoder = False
        self.feature_steering_targets = ()
        self.feature_steering_mode = "min"
        self._init_transcoder_weights()

    def _init_transcoder_weights(self) -> None:
        scale_d_model = 1 / math.sqrt(self.hidden_size)
        nn.init.uniform_(self.transcoder_enc.weight, -scale_d_model, scale_d_model)
        nn.init.zeros_(self.transcoder_enc.bias)
        nn.init.zeros_(self.transcoder_dec.weight)
        if self.dec_bias:
            nn.init.zeros_(self.transcoder_dec.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_output = super().forward(x)
        if self.disable_transcoder:
            return original_output

        features = F.relu(self.transcoder_enc(x))
        features = apply_feature_steering(
            features,
            self.feature_steering_targets,
            self.feature_steering_mode,
        )
        transcoder_output = self.transcoder_dec(features)
        return original_output + transcoder_output


class Gemma2ForCausalLMWithTranscoder(FeatureSteeringMixin, VLLMGemma2ForCausalLM):
    """vLLM Gemma2 causal LM with integrated transcoder adapters."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        n_features = getattr(config, "transcoder_n_features", 512)
        dec_bias = getattr(config, "transcoder_dec_bias", False)
        hidden_act = getattr(config, "hidden_act", getattr(config, "hidden_activation", None))
        hidden_activation = getattr(config, "hidden_activation", hidden_act)

        for layer_idx, layer in enumerate(self.model.layers):
            if not hasattr(layer, "mlp"):
                continue
            layer.mlp = Gemma2MLPWithTranscoder(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=hidden_act,
                hidden_activation=hidden_activation,
                quant_config=quant_config,
                prefix=_join_prefix(prefix, "model", "layers", str(layer_idx), "mlp"),
                transcoder_n_features=n_features,
                transcoder_dec_bias=dec_bias,
            )

        specs, mode = get_feature_steering_from_config(config)
        if specs:
            self.set_feature_steering(specs, mode=mode)

    def _transcoder_mlps(self) -> Iterator[Gemma2MLPWithTranscoder]:
        for layer in self.model.layers:
            mlp = getattr(layer, "mlp", None)
            if isinstance(mlp, Gemma2MLPWithTranscoder):
                yield mlp


def register_gemma2_vllm_transcoder() -> None:
    """Register the Gemma2 transcoder architecture with vLLM."""
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "Gemma2ForCausalLMWithTranscoder",
        "models.gemma2_transcoder_vllm:Gemma2ForCausalLMWithTranscoder",
    )


def register_vllm_transcoder() -> None:
    """Compatibility alias matching the Qwen2 vLLM module."""
    register_gemma2_vllm_transcoder()
