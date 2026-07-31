"""Gemma4 model with integrated transcoder adapters."""

import math
from collections.abc import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Gemma4ForCausalLM, Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextMLP

from models.steering import FeatureSteeringMixin, apply_feature_steering


_GEMMA4_UNUSED_MULTIMODAL_KEY_PATTERNS = [
    r"^model\.audio_tower\.",
    r"^model\.embed_audio\.",
    r"^model\.embed_vision\.",
    r"^model\.vision_tower\.",
]


class Gemma4ConfigWithTranscoder(Gemma4TextConfig):
    """Gemma4 config with transcoder parameters."""

    # Preserve the upstream text config type; the model registry maps it to gemma4.
    model_type = "gemma4_text"

    def __init__(self, transcoder_n_features=512, transcoder_dec_bias=False, **kwargs):
        self.transcoder_n_features = transcoder_n_features
        self.transcoder_dec_bias = transcoder_dec_bias
        super().__init__(**kwargs)
        self.architectures = ["Gemma4ForCausalLMWithTranscoder"]


class Gemma4MLPWithTranscoder(Gemma4TextMLP):
    """Gemma4 MLP with integrated transcoder branch."""

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.config = config
        self.d_model = config.hidden_size
        self.n_features = getattr(config, "transcoder_n_features", 512)
        self.dec_bias = getattr(config, "transcoder_dec_bias", False)

        self.transcoder_enc = nn.Linear(self.d_model, self.n_features, bias=True)
        self.transcoder_dec = nn.Linear(self.n_features, self.d_model, bias=self.dec_bias)
        self._init_transcoder_weights()

        self.disable_transcoder = False
        self.cache_features = False
        self.cached_l1 = None
        self.cached_l0 = None
        self._dead_feature_counters = torch.zeros(self.n_features)
        self._attention_mask = None
        self.feature_steering_targets = ()
        self.feature_steering_mode = "min"

    def _init_transcoder_weights(self):
        scale_d_model = 1 / math.sqrt(self.d_model)
        nn.init.uniform_(self.transcoder_enc.weight, -scale_d_model, scale_d_model)
        nn.init.zeros_(self.transcoder_enc.bias)
        nn.init.zeros_(self.transcoder_dec.weight)
        if self.dec_bias:
            nn.init.zeros_(self.transcoder_dec.bias)

    def forward(self, hidden_states):  # type: ignore[override]
        original_output = super().forward(hidden_states)
        if getattr(self, "disable_transcoder", False):
            return original_output

        features = F.relu(self.transcoder_enc(hidden_states))
        features = apply_feature_steering(
            features,
            self.feature_steering_targets,
            self.feature_steering_mode,
        )
        transcoder_output = self.transcoder_dec(features)

        if self.cache_features:
            batch_size = features.shape[0]
            dec_column_norms = torch.norm(self.transcoder_dec.weight, dim=0)
            weighted_features = features * dec_column_norms.unsqueeze(0).unsqueeze(0)
            per_token_l1 = weighted_features.sum(dim=-1)

            if self._attention_mask is not None:
                mask = self._attention_mask.bool().to(per_token_l1.device)
                n_real = mask.sum().clamp(min=1)
                self.cached_l1 = (per_token_l1 * mask).sum() / n_real
            else:
                self.cached_l1 = per_token_l1.mean()

            with torch.no_grad():
                feature_active = features > 0
                per_token_l0 = feature_active.float().sum(dim=-1)
                if self._attention_mask is not None:
                    mask = self._attention_mask.bool().to(per_token_l0.device)
                    n_real = mask.sum().clamp(min=1)
                    self.cached_l0 = ((per_token_l0 * mask).sum() / n_real).item()
                else:
                    self.cached_l0 = per_token_l0.mean().item()

                if self._dead_feature_counters.is_meta:
                    self._dead_feature_counters = torch.zeros(self.n_features, device=features.device)
                else:
                    self._dead_feature_counters = self._dead_feature_counters.to(features.device)
                self._dead_feature_counters += batch_size
                self._dead_feature_counters[feature_active.any(dim=(0, 1))] = 0

        return original_output + transcoder_output


class Gemma4ForCausalLMWithTranscoder(FeatureSteeringMixin, Gemma4ForCausalLM):
    """Gemma4 causal LM with integrated transcoder adapters."""

    config_class = Gemma4ConfigWithTranscoder

    def __init__(self, config):
        super().__init__(config)
        if hasattr(self.model, "language_model"):
            self.model = self.model.language_model
        self._keys_to_ignore_on_load_unexpected = [
            *(self._keys_to_ignore_on_load_unexpected or []),
            *_GEMMA4_UNUSED_MULTIMODAL_KEY_PATTERNS,
        ]
        for layer_idx, layer in enumerate(self.model.layers):
            layer.mlp = Gemma4MLPWithTranscoder(config, layer_idx)

    def forward(self, input_ids=None, attention_mask=None, *args, **kwargs):
        for layer in self.model.layers:
            layer.mlp._attention_mask = attention_mask  # type: ignore[union-attr]
        return super().forward(input_ids, attention_mask, *args, **kwargs)

    def _transcoder_mlps(self) -> Iterator[Gemma4MLPWithTranscoder]:
        for layer in self.model.layers:
            yield layer.mlp  # type: ignore[misc]

    def set_cache_features(self, enabled: bool):
        for mlp in self._transcoder_mlps():
            mlp.cache_features = enabled

    def collect_sparsity_loss(self) -> torch.Tensor:
        device = next(self.parameters()).device
        total = torch.tensor(0.0, device=device)
        for mlp in self._transcoder_mlps():
            if mlp.cached_l1 is not None:
                total = total + mlp.cached_l1.to(device)
        return total

    def collect_transcoder_stats(self) -> dict:
        stats = {}
        l0_values = []
        dead_100_total = 0
        dead_1000_total = 0
        for i, mlp in enumerate(self._transcoder_mlps()):
            if mlp.cached_l0 is not None:
                l0_values.append(mlp.cached_l0)
                stats[f"adapter_stats_per_layer/layer_{i}_l0_count"] = mlp.cached_l0
            dead_100 = (mlp._dead_feature_counters >= 100).sum().item()
            dead_1000 = (mlp._dead_feature_counters >= 1000).sum().item()
            dead_100_total += dead_100
            dead_1000_total += dead_1000
            stats[f"adapter_stats_per_layer/layer_{i}_dead_features_100"] = dead_100
            stats[f"adapter_stats_per_layer/layer_{i}_dead_features_1000"] = dead_1000
        n_layers = len(self.model.layers)
        if l0_values:
            stats["adapter_stats_avg/l0_count"] = sum(l0_values) / len(l0_values)
        stats["adapter_stats_avg/dead_features_100"] = dead_100_total / n_layers
        stats["adapter_stats_avg/dead_features_1000"] = dead_1000_total / n_layers
        return stats

    def clear_cached_stats(self):
        for mlp in self._transcoder_mlps():
            mlp.cached_l1 = None
            mlp.cached_l0 = None
            mlp._attention_mask = None


def register_gemma4_transcoder():
    import transformers

    setattr(transformers, "Gemma4ForCausalLMWithTranscoder", Gemma4ForCausalLMWithTranscoder)


register_gemma4_transcoder()
