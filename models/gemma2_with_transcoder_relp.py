"""Gemma2 model with RelP-aware backward pass for attribution analysis."""

from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, Gemma2Config
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.gemma2.modeling_gemma2 import (
    Gemma2RotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)

from models.relp_common import (
    RelPCausalLMWithTranscoderMixin,
    RelPMLPWithTranscoder,
    RelPRMSNorm,
    load_checkpoint_state_dict,
    resolve_checkpoint_path,
)


class Gemma2MLPWithTranscoderRelP(RelPMLPWithTranscoder):
    """Gemma2 gated MLP plus transcoder branch with RelP-linearized GeLU."""

    def __init__(self, config: Gemma2Config):
        super().__init__(
            config,
            act_fn=ACT2FN[config.hidden_activation],
            bias=False,
        )


class Gemma2AttentionRelP(nn.Module):
    """Gemma2 attention with RelP-aware detached attention probabilities."""

    def __init__(self, config: Gemma2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        layer_types = getattr(config, "layer_types", None)
        self.layer_type = layer_types[layer_idx] if layer_types is not None else (
            "sliding_attention" if bool((layer_idx + 1) % 2) else "full_attention"
        )
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = config.query_pre_attn_scalar**-0.5
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None
        self.attn_logit_softcapping = config.attn_logit_softcapping

        bias = getattr(config, "attention_bias", False)
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=bias)

        self.relp_enabled = True
        self.stop_grad_at: set[str] = set()
        self.use_chunked_attention = True
        self.attention_chunk_size = 512

    def _softcap(self, scores: torch.Tensor) -> torch.Tensor:
        if self.attn_logit_softcapping is None:
            return scores
        scores = scores / self.attn_logit_softcapping
        scores = torch.tanh(scores)
        return scores * self.attn_logit_softcapping

    def _chunked_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> torch.Tensor:
        batch, heads, seq_len, _ = query_states.shape
        chunk_size = self.attention_chunk_size
        outputs = []

        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)
            q_chunk = query_states[:, :, chunk_start:chunk_end]
            k_slice = key_states[:, :, :chunk_end]
            v_slice = value_states[:, :, :chunk_end]

            scores = torch.matmul(q_chunk, k_slice.transpose(-2, -1)) * self.scaling
            scores = self._softcap(scores)

            chunk_len = chunk_end - chunk_start
            kv_len = chunk_end
            row_idx = torch.arange(chunk_len, device=query_states.device).unsqueeze(1)
            col_idx = torch.arange(kv_len, device=query_states.device).unsqueeze(0)
            absolute_query_pos = row_idx + chunk_start
            valid = col_idx <= absolute_query_pos
            if self.sliding_window is not None:
                valid = valid & (col_idx > absolute_query_pos - self.sliding_window)
            scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)

            weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
            if self.relp_enabled:
                weights = weights.detach()
            outputs.append(torch.matmul(weights, v_slice))

        return torch.cat(outputs, dim=2)

    def _full_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        scores = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        scores = self._softcap(scores)
        if attention_mask is not None:
            scores = scores + attention_mask[:, :, :, : key_states.shape[-2]]
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        if self.relp_enabled:
            weights = weights.detach()
        return torch.matmul(weights, value_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        batch, seq_len, _ = hidden_states.shape
        hidden_shape = (batch, seq_len, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if self.use_chunked_attention:
            attn_output = self._chunked_attention(query_states, key_states, value_states)
        else:
            attn_output = self._full_attention(query_states, key_states, value_states, attention_mask)

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch, seq_len, -1)
        attn_output = self.o_proj(attn_output)
        if "attn_output" in self.stop_grad_at:
            attn_output = attn_output.detach()
        return attn_output, None


class Gemma2DecoderLayerRelP(nn.Module):
    """Gemma2 decoder layer with RelP-aware attention, norms, and MLP."""

    relp_norm_names = (
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
    )

    def __init__(self, config: Gemma2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        self.self_attn = Gemma2AttentionRelP(config, layer_idx)
        self.mlp = Gemma2MLPWithTranscoderRelP(config)
        self.input_layernorm = RelPRMSNorm(config.hidden_size, eps=config.rms_norm_eps, gemma_offset=True)
        self.post_attention_layernorm = RelPRMSNorm(config.hidden_size, eps=config.rms_norm_eps, gemma_offset=True)
        self.pre_feedforward_layernorm = RelPRMSNorm(config.hidden_size, eps=config.rms_norm_eps, gemma_offset=True)
        self.post_feedforward_layernorm = RelPRMSNorm(config.hidden_size, eps=config.rms_norm_eps, gemma_offset=True)
        layer_types = getattr(config, "layer_types", None)
        self.attention_type = layer_types[layer_idx] if layer_types is not None else (
            "sliding_attention" if bool((layer_idx + 1) % 2) else "full_attention"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Gemma2ModelRelP(nn.Module):
    """Gemma2 backbone with RelP-aware layers."""

    def __init__(self, config: Gemma2Config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            Gemma2DecoderLayerRelP(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RelPRMSNorm(config.hidden_size, eps=config.rms_norm_eps, gemma_offset=True)
        self.rotary_emb = Gemma2RotaryEmbedding(config=config)

    def _layer_type(self, layer_idx: int) -> str:
        layer_types = getattr(self.config, "layer_types", None)
        if layer_types is not None:
            return layer_types[layer_idx]
        return "sliding_attention" if bool((layer_idx + 1) % 2) else "full_attention"

    def _make_causal_mask(
        self,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
        *,
        sliding_window: int | None = None,
    ) -> torch.Tensor:
        row = torch.arange(seq_len, device=device).unsqueeze(1)
        col = torch.arange(seq_len, device=device).unsqueeze(0)
        valid = col <= row
        if sliding_window is not None:
            valid = valid & (col > row - sliding_window)
        mask = torch.zeros((seq_len, seq_len), dtype=dtype, device=device)
        mask = mask.masked_fill(~valid, torch.finfo(dtype).min)
        return mask.unsqueeze(0).unsqueeze(0)

    def _causal_mask_mapping(
        self,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        return {
            "full_attention": self._make_causal_mask(seq_len, dtype, device),
            "sliding_attention": self._make_causal_mask(
                seq_len, dtype, device, sliding_window=self.config.sliding_window
            ),
        }

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            assert inputs_embeds is not None

        hidden_states = inputs_embeds
        seq_len = hidden_states.shape[1]
        position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        mask_mapping = self._causal_mask_mapping(seq_len, hidden_states.dtype, hidden_states.device)

        for i, layer in enumerate(self.layers):
            attention_type = self._layer_type(i)
            hidden_states = layer(
                hidden_states,
                attention_mask=mask_mapping[attention_type],
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


class Gemma2ForCausalLMWithTranscoderRelP(RelPCausalLMWithTranscoderMixin, nn.Module):
    """Gemma2 + transcoder with RelP-aware backward pass."""

    relp_decoder_layer_cls = Gemma2DecoderLayerRelP

    def __init__(self, config: Gemma2Config):
        super().__init__()
        self.config = config
        self.model = Gemma2ModelRelP(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.relp_enabled = True
        self.stop_grad_at: set[str] = set()

    def _decoder_layers(self) -> Iterator[Gemma2DecoderLayerRelP]:
        for layer in self.model.layers:
            yield layer  # type: ignore[misc]

    def run_backbone_with_cache(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        resid_cache: dict[int | str, torch.Tensor],
    ) -> torch.Tensor:
        self.set_stop_grad_at({"transcoder_features"})
        input_ids = tokens.unsqueeze(0).expand(batch_size, -1)
        hidden_states = self.model.embed_tokens(input_ids)
        hidden_states.retain_grad()
        resid_cache["embed"] = hidden_states

        seq_len = hidden_states.shape[1]
        position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
        mask_mapping = self.model._causal_mask_mapping(seq_len, hidden_states.dtype, hidden_states.device)

        for layer_idx, layer in enumerate(self.model.layers):
            hidden_states.retain_grad()
            resid_cache[layer_idx] = hidden_states
            hidden_states = layer(
                hidden_states,
                attention_mask=mask_mapping[self.model._layer_type(layer_idx)],
                position_embeddings=position_embeddings,
            )

        hidden_states = self.model.norm(hidden_states)
        hidden_states.retain_grad()
        resid_cache[self.config.num_hidden_layers] = hidden_states
        return hidden_states

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        logits = self.lm_head(outputs.last_hidden_state)
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping
        return CausalLMOutputWithPast(logits=logits)

    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        path = resolve_checkpoint_path(path)
        config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        if not isinstance(config, Gemma2Config):
            raise TypeError(f"Expected Gemma2Config, got {type(config).__name__}")

        model = cls(config)
        state_dict = load_checkpoint_state_dict(path)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if "lm_head.weight" in missing and getattr(config, "tie_word_embeddings", False):
            model.lm_head.weight = model.model.embed_tokens.weight
        return model
