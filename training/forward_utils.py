"""Forward pass utilities for bridging experiments."""

from typing import TYPE_CHECKING

import torch
import random

from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
if TYPE_CHECKING:
    from transformers import PreTrainedModel


def _language_backbone(model: "PreTrainedModel"):
    backbone = getattr(model, "model", model)
    if hasattr(backbone, "layers"):
        return backbone
    if hasattr(backbone, "language_model") and hasattr(backbone.language_model, "layers"):
        return backbone.language_model
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model
    raise AttributeError(f"Could not find language backbone on {type(model).__name__}")


def _layer_type(backbone, layer, layer_idx: int) -> str:
    if hasattr(backbone.config, "layer_types"):
        return backbone.config.layer_types[layer_idx]
    if hasattr(layer, "attention_type"):
        return layer.attention_type
    if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_type"):
        return layer.self_attn.layer_type
    return "full_attention"


def _per_layer_inputs(backbone, input_ids: torch.Tensor, inputs_embeds: torch.Tensor):
    if not getattr(backbone, "hidden_size_per_layer_input", 0):
        return None
    token_inputs = backbone.get_per_layer_inputs(input_ids, inputs_embeds)
    return backbone.project_per_layer_inputs(inputs_embeds, token_inputs)


def _run_layer(layer, hidden_states, *, per_layer_input, **kwargs):
    if per_layer_input is None:
        return layer(hidden_states, **kwargs)
    return layer(hidden_states, per_layer_input=per_layer_input, **kwargs)


def forward_mixed(
    model1: "PreTrainedModel",
    model2: "PreTrainedModel",
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    switch_layer: int,
) -> torch.Tensor:
    """
    Forward pass that switches from model1 to model2 at switch_layer.

    Runs model1 for layers 0..switch_layer-1, then model2 for layers switch_layer..L.
    Uses model1's embeddings and model2's final norm + lm_head.

    Args:
        model1: First model (e.g., adapter model)
        model2: Second model (e.g., reference model)
        input_ids: input token ids [batch, seq_len]
        attention_mask: attention mask [batch, seq_len]
        switch_layer: layer index to switch at (0 = all model2, L = all model1)

    Returns:
        logits: output logits [batch, seq_len, vocab_size]
    """
    backbone1 = _language_backbone(model1)
    backbone2 = _language_backbone(model2)

    # Get embeddings from model1
    h = backbone1.embed_tokens(input_ids)
    per_layer_inputs1 = _per_layer_inputs(backbone1, input_ids, h)

    # Setup position ids and cache position
    seq_len = h.shape[1]
    device = h.device
    cache_position = torch.arange(seq_len, device=device)
    position_ids = cache_position.unsqueeze(0)

    # Create causal mask (use model1's config, should be same arch)
    mask_kwargs = {
        "config": backbone1.config,
        "inputs_embeds": h,
        "attention_mask": attention_mask,
        "cache_position": cache_position,
        "past_key_values": None,
        "position_ids": position_ids,
    }
    causal_mask_mapping = {
        "full_attention": create_causal_mask(**mask_kwargs), # type: ignore
    }
    if any(_layer_type(backbone1, layer, i) == "sliding_attention" for i, layer in enumerate(backbone1.layers)):
        causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs) # type: ignore

    # Position embeddings from model1
    position_embeddings = {
        layer_type: backbone1.rotary_emb(h, position_ids, layer_type)
        for layer_type in set(getattr(backbone1.config, "layer_types", ["full_attention"]))
    } if hasattr(backbone1.config, "layer_types") else backbone1.rotary_emb(h, position_ids)

    # Model1 layers: 0 to switch_layer-1
    shared_kv_states1 = {}
    for i, layer in enumerate(backbone1.layers[:switch_layer]):
        lt = _layer_type(backbone1, layer, i)
        layer_attn_mask = causal_mask_mapping[lt]
        layer_position_embeddings = position_embeddings[lt] if isinstance(position_embeddings, dict) else position_embeddings
        h = _run_layer(
            layer,
            h,
            per_layer_input=per_layer_inputs1[:, :, i, :] if per_layer_inputs1 is not None else None,
            attention_mask=layer_attn_mask,
            position_ids=position_ids,
            position_embeddings=layer_position_embeddings,
            cache_position=cache_position,
            shared_kv_states=shared_kv_states1,
            past_key_values=None,
        )
        if isinstance(h, tuple):
            h = h[0]

    # Model2 layers: switch_layer to L
    # Need model2's position embeddings for its layers
    per_layer_inputs2 = _per_layer_inputs(backbone2, input_ids, h)
    position_embeddings_2 = {
        layer_type: backbone2.rotary_emb(h, position_ids, layer_type)
        for layer_type in set(getattr(backbone2.config, "layer_types", ["full_attention"]))
    } if hasattr(backbone2.config, "layer_types") else backbone2.rotary_emb(h, position_ids)
    shared_kv_states2 = dict(shared_kv_states1)
    for i, layer in enumerate(backbone2.layers[switch_layer:], start=switch_layer):
        lt = _layer_type(backbone2, layer, i)
        layer_attn_mask = causal_mask_mapping[lt]
        layer_position_embeddings = position_embeddings_2[lt] if isinstance(position_embeddings_2, dict) else position_embeddings_2
        h = _run_layer(
            layer,
            h,
            per_layer_input=per_layer_inputs2[:, :, i, :] if per_layer_inputs2 is not None else None,
            attention_mask=layer_attn_mask,
            position_ids=position_ids,
            position_embeddings=layer_position_embeddings,
            cache_position=cache_position,
            shared_kv_states=shared_kv_states2,
            past_key_values=None,
        )
        if isinstance(h, tuple):
            h = h[0]

    # Final norm + lm_head from model2
    h = backbone2.norm(h)
    logits = model2.lm_head(h)

    # Gemma2 applies tanh softcapping to final logits inside its forward(),
    # but we bypass forward() here. Replicate it so bridging logits are on
    # the same scale as reference logits (which go through the full forward).
    cap = getattr(model2.config, 'final_logit_softcapping', None)
    if cap:
        logits = torch.tanh(logits / cap) * cap

    return logits


def sample_cutoffs(
    n_layers: int,
    n_cutoffs: int,
    sampling: str | list[int] = "uniform",
) -> list[int]:
    """
    Sample layer cutoff indices without replacement.

    Args:
        n_layers: total number of layers in the model
        n_cutoffs: number of cutoffs to sample
        sampling: "uniform" or list of fixed layer indices

    Returns:
        List of layer indices in [0, n_layers] (inclusive)
        - 0 means switch at the very beginning (no layers from first model)
        - n_layers means switch at the very end (all layers from first model)
    """
    if isinstance(sampling, list):
        # Fixed layer indices
        return sampling

    if sampling == "uniform":
        # Sample without replacement from [0, n_layers]
        all_layers = list(range(n_layers + 1))
        n_to_sample = min(n_cutoffs, len(all_layers))
        return sorted(random.sample(all_layers, n_to_sample))

    raise ValueError(f"Unknown sampling strategy: {sampling}")
