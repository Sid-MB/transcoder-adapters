"""Loss functions for bridging experiments."""

import torch
import torch.nn.functional as F

from training.forward_utils import (
    _language_backbone,
    _layer_type,
    _per_layer_inputs,
    _rotary_position_embeddings,
    _run_layer,
)


def compute_kl_loss(
    logits: torch.Tensor,
    ref_logits: torch.Tensor,
    labels: torch.Tensor | None = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Compute KL divergence from reference logits for next-token prediction.

    KL(ref || model) - we want model to match ref distribution.
    Shifts logits by 1 to match LM loss convention (predicting next token).

    Args:
        logits: model logits [batch, seq_len, vocab_size]
        ref_logits: reference model logits [batch, seq_len, vocab_size]
        labels: optional labels to mask padding [batch, seq_len]
        ignore_index: index to ignore in labels

    Returns:
        Scalar KL divergence loss
    """
    # Ensure tensors are on the same device (for multi-GPU)
    device = logits.device
    ref_logits = ref_logits.to(device)
    if labels is not None:
        labels = labels.to(device)

    # Shift for next-token prediction (same as LM loss)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_ref_logits = ref_logits[..., :-1, :].contiguous()

    # Flatten to [batch * (seq_len-1), vocab_size]
    logits_flat = shift_logits.view(-1, shift_logits.size(-1))
    ref_logits_flat = shift_ref_logits.view(-1, shift_ref_logits.size(-1))

    # Prepare mask before chunked loop
    mask = None
    if labels is not None:
        shift_labels = labels[..., 1:].contiguous()
        mask = (shift_labels.view(-1) != ignore_index).float()

    # Chunked KL computation to avoid materializing full [tokens, vocab] tensors.
    # With vocab_size=256k and seq_len=10k, each full tensor is ~5GB in bf16;
    # computing log_softmax + softmax + kl_div simultaneously requires ~15GB.
    # Chunking keeps only ~0.75GB alive at a time.
    num_tokens = logits_flat.size(0)
    chunk_size = 512
    kl_sum = torch.zeros((), device=logits_flat.device, dtype=torch.float32)

    for i in range(0, num_tokens, chunk_size):
        end = min(i + chunk_size, num_tokens)
        log_probs = F.log_softmax(logits_flat[i:end], dim=-1)
        log_ref_probs = F.log_softmax(ref_logits_flat[i:end], dim=-1)
        kl_chunk = F.kl_div(log_probs, log_ref_probs, reduction='none', log_target=True).sum(dim=-1)

        if mask is not None:
            kl_chunk = kl_chunk * mask[i:end]

        kl_sum = kl_sum + kl_chunk.sum()

    if mask is not None:
        return kl_sum / mask.sum().clamp(min=1)
    return kl_sum / num_tokens


def compute_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Compute standard language modeling cross-entropy loss.

    Args:
        logits: model logits [batch, seq_len, vocab_size]
        labels: target token ids [batch, seq_len]
        ignore_index: index to ignore in loss computation

    Returns:
        Scalar cross-entropy loss
    """
    # Ensure tensors are on the same device (for multi-GPU)
    labels = labels.to(logits.device)

    # Shift for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten and compute loss
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )

    return loss


def _layer_nmse(h_adapt: torch.Tensor, h_ref: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
    """Compute NMSE for a single layer's hidden states, optionally masking padding."""
    # Ensure both tensors are on the same device
    h_ref = h_ref.to(h_adapt.device)
    if mask is not None:
        mask_3d = mask.unsqueeze(-1).to(h_adapt.device)  # [batch, seq, 1]
        diff = (h_adapt - h_ref) * mask_3d
        ref_masked = h_ref * mask_3d
        n_elements = torch.clamp(mask_3d.sum() * h_adapt.shape[-1], min=1)
        mse = diff.pow(2).sum() / n_elements
        norm = ref_masked.pow(2).sum() / n_elements
    else:
        mse = (h_adapt - h_ref).pow(2).mean()
        norm = h_ref.pow(2).mean()
    return mse / (norm + 1e-8)


def compute_nmse_loss(
    model,
    ref_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    return_layerwise: bool = False,
):
    """
    Compute NMSE between hidden states at all layers, memory-efficiently.

    Runs both models layer-by-layer, computing NMSE incrementally without
    storing all hidden states. Only 2 hidden states are kept at a time.

    Args:
        model: adapter model (gradients enabled)
        ref_model: reference model (frozen)
        input_ids: input token ids [batch, seq_len]
        attention_mask: attention mask [batch, seq_len]
        return_layerwise: if True, also return dict of per-layer NMSE values

    Returns:
        If return_layerwise=False: Scalar NMSE loss averaged over all layers
        If return_layerwise=True: (scalar_loss, dict of layer -> nmse_value)
    """
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

    backbone = _language_backbone(model)
    ref_backbone = _language_backbone(ref_model)
    n_layers = len(backbone.layers)
    device = model.get_input_embeddings().weight.device

    # Get embeddings
    h_adapt = backbone.embed_tokens(input_ids)
    per_layer_inputs_adapt = _per_layer_inputs(backbone, input_ids, h_adapt)
    with torch.no_grad():
        h_ref = ref_backbone.embed_tokens(input_ids)
        per_layer_inputs_ref = _per_layer_inputs(ref_backbone, input_ids, h_ref)

    # Setup position ids and cache position
    seq_len = h_adapt.shape[1]
    cache_position = torch.arange(seq_len, device=device)
    position_ids = cache_position.unsqueeze(0)

    # Create causal masks (full + sliding window if needed)
    mask_kwargs = {
        "config": backbone.config,
        "input_embeds": h_adapt,
        "attention_mask": attention_mask,
        "cache_position": cache_position,
        "past_key_values": None,
        "position_ids": position_ids,
    }
    causal_mask_mapping = {
        "full_attention": create_causal_mask(**mask_kwargs),  # type: ignore
    }
    if any(_layer_type(backbone, layer, i) == "sliding_attention" for i, layer in enumerate(backbone.layers)):
        causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)  # type: ignore

    # Position embeddings
    position_embeddings_adapt = _rotary_position_embeddings(backbone, h_adapt, position_ids)
    with torch.no_grad():
        position_embeddings_ref = _rotary_position_embeddings(ref_backbone, h_ref, position_ids)

    # NMSE on embeddings (layer 0)
    layer_nmse_0 = _layer_nmse(h_adapt, h_ref.detach(), mask=attention_mask)
    nmse_total = layer_nmse_0.to(device)
    layerwise = {0: layer_nmse_0.item()} if return_layerwise else None

    # Layer by layer
    shared_kv_states_adapt = {}
    shared_kv_states_ref = {}
    for i, (layer_adapt, layer_ref) in enumerate(zip(backbone.layers, ref_backbone.layers)):
        lt = _layer_type(backbone, layer_adapt, i)
        layer_mask = causal_mask_mapping[lt]
        layer_position_embeddings_adapt = (
            position_embeddings_adapt[lt] if isinstance(position_embeddings_adapt, dict) else position_embeddings_adapt
        )
        layer_position_embeddings_ref = (
            position_embeddings_ref[lt] if isinstance(position_embeddings_ref, dict) else position_embeddings_ref
        )

        h_adapt = _run_layer(
            layer_adapt,
            h_adapt,
            per_layer_input=per_layer_inputs_adapt[:, :, i, :] if per_layer_inputs_adapt is not None else None,
            attention_mask=layer_mask,
            position_ids=position_ids,
            position_embeddings=layer_position_embeddings_adapt,
            cache_position=cache_position,
            shared_kv_states=shared_kv_states_adapt,
            past_key_values=None,
        )
        if isinstance(h_adapt, tuple):
            h_adapt = h_adapt[0]

        with torch.no_grad():
            h_ref = _run_layer(
                layer_ref,
                h_ref,
                per_layer_input=per_layer_inputs_ref[:, :, i, :] if per_layer_inputs_ref is not None else None,
                attention_mask=layer_mask,
                position_ids=position_ids,
                position_embeddings=layer_position_embeddings_ref,
                cache_position=cache_position,
                shared_kv_states=shared_kv_states_ref,
                past_key_values=None,
            )
            if isinstance(h_ref, tuple):
                h_ref = h_ref[0]

        # Accumulate NMSE (move to consistent device for multi-GPU)
        layer_nmse_i = _layer_nmse(h_adapt, h_ref.detach(), mask=attention_mask)
        nmse_total = nmse_total + layer_nmse_i.to(device)
        if return_layerwise:
            assert layerwise is not None
            layerwise[i + 1] = layer_nmse_i.item()

    # Average over all layers (embeddings + n_layers)
    avg_nmse = nmse_total / (n_layers + 1)

    if return_layerwise:
        return avg_nmse, layerwise
    return avg_nmse
