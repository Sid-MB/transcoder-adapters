"""Shared length-aware batching for the feature-activation collectors.

Both front-ends — the adapter collector (``collect_feature_activations``) and the
base/GemmaScope collector (``collect_base_feature_activations``) — turn a flat list
of prepared sequences into padding-efficient batches the same way: sort by length so
similar sizes pack together (less padding waste), then greedily fill each batch up to
``batch_size`` sequences or a GPU-memory token budget, whichever comes first. This is
factored here so both share one packing implementation (and one place to fix it).

An ``item`` is the 4-tuple the collectors build per sequence:
``(tokens: list[int], domain: str, markers: dict, source_metadata: dict)``; only
``len(item[0])`` (the token count) is read here, so the rest is opaque.
"""

from __future__ import annotations

from typing import Any

import torch

from helpers.log import logger

CollectionItem = tuple[list[int], str, dict, dict[str, Any]]


def compute_max_batch_tokens(
    *,
    device: Any,
    n_layers: int,
    n_features: int,
    hidden: int,
    bytes_per_element: int = 2,
    extra_bytes_per_token: int = 0,
    slack: float = 1.35,
    safety_margin: float = 0.55,
) -> int:
    """Padded-token budget for one forward pass, derived from free GPU memory.

    The dominant term is the per-layer hook tensors: each layer keeps a
    ``[batch, seq, n_features]`` activation (``bytes_per_element`` each, bf16 = 2)
    alive until the forward finishes, so ``n_layers * n_features`` bytes per padded
    token. ``extra_bytes_per_token`` covers any front-end-specific term the hooks miss
    (e.g. the base collector's full ``[batch, seq, vocab]`` logits, which the adapter
    collector avoids by skipping ``lm_head``). ``slack`` pads the hook estimate;
    ``safety_margin`` is the fraction of *free* memory we're willing to spend.
    """
    hook_bytes_per_token = n_layers * n_features * bytes_per_element
    scratch_bytes_per_token = max(4096, hidden * 16)
    total_bytes_per_token = (
        int(hook_bytes_per_token * slack) + scratch_bytes_per_token + extra_bytes_per_token
    )
    free_bytes = torch.cuda.mem_get_info(device)[0]
    max_batch_tokens = int(free_bytes * safety_margin / total_bytes_per_token)
    logger.info(
        f"GPU free memory: {free_bytes / 1e9:.1f} GB, "
        f"per-token budget: {total_bytes_per_token / 1e6:.1f} MB "
        f"(hooks {hook_bytes_per_token / 1e6:.1f} × {slack:.2f}"
        f"{f' + logits {extra_bytes_per_token / 1e6:.1f}' if extra_bytes_per_token else ''} "
        f"+ scratch {scratch_bytes_per_token / 1e6:.1f}), token budget: {max_batch_tokens:,}"
    )
    return max_batch_tokens


def form_length_packed_batches(
    items: list[CollectionItem],
    *,
    batch_size: int,
    max_batch_tokens: int,
    shuffle: bool,
    shuffle_seed: int,
) -> list[list[CollectionItem]]:
    """Sort by length and greedily pack into batches; optionally shuffle run order.

    A new batch is started when adding the next item would exceed ``batch_size``
    sequences or push the padded token count (``len(batch) * max_len_in_batch``) past
    ``max_batch_tokens``. With ``shuffle`` on, the *membership* of batches is unchanged
    but which batch runs first is permuted (seeded ``torch.randperm``), so cheap short
    batches and expensive long ones interleave and tqdm ETAs are less skewed.
    """
    items = sorted(items, key=lambda x: len(x[0]))
    logger.info(
        f"Sorted {len(items)} sequences by length "
        f"(shortest={len(items[0][0])}, longest={len(items[-1][0])})"
    )

    batches: list[list[CollectionItem]] = []
    current_batch: list[CollectionItem] = []
    current_max_len = 0
    for item in items:
        item_len = len(item[0])
        new_max_len = max(current_max_len, item_len)
        padded_tokens = (len(current_batch) + 1) * new_max_len
        if current_batch and (len(current_batch) >= batch_size or padded_tokens > max_batch_tokens):
            batches.append(current_batch)
            current_batch = [item]
            current_max_len = item_len
        else:
            current_batch.append(item)
            current_max_len = new_max_len
    if current_batch:
        batches.append(current_batch)

    batch_sizes = [len(b) for b in batches]
    logger.info(
        f"Formed {len(batches)} batches (sizes {min(batch_sizes)}-{max(batch_sizes)}, "
        f"token budget={max_batch_tokens:,})"
    )

    if shuffle:
        g = torch.Generator()
        g.manual_seed(int(shuffle_seed))
        order = torch.randperm(len(batches), generator=g).tolist()
        batches = [batches[i] for i in order]
        logger.info(
            "Shuffling batch execution order: %s batches permuted with torch.randperm "
            "(seed=%s). Batch membership is unchanged; only run order differs so step times "
            "are mixed. Disable with --no-shuffle_batches for shortest-batches-first order.",
            len(batches),
            shuffle_seed,
        )
    else:
        logger.info(
            "Batch execution order: sequential after length-aware packing (shortest batches "
            "first; progress may look fast early then slow). Enable default --shuffle_batches to "
            "interleave cheap and expensive steps."
        )
    return batches
