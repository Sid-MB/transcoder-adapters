"""One-off diagnostic: measure GemmaScope base-feature density on chat vs web and
time FeatureCollector.accumulate_batch with GPU vs CPU activations.

Run (GPU, outside sandbox):
    uv run --extra viz python -m misc_scripts.diag_base_density
"""

from __future__ import annotations

import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from helpers.log import logger, setup_logging
from analysis.features.activation_histograms import (
    DEFAULT_ACTIVATION_EXAMPLE_RANGES,
    parse_activation_example_ranges,
)
from analysis.features.collect_feature_activations import FeatureCollector
from analysis.features.collect_base_feature_activations import (
    load_base_replacement_model,
    prepare_items,
)
from models.tokens import detect_special_tokens


def _fresh_collector(n_layers: int, n_features: int, domain_names: list[str]) -> FeatureCollector:
    return FeatureCollector(
        n_layers=n_layers,
        n_features=n_features,
        top_k=20,
        n_random=10,
        domain_top_k=10,
        context_before=75,
        context_after=20,
        domain_names=domain_names,
        activation_example_ranges=parse_activation_example_ranges(DEFAULT_ACTIVATION_EXAMPLE_RANGES),
        activation_range_examples_per_domain=4,
    )


def main() -> None:
    setup_logging()
    args = Namespace(
        base_model="google/gemma-2-2b",
        prompt_tokenizer_model="google/gemma-2-2b-it",
        gemmascope_repo="google/gemma-scope-2b-pt-transcoders",
        gemmascope_width="width_16k",
        gemmascope_l0="average_l0_76",
        gemmascope_l0_match="nearest",
        gemmascope_n_layers=26,
        feature_input_hook="ln2.hook_normalized",
        feature_output_hook="hook_mlp_out",
        base_backend="transformerlens",
        device="cuda",
        dtype="bfloat16",
        val_data=[
            "chat:siddharthmb/2026.transcoder-adapters.lmsys-chat-1m-splits",
            "fineweb:science-of-finetuning/fineweb-1m-sample",
        ],
        max_samples=3,
        max_length=512,
    )
    out = Path("/tmp/user/24430/diag_base_out")
    out.mkdir(parents=True, exist_ok=True)

    model, device = load_base_replacement_model(args, out)
    n_layers = model.cfg.n_layers
    n_features = model.transcoders.d_transcoder
    tokenizer = AutoTokenizer.from_pretrained(args.prompt_tokenizer_model)
    special_tokens = detect_special_tokens(tokenizer, model_type="gemma2")
    items, _ = prepare_items(args, tokenizer, special_tokens, has_thinking=False, shuffle_seed=None)
    domain_names = sorted({d for _, d, _, _ in items})

    # Group one item per domain.
    by_domain: dict[str, tuple] = {}
    for it in items:
        by_domain.setdefault(it[1], it)

    for domain, (tokens, _, markers, meta) in by_domain.items():
        t = torch.tensor([tokens], dtype=torch.long, device=device)
        torch.cuda.synchronize()
        t0 = time.time()
        _, acts = model.get_activations(t)  # [n_layers, seq, n_features]
        torch.cuda.synchronize()
        fwd_s = time.time() - t0

        seq = acts.shape[1]
        nz = int((acts > 0).sum().item())
        per_token = nz / (n_layers * seq)
        logger.info(
            f"[{domain}] seq={seq} forward={fwd_s:.2f}s nonzeros={nz:,} "
            f"mean_active_features/token/layer={per_token:.1f}"
        )

        for label, make in (
            ("GPU acts", lambda layer: acts[layer].unsqueeze(0)),
            ("CPU acts", lambda layer: acts[layer].unsqueeze(0).cpu()),
        ):
            collector = _fresh_collector(n_layers, n_features, domain_names)
            layer_activations = {layer: make(layer) for layer in range(n_layers)}
            torch.cuda.synchronize()
            t1 = time.time()
            collector.accumulate_batch(
                batch_tokens=[tokens],
                batch_domains=[domain],
                batch_markers=[markers],
                batch_seq_idxs=[0],
                batch_source_metadata=[meta],
                layer_activations=layer_activations,
            )
            torch.cuda.synchronize()
            logger.info(f"[{domain}] accumulate_batch ({label}, fresh buffers): {time.time() - t1:.2f}s")
        del acts


if __name__ == "__main__":
    main()
