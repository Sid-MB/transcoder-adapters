"""Collect BASE-model (GemmaScope) transcoder feature activations on our data.

This is the base-model counterpart to ``collect_feature_activations``. Instead of
collecting from a trained transcoder-adapter checkpoint, it runs the *frozen* base
model (e.g. ``google/gemma-2-2b``) with the pretrained GemmaScope base transcoders
fully replacing each MLP, and collects the same per-feature activation examples,
histograms, and metadata for the GemmaScope features.

Why this exists (Nathan 5/28 meeting, row 3): the predone GemmaScope feature
collections are gathered on web/pretraining text, so they cannot show how base
features behave on *chat* data. Running the base transcoders over our chat + web
validation splits surfaces, for each base feature, how it generalizes to chat --
for example, why some base features fire on ``<end_of_turn>`` / turn-boundary
tokens. The output is also a drop-in ``/base_features`` scan for the base-vs-adapter
attribution overlay (``run_base_adapter_comparison --base_feature_data_path``).

What differs from ``collect_feature_activations`` is only the model front-end:
  * model:       circuit-tracer ``ReplacementModel`` (base model + GemmaScope
                 transcoders) instead of ``AutoModelForCausalLMWithTranscoder``.
  * activations: ``model.get_activations(tokens)`` per prompt (post JumpReLU)
                 instead of forward hooks on ``mlp.transcoder_enc``.
  * logit lens:  GemmaScope ``W_dec @ W_U`` instead of ``transcoder_dec @ lm_head``.
All example selection, histogram, metadata, export, and Hub-upload logic is reused
unchanged from ``collect_feature_activations`` via ``FeatureCollector.accumulate_batch``.

Chat templating note: the base tokenizer (``google/gemma-2-2b``) has no chat
template, so chat data is rendered with the instruction-tuned tokenizer
(``--prompt_tokenizer_model``, default ``google/gemma-2-2b-it``). Base and IT share
the same vocabulary, so the resulting token ids run through the base model directly.
This is what lets us study turn-boundary firing on chat data while still using the
base model (per the meeting decision "use base").

Usage (smoke, no upload):
    uv run --extra viz python -m analysis.features.collect_base_feature_activations \
        --base_model google/gemma-2-2b \
        --gemmascope_width width_16k --gemmascope_l0 average_l0_76 \
        --val_data chat:siddharthmb/2026.transcoder-adapters.lmsys-chat-1m-splits \
                   fineweb:science-of-finetuning/fineweb-1m-sample \
        --max_samples 20 --no-upload_circuit_tracer_features_to_hub

Output: identical layout to collect_feature_activations (features/, packed
circuit_tracer_features/, activation_histograms.npz, feature_metadata.json, ...).
Browse with: python -m analysis.features.visualize.feature_dashboard --data_dir <out>
"""

from __future__ import annotations

import argparse
import json
import shlex
import textwrap
from functools import partial
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from helpers.log import logger, setup_logging
from helpers.paths.output_path import generate_output_path

from analysis.features.activation_histograms import (
    DEFAULT_ACTIVATION_EXAMPLE_RANGES,
    parse_activation_example_ranges,
)
from analysis.features.batching import compute_max_batch_tokens, form_length_packed_batches
from analysis.features.collection_wandb import CollectionWandbLogger, add_collection_wandb_args
from analysis.features.collect_feature_activations import (
    DEFAULT_BATCH_SHUFFLE_SEED,
    DEFAULT_SHUFFLE_SEED,
    ArgumentDefaultsRawTextHelpFormatter,
    FeatureCollector,
    _build_example_source_metadata,
    _dataset_indices_to_process,
    _parse_val_data_entry,
    _value_to_flag_string,
    annotate_collected_features,
    export_activation_histograms,
    export_circuit_tracer_json,
    export_metadata,
    log_startup_output_size_estimate,
)
from analysis.features.load_val_data import (
    FeatureDataSourceSettings,
    load_val_data_from_settings,
)
from analysis.attribution.run_base_adapter_comparison import (
    _load_gemmascope_transcoders,
    _torch_device,
    _torch_dtype,
    build_gemmascope_transcoder_config,
    resolve_gemmascope_l0_values,
)
from models.tokens import detect_special_tokens, find_token_positions


def _slugify(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in value).strip("_") or "run"


def base_feature_model_tag(*, base_model: str, gemmascope_width: str, gemmascope_l0: str) -> str:
    """Deterministic, readable tag identifying a base/GemmaScope collection run.

    Used both for the default output directory and (as a synthetic ``model_path``)
    for the deterministic Hugging Face feature-repo name shared with the adapter
    collector's Hub-upload helpers.
    """
    return "_".join(
        [
            _slugify(base_model.rstrip("/").split("/")[-1]),
            "gemmascope",
            _slugify(gemmascope_width),
            _slugify(gemmascope_l0),
        ]
    )


def compute_gemmascope_logit_lens(
    model: Any,
    n_layers: int,
    top_k: int = 10,
) -> list[dict]:
    """Top/bottom unembedding tokens for each GemmaScope feature, per layer.

    Mirrors ``collect_feature_activations.compute_logit_lens`` but reads the
    GemmaScope decoder ``W_dec`` (``[n_features, d_model]``) and the base model's
    TransformerLens unembedding ``W_U`` (``[d_model, d_vocab]``). Like the adapter
    logit lens, this ignores the final layernorm/softcap, so it is an approximation.
    """
    logger.info("Computing GemmaScope logit lens...")
    W_U = model.W_U  # [d_model, d_vocab]
    logit_lens_data: list[dict] = []
    for layer_idx in tqdm(range(n_layers), desc="Logit lens"):
        # W_dec maps feature space -> residual/MLP-out space: [n_features, d_model]
        W_dec = model.transcoders.transcoders[layer_idx].W_dec
        with torch.no_grad():
            logits = W_dec.to(W_U.dtype) @ W_U  # [n_features, d_vocab]
            top_vals, top_ids = logits.topk(top_k, dim=1)
            bot_vals, bot_ids = logits.topk(top_k, dim=1, largest=False)
        logit_lens_data.append(
            {
                "top_ids": top_ids.cpu().tolist(),  # [n_features, top_k]
                "top_vals": top_vals.float().cpu().tolist(),
                "bot_ids": bot_ids.cpu().tolist(),
                "bot_vals": bot_vals.float().cpu().tolist(),
            }
        )
        del logits
    return logit_lens_data


def build_replay_command(parser: argparse.ArgumentParser, args: argparse.Namespace) -> str:
    """Pasteable replay command for this base-collection run."""
    parts = ["uv run --extra viz python -m analysis.features.collect_base_feature_activations"]
    for action in parser._actions:
        if not action.option_strings or action.dest in {"help"}:
            continue
        flag = _value_to_flag_string(action, getattr(args, action.dest, None))
        if flag is not None:
            parts.append(flag)
    return " \\\n    ".join(parts)


def export_run_arguments(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    """Write parsed run settings and a pasteable replay command."""
    args_payload = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    (output_dir / "collect_base_feature_activations_args.json").write_text(
        json.dumps(args_payload, indent=2, sort_keys=True) + "\n"
    )
    command_path = output_dir / "collect_base_feature_activations_command.sh"
    command_path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n\n" + build_replay_command(parser, args) + "\n"
    )
    logger.info(f"Saved parsed run arguments to {output_dir / 'collect_base_feature_activations_args.json'}")
    logger.info(f"Saved pasteable replay command to {command_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect base-model (GemmaScope) transcoder feature activations on chat/web data.",
        formatter_class=ArgumentDefaultsRawTextHelpFormatter,
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="google/gemma-2-2b",
        help="Base model to run (frozen). GemmaScope transcoders fully replace its MLPs.",
    )
    parser.add_argument(
        "--prompt_tokenizer_model",
        type=str,
        default="google/gemma-2-2b-it",
        help=(
            "Tokenizer used to render chat data and decode example tokens. The base "
            "tokenizer has no chat_template; the IT tokenizer shares the base vocab, so "
            "its token ids run through the base model directly. Set to the base model to "
            "disable chat templating (e.g. for web-only collection)."
        ),
    )
    parser.add_argument(
        "--val_data",
        type=str,
        nargs="+",
        required=True,
        help=textwrap.dedent(
            """
            Validation data source(s). Each entry is 'path' or 'domain:path'. Mixing a
            chat domain and a web domain is what isolates chat-specific base behavior:
              chat:siddharthmb/2026.transcoder-adapters.lmsys-chat-1m-splits
              fineweb:science-of-finetuning/fineweb-1m-sample
            """
        ).strip(),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: PRODUCTS_DIR/feature_data/<base-gemmascope tag>_<timestamp>).",
    )

    # GemmaScope transcoder selection (mirrors run_base_adapter_comparison).
    parser.add_argument("--gemmascope_repo", default="google/gemma-scope-2b-pt-transcoders", help="HF repo of GemmaScope PT transcoders.")
    parser.add_argument("--gemmascope_width", required=True, help="GemmaScope width folder, e.g. width_16k.")
    parser.add_argument(
        "--gemmascope_l0",
        required=True,
        help="GemmaScope L0 folder, target L0, comma-separated per-layer folders, or a YAML/JSON list/dict path. Example: average_l0_76",
    )
    parser.add_argument(
        "--gemmascope_l0_match",
        choices=["exact", "nearest"],
        default="nearest",
        help="How to interpret a single --gemmascope_l0 value. 'nearest' picks the closest available average_l0_* folder per layer.",
    )
    parser.add_argument("--gemmascope_n_layers", type=int, default=26, help="Number of base-model layers / GemmaScope transcoders (gemma-2-2b = 26).")
    parser.add_argument("--feature_input_hook", default="ln2.hook_normalized", help="TransformerLens hook feeding each transcoder's encoder.")
    parser.add_argument("--feature_output_hook", default="hook_mlp_out", help="TransformerLens hook the transcoder reconstructs.")
    parser.add_argument("--base_backend", choices=["transformerlens", "nnsight"], default="transformerlens", help="circuit-tracer backend. transformerlens exposes W_U for the logit lens.")
    parser.add_argument("--device", default="cuda", help="Torch device for the model and transcoders.")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32", "bf16", "fp16", "fp32"], default="bfloat16", help="Model/transcoder dtype.")

    # Collection knobs (same meaning as collect_feature_activations).
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples per data source (default: all).")
    parser.add_argument("--shuffle", action="store_true", help=f"Sample rows via a seeded permutation instead of sequential order. Uses seed {DEFAULT_SHUFFLE_SEED} unless --shuffle_seed is set.")
    parser.add_argument("--shuffle_seed", type=int, default=None, help=f"Seed for row sampling (overrides the default {DEFAULT_SHUFFLE_SEED}). Each --val_data source uses an independent offset.")
    parser.add_argument("--max_length", type=int, default=2048, help="Max sequence length in tokens (longer sequences truncated). Bounds the per-prompt [n_layers, seq, n_features] activation tensor.")
    parser.add_argument("--batch_size", type=int, default=16, help="Max sequences per batched forward pass. A GPU token budget is auto-computed from free memory and also caps each batch; this is the per-batch sequence ceiling. Lower if CPU-side accumulation stalls the GPU on short sequences; raise on big GPUs (e.g. B200) for throughput. 1 reproduces the old per-prompt behavior.")
    parser.add_argument("--shuffle_batches", action=argparse.BooleanOptionalAction, default=True, help="After length-aware packing, randomly reorder which batch runs first (seeded) so cheap short batches and expensive long ones interleave and tqdm ETAs are less skewed. Total compute is identical. Use --no-shuffle_batches for strict shortest-first order.")
    parser.add_argument("--shuffle_batches_seed", type=int, default=DEFAULT_BATCH_SHUFFLE_SEED, help=f"Seed for --shuffle_batches (ignored with --no-shuffle_batches). Default: {DEFAULT_BATCH_SHUFFLE_SEED}.")
    parser.add_argument("--num_shards", type=int, default=1, help="Split the work across N independent shard processes to parallelize the CPU-bound dense accumulation across cores (the bottleneck for GemmaScope). Each shard processes items where prepared_item_idx %% num_shards == shard_index, then pickles its collector state to <output_dir>/shard_<i>.pkl and skips export. Merge afterwards with --merge_shards. Requires --n_random 0 and no activation ranges (so partial states merge exactly).")
    parser.add_argument("--shard_index", type=int, default=0, help="This shard's index in [0, num_shards). Ignored when --num_shards 1.")
    parser.add_argument("--merge_shards", type=str, default=None, help="Glob of shard pickles (e.g. '<dir>/shard_*.pkl') to merge into one collection and export/upload. In this mode no data is re-read; the model is loaded only for the logit lens and the existing export path runs on the merged collector.")
    parser.add_argument("--top_k", type=int, default=20, help="Number of global top-activating examples per feature.")
    parser.add_argument("--domain_top_k", type=int, default=10, help="Number of top-activating examples per feature per domain.")
    parser.add_argument("--n_random", type=int, default=10, help="Number of random reservoir samples per feature.")
    parser.add_argument("--context_before", type=int, default=75, help="Tokens saved before each retained example (affects JSON size only).")
    parser.add_argument("--context_after", type=int, default=20, help="Tokens saved after each retained example (affects JSON size only).")
    parser.add_argument("--activation_example_ranges", type=str, default=DEFAULT_ACTIVATION_EXAMPLE_RANGES, help="Comma-separated lo:hi activation bands for medium-activation example tabs. Empty string disables.")
    parser.add_argument("--activation_range_examples_per_domain", type=int, default=4, help="Max reservoir examples per (feature, domain, range) bucket. 0 disables range tabs.")
    parser.add_argument("--relative_target_domain", type=str, default="chat", help="Target domain for relative feature scores (chat-vs-web).")
    parser.add_argument("--relative_baseline_domain", type=str, default="fineweb", help="Baseline domain for relative feature scores (chat-vs-web).")

    # Export / Hub upload (mirrors collect_feature_activations).
    parser.add_argument(
        "--no-export_circuit_tracer_features",
        dest="export_circuit_tracer_features",
        action="store_false",
        default=True,
        help="Skip writing the packed circuit-tracer local feature cache (circuit_tracer_features/).",
    )
    parser.add_argument(
        "--upload_circuit_tracer_features_to_hub",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Upload the packed feature cache to HF. Defaults to true when packed features are exported. Use --no-upload_circuit_tracer_features_to_hub for the smoke test.",
    )
    parser.add_argument(
        "--no_per_feature_json",
        action="store_true",
        help=(
            "Skip writing per-feature features/{cantor_id}.json files (and the dashboard "
            "annotation pass); write only the packed circuit_tracer_features/ cache used by "
            "the attribution overlay. Use this for dense base/GemmaScope collections where "
            "nearly all ~425k features fire, so per-feature JSON writing dominates wall-clock."
        ),
    )
    parser.add_argument("--hf_feature_repo_id", type=str, default=None, help="Explicit HF model repo ID for uploaded features (default: deterministic name from base model, GemmaScope, data, and collection hyperparameters).")
    parser.add_argument("--hub_org", type=str, default=None, help="HF namespace/org for the uploaded feature repo (default: authenticated user).")
    add_collection_wandb_args(parser)

    parser.epilog = textwrap.dedent(
        """
        Example (full run + upload, via sbatch):
          uv run --extra viz python -m analysis.features.collect_base_feature_activations \\
            --base_model google/gemma-2-2b \\
            --gemmascope_width width_16k --gemmascope_l0 average_l0_76 \\
            --val_data chat:siddharthmb/2026.transcoder-adapters.lmsys-chat-1m-splits \\
                       fineweb:science-of-finetuning/fineweb-1m-sample \\
            --max_samples 4000 --shuffle
        """
    )
    return parser


def load_base_replacement_model(args: argparse.Namespace, output_dir: Path):
    """Load the base model with GemmaScope transcoders fully replacing each MLP."""
    from circuit_tracer import ReplacementModel

    device_obj = _torch_device(args.device)
    dtype_obj = _torch_dtype(args.dtype)

    l0_values = resolve_gemmascope_l0_values(
        repo=args.gemmascope_repo,
        width=args.gemmascope_width,
        l0=args.gemmascope_l0,
        n_layers=args.gemmascope_n_layers,
        l0_match=args.gemmascope_l0_match,
    )
    config = build_gemmascope_transcoder_config(
        repo=args.gemmascope_repo,
        width=args.gemmascope_width,
        l0=args.gemmascope_l0,
        n_layers=args.gemmascope_n_layers,
        model_name=args.base_model,
        feature_input_hook=args.feature_input_hook,
        feature_output_hook=args.feature_output_hook,
        l0_values=l0_values,
        l0_match=args.gemmascope_l0_match,
    )
    (output_dir / "gemmascope_config.json").write_text(json.dumps(config, indent=2) + "\n")
    logger.info(f"GemmaScope scan: {config['scan_name']}")

    logger.info("Loading GemmaScope transcoders...")
    transcoders = _load_gemmascope_transcoders(config, device=device_obj, dtype=dtype_obj)
    logger.info(f"Loading base replacement model: {args.base_model} ({args.base_backend})")
    model = ReplacementModel.from_pretrained_and_transcoders(
        model_name=args.base_model,
        transcoders=transcoders,
        backend=args.base_backend,
        device=device_obj,
        dtype=dtype_obj,
    )
    model.eval()
    return model, device_obj


def prepare_items(
    args: argparse.Namespace,
    tokenizer: Any,
    special_tokens: Any,
    *,
    has_thinking: bool,
    shuffle_seed: int | None,
) -> tuple[list[tuple[list[int], str, dict, dict[str, Any]]], list[FeatureDataSourceSettings]]:
    """Load and tokenize all val_data sources into (tokens, domain, markers, meta) items."""
    val_data_sources = [_parse_val_data_entry(entry) for entry in args.val_data]
    data_source_settings = [
        FeatureDataSourceSettings(
            source_path=path,
            max_length=args.max_length,
            model_type="gemma2",
            domain=domain_label,
        )
        for domain_label, path in val_data_sources
    ]

    loaded_sources: list[tuple[Any, list[dict] | None]] = []
    for source_settings in data_source_settings:
        logger.info(f"Loading source {source_settings.source_path!r} (domain={source_settings.domain!r})")
        dataset, examples_meta = load_val_data_from_settings(source_settings, tokenizer)
        loaded_sources.append((dataset, examples_meta))

    items: list[tuple[list[int], str, dict, dict[str, Any]]] = []
    skipped = 0
    for source_idx, (dataset, examples_meta) in enumerate(loaded_sources):
        domain_label = val_data_sources[source_idx][0]
        indices = _dataset_indices_to_process(len(dataset), args.max_samples, shuffle_seed, source_idx)
        for idx in tqdm(indices, desc=f"Preparing source {source_idx + 1}/{len(loaded_sources)}"):
            item = dataset[idx]
            meta = examples_meta[idx] if examples_meta is not None else {}
            tokens = item["input_ids"]
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.tolist()
            if len(tokens) < 2:
                skipped += 1
                continue
            domain = meta.get("domain", domain_label or "unknown")
            markers = find_token_positions(tokens, special_tokens)
            if has_thinking and (markers["think_start"] is None or markers["think_end"] is None):
                skipped += 1
                continue
            source_metadata = _build_example_source_metadata(
                dataset=dataset,
                examples_meta=examples_meta,
                item=item,
                dataset_row_idx=idx,
                source_idx=source_idx,
                source_path=val_data_sources[source_idx][1],
                domain_label=domain_label,
                domain=domain,
                prepared_item_idx=len(items),
            )
            items.append((tokens, domain, markers, source_metadata))

    if skipped:
        logger.warning(f"Skipped {skipped} samples (too short or missing required markers).")
    return items, data_source_settings


def collect_base_layer_activations_batched(model: Any, padded_tokens: torch.Tensor) -> dict[int, torch.Tensor]:
    """Batched analog of circuit-tracer ``ReplacementModel.get_activations``.

    Runs one forward over a right-padded ``[B, max_len]`` token batch and returns
    ``{layer_idx: tensor[B, max_len, n_features]}`` of post-activation GemmaScope feature
    values, left on the model device. This mirrors
    ``TransformerLensReplacementModel._get_activation_caching_hooks`` exactly --
    ``transcoders.encode_layer(mlp_in, layer, apply_activation_function=True)`` at each
    ``feature_input_hook``, with the BOS position zeroed -- but keeps the batch dimension
    instead of squeezing it, and zeroes ``[:, zero_positions]`` per example.

    Two correctness points:
      * Right-padding is safe because gemma-2 attention is causal / sliding-window: a real
        token never attends to padding that follows it, so every real token's activation is
        identical to the single-sequence path. ``accumulate_batch`` then slices ``[:seq_len]``
        to drop the padding columns.
      * ``stop_at_layer=n_layers`` runs every block (so all per-layer hooks fire) but skips
        ``ln_final``/unembed, so we never materialize the unused ``[B, seq, vocab]`` logits.

    Activations stay on the model device; ``accumulate_batch`` slices and reduces them there.
    This matches the GPU token budget (``n_layers * n_features`` bytes/token), which is sized
    for holding all layers' activations resident during one batch.
    """
    n_layers = model.cfg.n_layers
    layer_activations: dict[int, torch.Tensor] = {}

    def _cache(acts, hook, layer):
        transcoder_acts = model.transcoders.encode_layer(
            acts, layer, apply_activation_function=True
        ).detach()
        transcoder_acts[:, model.zero_positions] = 0
        layer_activations[layer] = transcoder_acts

    hooks = [
        (f"blocks.{layer}.{model.feature_input_hook}", partial(_cache, layer=layer))
        for layer in range(n_layers)
    ]
    with torch.inference_mode(), model.hooks(hooks):
        model(padded_tokens, stop_at_layer=n_layers)
    return layer_activations


def _finalize_base_collection(
    collector,
    *,
    model,
    tokenizer,
    output_dir: Path,
    args: argparse.Namespace,
    n_layers: int,
    data_source_settings,
    hf_feature_repo_id,
    hf_feature_config,
) -> None:
    """Export + (optionally) upload a finished collector. Shared by the single-run/merge paths."""
    logit_lens_data = compute_gemmascope_logit_lens(model, n_layers)

    export_circuit_tracer_json(
        collector,
        logit_lens_data,
        tokenizer,
        output_dir,
        export_circuit_tracer_features=args.export_circuit_tracer_features,
        write_per_feature_json=not args.no_per_feature_json,
    )
    export_activation_histograms(collector, output_dir)
    export_metadata(
        collector,
        output_dir,
        target_domain=args.relative_target_domain,
        baseline_domain=args.relative_baseline_domain,
        # Store the real base model (loadable) so the dashboard's transcript-
        # reconstruction tokenizer resolves. args.model_path is a synthetic tag used
        # only for deterministic Hub naming. tokenizer_path pins the IT tokenizer that
        # actually rendered the chat data.
        model_path=args.base_model,
        tokenizer_path=args.prompt_tokenizer_model,
        data_sources=data_source_settings,
    )
    # Annotations are derived from feature_metadata.json (always written), not the
    # per-feature features/{id}.json files, so they run even with --no_per_feature_json.
    # They are also a required sidecar for the Hub upload below.
    annotate_collected_features(output_dir)

    if hf_feature_repo_id and hf_feature_config:
        from analysis.features.hub_upload import upload_circuit_tracer_features_to_hub

        upload_circuit_tracer_features_to_hub(
            repo_id=hf_feature_repo_id,
            output_dir=output_dir,
            config=hf_feature_config,
        )
        logger.info(f"  Hugging Face feature repo: https://huggingface.co/{hf_feature_repo_id}")

    logger.info(f"Done! Output written to {output_dir}")
    logger.info(f"  Browse: uv run python -m analysis.features.visualize.feature_dashboard --data_dir {output_dir}")


def run_collection(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.upload_circuit_tracer_features_to_hub is None:
        args.upload_circuit_tracer_features_to_hub = bool(args.export_circuit_tracer_features)
    if args.upload_circuit_tracer_features_to_hub and not args.export_circuit_tracer_features:
        parser.error("--upload_circuit_tracer_features_to_hub requires the packed feature cache; drop --no-export_circuit_tracer_features.")

    # Synthetic model_path makes the adapter collector's deterministic Hub-naming work.
    args.model_path = base_feature_model_tag(
        base_model=args.base_model,
        gemmascope_width=args.gemmascope_width,
        gemmascope_l0=args.gemmascope_l0,
    )

    try:
        activation_example_ranges = parse_activation_example_ranges(args.activation_example_ranges)
    except ValueError as exc:
        parser.error(str(exc))

    if args.shuffle_seed is not None:
        shuffle_seed: int | None = args.shuffle_seed
    elif args.shuffle:
        shuffle_seed = DEFAULT_SHUFFLE_SEED
    else:
        shuffle_seed = None

    if args.output_dir is None:
        output_dir = generate_output_path("feature_data", args.model_path)
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    export_run_arguments(parser, args, output_dir)

    # Reserve the Hub repo before the expensive forward passes (avoids duplicate work).
    # Shard workers (--num_shards > 1 without --merge_shards) never export/upload -- they only
    # pickle partial state -- so they must NOT reserve the repo (the merge step does that).
    is_shard_worker = args.num_shards > 1 and not args.merge_shards
    hf_feature_repo_id = None
    hf_feature_config = None
    if args.upload_circuit_tracer_features_to_hub and not is_shard_worker:
        from analysis.features.hub_upload import (
            build_feature_collection_repo_id,
            reserve_feature_collection_repo,
        )

        hf_feature_repo_id, hf_feature_config = build_feature_collection_repo_id(args)
        if reserve_feature_collection_repo(hf_feature_repo_id, hf_feature_config):
            raise RuntimeError(
                f"Feature collection already exists/reserved: https://huggingface.co/{hf_feature_repo_id}"
            )

    from transformers import AutoTokenizer
    from models.auto import ensure_fast_tokenizer

    # Require the fast tokenizer; error rather than silently fall back to the slow one.
    tokenizer = ensure_fast_tokenizer(
        AutoTokenizer.from_pretrained(args.prompt_tokenizer_model, use_fast=True),
        args.prompt_tokenizer_model,
    )
    special_tokens = detect_special_tokens(tokenizer, model_type="gemma2")
    has_thinking = special_tokens.think_start is not None and special_tokens.think_end is not None
    logger.info(f"Detected special tokens: {special_tokens}")

    model, device_obj = load_base_replacement_model(args, output_dir)
    n_layers = model.cfg.n_layers
    n_features = model.transcoders.d_transcoder
    logger.info(f"Base model: {n_layers} layers, {n_features} GemmaScope features per layer")
    if n_layers != args.gemmascope_n_layers:
        logger.warning(f"Model has {n_layers} layers but --gemmascope_n_layers={args.gemmascope_n_layers}")

    if args.merge_shards:
        import glob
        import pickle

        shard_paths = sorted(glob.glob(args.merge_shards))
        if not shard_paths:
            parser.error(f"--merge_shards matched no files: {args.merge_shards}")
        logger.info(f"Merging {len(shard_paths)} shard collectors from {args.merge_shards}")
        with open(shard_paths[0], "rb") as fh:
            payload = pickle.load(fh)
        collector = payload["collector"]
        merged_data_sources = payload["data_source_settings"]
        for shard_path in shard_paths[1:]:
            with open(shard_path, "rb") as fh:
                collector.merge_from(pickle.load(fh)["collector"])
            logger.info(f"  merged {shard_path} -> {collector.total_tokens:,} tokens total")
        _finalize_base_collection(
            collector,
            model=model,
            tokenizer=tokenizer,
            output_dir=output_dir,
            args=args,
            n_layers=n_layers,
            data_source_settings=merged_data_sources,
            hf_feature_repo_id=hf_feature_repo_id,
            hf_feature_config=hf_feature_config,
        )
        return

    items, data_source_settings = prepare_items(
        args, tokenizer, special_tokens, has_thinking=has_thinking, shuffle_seed=shuffle_seed
    )
    if not items:
        logger.error("No sequences left to process after filtering. Exiting.")
        return

    if args.num_shards > 1:
        items = [it for it in items if int(it[3]["prepared_item_idx"]) % args.num_shards == args.shard_index]
        logger.info(f"Shard {args.shard_index}/{args.num_shards}: processing {len(items)} of the prepared items")
        if not items:
            logger.error("No items fall in this shard. Exiting.")
            return

    domain_names = sorted({domain for _, domain, _, _ in items})
    logger.info(f"Collecting {len(items)} sequences across domains {domain_names}")
    log_startup_output_size_estimate(
        n_layers=n_layers,
        n_features=n_features,
        items=items,
        tokenizer=tokenizer,
        top_k=args.top_k,
        n_random=args.n_random,
        domain_top_k=args.domain_top_k,
        context_before=args.context_before,
        context_after=args.context_after,
        domain_names=domain_names,
        activation_example_ranges=activation_example_ranges,
        activation_range_examples_per_domain=args.activation_range_examples_per_domain,
    )

    collector = FeatureCollector(
        n_layers=n_layers,
        n_features=n_features,
        top_k=args.top_k,
        n_random=args.n_random,
        domain_top_k=args.domain_top_k,
        context_before=args.context_before,
        context_after=args.context_after,
        domain_names=domain_names,
        activation_example_ranges=activation_example_ranges,
        activation_range_examples_per_domain=args.activation_range_examples_per_domain,
    )

    # Batched forward (see collect_base_layer_activations_batched). circuit-tracer's
    # get_activations is single-prompt, so we run our own batched analog and feed the
    # right-padded [B, max_len, n_features] activations to accumulate_batch, which slices each
    # example's [:seq_len]. Sequences are length-sorted and packed up to --batch_size or a
    # GPU token budget (shared with the adapter collector via analysis.features.batching).
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    max_batch_tokens = compute_max_batch_tokens(
        device=device_obj,
        n_layers=n_layers,
        n_features=n_features,
        hidden=model.cfg.d_model,
    )
    batches = form_length_packed_batches(
        items,
        batch_size=args.batch_size,
        max_batch_tokens=max_batch_tokens,
        shuffle=args.shuffle_batches,
        shuffle_seed=args.shuffle_batches_seed,
    )

    wandb_logger = CollectionWandbLogger.from_args(
        args,
        run_name=output_dir.name,
        config={
            **{k: v for k, v in vars(args).items() if not k.startswith("_")},
            "collector": "base_gemmascope",
            "n_layers": n_layers,
            "n_features": n_features,
            "n_sequences": len(items),
            "n_batches": len(batches),
        },
        total_sequences=len(items),
    )
    wandb_logger.reset_timer()

    for batch in tqdm(batches, desc="Collecting base activations"):
        batch_tokens = [t for t, _, _, _ in batch]
        batch_domains = [d for _, d, _, _ in batch]
        batch_markers = [m for _, _, m, _ in batch]
        batch_source_metadata = [meta for _, _, _, meta in batch]
        batch_seq_idxs = [int(meta["prepared_item_idx"]) for meta in batch_source_metadata]

        B = len(batch_tokens)
        max_len = max(len(t) for t in batch_tokens)
        padded = torch.full((B, max_len), pad_token_id, dtype=torch.long, device=device_obj)
        for b, tokens in enumerate(batch_tokens):
            padded[b, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)

        layer_activations = collect_base_layer_activations_batched(model, padded)
        collector.accumulate_batch(
            batch_tokens=batch_tokens,
            batch_domains=batch_domains,
            batch_markers=batch_markers,
            batch_seq_idxs=batch_seq_idxs,
            batch_source_metadata=batch_source_metadata,
            layer_activations=layer_activations,
        )
        wandb_logger.log_batch(n_seqs=B, n_tokens=sum(len(t) for t in batch_tokens))
        del padded, layer_activations

    logger.info("Collection summary:")
    logger.info(f"  Total tokens: {collector.total_tokens:,}")
    logger.info(f"  Domains: {dict(collector.tokens_per_domain)}")
    logger.info(f"  Regions: {dict(collector.tokens_per_region)}")
    wandb_logger.finish(
        summary={
            "final/total_tokens": collector.total_tokens,
            "final/n_sequences": len(items),
            "final/tokens_per_domain": dict(collector.tokens_per_domain),
        }
    )

    if args.num_shards > 1:
        import pickle

        shard_path = output_dir / f"shard_{args.shard_index:03d}.pkl"
        with open(shard_path, "wb") as fh:
            pickle.dump(
                {"collector": collector, "data_source_settings": data_source_settings},
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        logger.info(
            f"Wrote shard {args.shard_index}/{args.num_shards} state to {shard_path} "
            f"({collector.total_tokens:,} tokens); skipping export. Merge with --merge_shards."
        )
        return

    _finalize_base_collection(
        collector,
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        args=args,
        n_layers=n_layers,
        data_source_settings=data_source_settings,
        hf_feature_repo_id=hf_feature_repo_id,
        hf_feature_config=hf_feature_config,
    )


def main() -> None:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args()
    run_collection(args, parser)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.error("Unhandled exception in collect_base_feature_activations", exc_info=True)
        raise
