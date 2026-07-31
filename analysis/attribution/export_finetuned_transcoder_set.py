"""Materialize a full GemmaScope+fine-tuned transcoder set as a local circuit-tracer set.

<!-- Created by Claude Code session "implement: new 07/02 gemmascope transcoder experiments". 2026-07-10. -->

Instead of passing --finetuned_transcoder_dir/--finetuned_layers to every tool (load-time patch),
this bakes the fine-tune into a self-contained on-disk transcoder set: it loads the pretrained
GemmaScope ``TranscoderSet``, patches the fine-tuned layers in place, then writes each of the 26
layers to ``layer_{L}.safetensors`` (``SingleLayerTranscoder.state_dict`` = W_enc/W_dec/b_enc/b_dec
/activation_function.threshold, the exact format circuit-tracer's ``load_transcoder`` reads) plus a
``config.yaml``. The result loads directly via ``ReplacementModel.from_pretrained`` /
``load_transcoder_set(..., special_load_fn=None)`` and can be uploaded to the Hub — no per-run flags,
one canonical set.

Usage:
    uv run --extra viz python -m analysis.attribution.export_finetuned_transcoder_set \
        --finetune_dir <ft run dir> --finetuned_layers 0 24 25 \
        --gemmascope_width width_16k --gemmascope_l0 average_l0_76 \
        --output_dir $LARGE_ARTIFACTS_DIR/transcoder-adapters/finetuned_transcoder_sets/<name>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file

from helpers.log import logger, setup_logging

from analysis.attribution.gemmascope_finetune import patch_finetuned_layers
from analysis.attribution.run_base_adapter_comparison import (
    _load_gemmascope_transcoders,
    build_gemmascope_transcoder_config,
    resolve_gemmascope_l0_values,
)


def _state_dict_for_save(transcoder) -> dict:
    """SingleLayerTranscoder tensors in circuit-tracer's local `load_transcoder` format."""
    sd = {
        "W_enc": transcoder.W_enc.detach().cpu().contiguous(),
        "W_dec": transcoder.W_dec.detach().cpu().contiguous(),
        "b_enc": transcoder.b_enc.detach().cpu().contiguous(),
        "b_dec": transcoder.b_dec.detach().cpu().contiguous(),
    }
    if hasattr(transcoder.activation_function, "threshold"):
        sd["activation_function.threshold"] = transcoder.activation_function.threshold.detach().cpu().contiguous()
    if getattr(transcoder, "W_skip", None) is not None:
        sd["W_skip"] = transcoder.W_skip.detach().cpu().contiguous()
    return sd


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--finetune_dir", type=Path, required=True, help="Dir with finetuned_layer_*.safetensors.")
    ap.add_argument("--finetuned_layers", nargs="+", type=int, required=True, help="Layers to bake in (e.g. 0 24 25).")
    ap.add_argument("--base_model", default="google/gemma-2-2b")
    ap.add_argument("--gemmascope_repo", default="google/gemma-scope-2b-pt-transcoders")
    ap.add_argument("--gemmascope_width", default="width_16k")
    ap.add_argument("--gemmascope_l0", default="average_l0_76")
    ap.add_argument("--gemmascope_l0_match", default="nearest")
    ap.add_argument("--gemmascope_n_layers", type=int, default=26)
    ap.add_argument("--feature_input_hook", default="ln2.hook_normalized")
    ap.add_argument("--feature_output_hook", default="hook_mlp_out")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True, help="Reload the written set and check the fine-tuned layers roundtrip.")
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    l0v = resolve_gemmascope_l0_values(repo=args.gemmascope_repo, width=args.gemmascope_width, l0=args.gemmascope_l0, n_layers=args.gemmascope_n_layers, l0_match=args.gemmascope_l0_match)
    config = build_gemmascope_transcoder_config(repo=args.gemmascope_repo, width=args.gemmascope_width, l0=args.gemmascope_l0, n_layers=args.gemmascope_n_layers, model_name=args.base_model, feature_input_hook=args.feature_input_hook, feature_output_hook=args.feature_output_hook, l0_values=l0v, l0_match=args.gemmascope_l0_match)

    logger.info("Loading pretrained GemmaScope set (%s)...", config["scan_name"])
    transcoders = _load_gemmascope_transcoders(config, device=torch.device("cpu"), dtype=dtype)
    patch_finetuned_layers(transcoders, args.finetune_dir, args.finetuned_layers, torch.device("cpu"), dtype)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    layer_paths = []
    for layer in range(args.gemmascope_n_layers):
        path = args.output_dir / f"layer_{layer}.safetensors"
        save_file(_state_dict_for_save(transcoders[layer]), str(path))
        layer_paths.append(path.name)

    out_config = {
        "model_name": args.base_model,
        "model_kind": "transcoder_set",
        "feature_input_hook": args.feature_input_hook,
        "feature_output_hook": args.feature_output_hook,
        "scan_name": f"{config['scan_name']}_finetuned_L{'-'.join(map(str, args.finetuned_layers))}",
        "finetuned_from": str(args.finetune_dir),
        "finetuned_layers": args.finetuned_layers,
        "transcoders": layer_paths,
    }
    (args.output_dir / "config.yaml").write_text(yaml.safe_dump(out_config, sort_keys=False))
    logger.info("Wrote %d layers + config.yaml to %s", args.gemmascope_n_layers, args.output_dir)

    if args.verify:
        from circuit_tracer.transcoder.single_layer_transcoder import load_transcoder_set

        paths = [str(args.output_dir / p) for p in layer_paths]
        reloaded = load_transcoder_set(paths, scan_name=out_config["scan_name"], feature_input_hook=args.feature_input_hook, feature_output_hook=args.feature_output_hook, special_load_fn=None, device=torch.device("cpu"), dtype=dtype, lazy_encoder=False, lazy_decoder=False)
        for layer in args.finetuned_layers:
            a = reloaded[layer].W_enc.float()
            b = transcoders[layer].W_enc.float()
            max_diff = (a - b).abs().max().item()
            assert max_diff < 1e-4, f"layer {layer} W_enc roundtrip diff {max_diff}"
        logger.info("Verify OK: fine-tuned layers %s roundtrip (max W_enc diff < 1e-4).", args.finetuned_layers)
    logger.info("Use it: circuit-tracer / ReplacementModel with transcoder_set dir %s; or --transcoder_set %s", args.output_dir, args.output_dir)


if __name__ == "__main__":
    main()
