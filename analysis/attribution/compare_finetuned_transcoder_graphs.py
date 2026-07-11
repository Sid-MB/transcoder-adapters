"""Circuit-tracer attribution graphs with ORIGINAL vs re-fine-tuned GemmaScope transcoders.

<!-- Created by Claude Code session "implement: new 07/02 gemmascope transcoder experiments". 2026-07-09. -->

Follow-up to the re-fine-tune (analysis/features/finetune_transcoder_shift.py): does swapping the
fine-tuned transcoders into circuit-tracer produce cleaner graphs? For each prompt we build the
attribution graph twice on the base model -- once with the pretrained GemmaScope transcoders and
once with the fine-tuned layers (0/24/25 by default) patched in -- and compare graph composition,
the graph-level analog of reconstruction fidelity being the **error node** (mlp reconstruction
error). Lower error-node fraction / mass with the fine-tuned set = the fine-tune helped the graph.

Reuses the exact attribution path from ``run_base_adapter_comparison.run_base_attribution``
(load GemmaScope TranscoderSet -> ReplacementModel -> ``circuit_tracer.attribute`` ->
create_graph_files). The fine-tuned weights are the ``finetuned_layer_*.safetensors`` written by
the fine-tune run; we copy W_enc/W_dec/b_enc/b_dec into the in-memory transcoders in place.

Both graph dirs can be served side by side (commands printed at the end).

Usage:
    uv run --extra viz python -m analysis.attribution.compare_finetuned_transcoder_graphs \
        --finetune_dir <ft run dir> --prompts analysis/attribution/prompts/interesting_small \
        --ft_layers 0 24 25 --gemmascope_width width_16k --gemmascope_l0 average_l0_76 \
        --max_feature_nodes 256 --max_n_logits 5
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

import torch

from helpers.log import logger, setup_logging
from helpers.paths.output_path import generate_output_path

from analysis.attribution.run_attribution import _list_prompt_files, load_prompt_file
from analysis.attribution.run_base_adapter_comparison import (
    _call_create_graph_files,
    _graph_slug,
    _load_gemmascope_transcoders,
    _model_type_for_prompt_loader,
    _prompt_tokenizer_for_base,
    _torch_device,
    _torch_dtype,
    build_gemmascope_transcoder_config,
    resolve_gemmascope_l0_values,
)


def _is_error_node(node: dict) -> bool:
    return "error" in str(node.get("feature_type") or "")


def _is_feature_node(node: dict) -> bool:
    ft = str(node.get("feature_type") or "")
    return "transcoder" in ft or ft == "cross layer transcoder"


def graph_composition(graph_json: Path) -> dict:
    """Error-node vs feature-node counts (+ error fraction) from one graph JSON."""
    payload = json.loads(graph_json.read_text())
    nodes = payload.get("nodes", [])
    err = sum(1 for n in nodes if _is_error_node(n))
    feat = sum(1 for n in nodes if _is_feature_node(n))
    return {"feature_nodes": feat, "error_nodes": err, "total_nodes": len(nodes), "error_fraction": err / max(1, err + feat)}


from analysis.attribution.gemmascope_finetune import patch_finetuned_layers as patch_transcoders


def run_graphs(*, model, prompts, output_dir: Path, run_name: str, args, prompt_tokenizer, model_type) -> dict[str, dict]:
    from circuit_tracer import attribute

    output_dir.mkdir(parents=True, exist_ok=True)
    comps: dict[str, dict] = {}
    for i, prompt_path in enumerate(prompts, 1):
        slug = _graph_slug(run_name, prompt_path)
        logger.info("[%s %d/%d] %s", run_name, i, len(prompts), slug)
        prompt_tokens, target_token, _ = load_prompt_file(prompt_path, prompt_tokenizer, prompt_format=args.prompt_format, model_type=model_type)
        graph = attribute(prompt=prompt_tokens, model=model, max_n_logits=args.max_n_logits, batch_size=args.batch_size, max_feature_nodes=args.max_feature_nodes, offload=args.offload, verbose=False)
        graph.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _call_create_graph_files(graph=graph, slug=slug, output_dir=output_dir, scan=args.scan_name, node_threshold=args.node_threshold, edge_threshold=args.edge_threshold)
        comps[prompt_path.stem] = graph_composition(output_dir / f"{slug}.json")
        logger.info("  %s: %s", prompt_path.stem, comps[prompt_path.stem])
    return comps


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--finetune_dir", type=Path, required=True, help="Fine-tune run dir with finetuned_layer_*.safetensors.")
    p.add_argument("--ft_layers", nargs="+", type=int, default=[0, 24, 25], help="Layers replaced with fine-tuned weights.")
    p.add_argument("--prompts", type=Path, default=Path("analysis/attribution/prompts/interesting_small"), help="Dir of .txt prompt files (or a single .txt).")
    p.add_argument("--max_prompts", type=int, default=None, help="Cap the number of prompts (both sides) for a quick comparison.")
    p.add_argument("--base_model", default="google/gemma-2-2b")
    p.add_argument("--gemmascope_repo", default="google/gemma-scope-2b-pt-transcoders")
    p.add_argument("--gemmascope_width", required=True)
    p.add_argument("--gemmascope_l0", required=True)
    p.add_argument("--gemmascope_l0_match", default="nearest", choices=["exact", "nearest"])
    p.add_argument("--gemmascope_n_layers", type=int, default=26)
    p.add_argument("--feature_input_hook", default="ln2.hook_normalized")
    p.add_argument("--feature_output_hook", default="hook_mlp_out")
    p.add_argument("--prompt_format", default="chat", choices=["chat", "raw"])
    p.add_argument("--prompt_tokenizer_model", default="google/gemma-2-2b-it")
    p.add_argument("--backend", default="transformerlens", choices=["transformerlens", "nnsight"])
    p.add_argument("--max_feature_nodes", type=int, default=256)
    p.add_argument("--max_n_logits", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--node_threshold", type=float, default=0.8)
    p.add_argument("--edge_threshold", type=float, default=0.98)
    p.add_argument("--offload", default=None, choices=["cpu", "disk", "none"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--output_dir", type=Path, default=None)
    return p


def main() -> None:
    setup_logging()
    args = build_parser().parse_args()
    if args.offload == "none":
        args.offload = None
    device, dtype = _torch_device(args.device), _torch_dtype(args.dtype)
    output_dir = args.output_dir or generate_output_path("transcoder_finetune_graphs", f"L{'-'.join(map(str, args.ft_layers))}")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Invocation: %s", " ".join(shlex.quote(a) for a in [sys.executable, *sys.argv]))
    logger.info("Output dir: %s", output_dir)

    prompts = _list_prompt_files(args.prompts) if args.prompts.is_dir() else [args.prompts]
    if args.max_prompts is not None:
        prompts = prompts[: args.max_prompts]
    logger.info("Prompts: %s", [p.stem for p in prompts])

    l0_values = resolve_gemmascope_l0_values(repo=args.gemmascope_repo, width=args.gemmascope_width, l0=args.gemmascope_l0, n_layers=args.gemmascope_n_layers, l0_match=args.gemmascope_l0_match)
    config = build_gemmascope_transcoder_config(repo=args.gemmascope_repo, width=args.gemmascope_width, l0=args.gemmascope_l0, n_layers=args.gemmascope_n_layers, model_name=args.base_model, feature_input_hook=args.feature_input_hook, feature_output_hook=args.feature_output_hook, l0_values=l0_values, l0_match=args.gemmascope_l0_match)
    args.scan_name = config["scan_name"]

    from circuit_tracer import ReplacementModel

    transcoders = _load_gemmascope_transcoders(config, device=device, dtype=dtype)
    model = ReplacementModel.from_pretrained_and_transcoders(model_name=args.base_model, transcoders=transcoders, backend=args.backend, device=device, dtype=dtype)
    model_type = _model_type_for_prompt_loader(model, args.base_model)
    prompt_tokenizer = _prompt_tokenizer_for_base(model.tokenizer, prompt_format=args.prompt_format, prompt_tokenizer_model=args.prompt_tokenizer_model)

    # ORIGINAL graphs.
    logger.info("=== ORIGINAL GemmaScope transcoders ===")
    original = run_graphs(model=model, prompts=prompts, output_dir=output_dir / "original", run_name="original", args=args, prompt_tokenizer=prompt_tokenizer, model_type=model_type)

    # Patch in place -> FINE-TUNED graphs.
    logger.info("=== FINE-TUNED transcoders (layers %s) ===", args.ft_layers)
    patch_transcoders(model.transcoders, args.finetune_dir, args.ft_layers, device, dtype)
    finetuned = run_graphs(model=model, prompts=prompts, output_dir=output_dir / "finetuned", run_name="finetuned", args=args, prompt_tokenizer=prompt_tokenizer, model_type=model_type)

    # Re-tag with LOCAL scans (leading '/') so the frontend loads feature examples from the
    # local --features_dir, add dropdown title prefixes, and flag the re-collected FT layers.
    from analysis.attribution.retag_graphs_for_local_features import retag_graph, retag_metadata_index

    ft_tag = "-".join(map(str, args.ft_layers))
    for sub, scan, title, marks in (
        ("original", "/gemmascope_original", "[ORIGINAL GemmaScope]", set()),
        ("finetuned", f"/gemmascope_finetuned_L{ft_tag}", f"[FINE-TUNED L{ft_tag}]", set(args.ft_layers)),
    ):
        gdir = output_dir / sub
        for gp in sorted(gdir.glob("*.json")):
            if gp.name == "graph-metadata.json":
                continue
            retag_graph(gp, scan=scan, title_prefix=title, mark_layers=marks, mark_text="re-collected on fine-tuned transcoder")
        if (gdir / "graph-metadata.json").exists():
            retag_metadata_index(gdir / "graph-metadata.json", scan=scan, title_prefix=title)

    # Compare.
    comparison = {"config": {"finetune_dir": str(args.finetune_dir), "ft_layers": args.ft_layers, "scan_name": args.scan_name, "output_dir": str(output_dir)}, "by_prompt": {}}
    lines = ["# Circuit-tracer graphs: original vs fine-tuned GemmaScope transcoders", "", f"Base model `{args.base_model}`, fine-tuned layers {args.ft_layers}. Error node = MLP reconstruction error; lower error fraction = cleaner graph.", "", "| prompt | orig error-frac | ft error-frac | orig err/feat | ft err/feat |", "|---|---|---|---|---|"]
    for stem in original:
        o, f = original[stem], finetuned.get(stem, {})
        comparison["by_prompt"][stem] = {"original": o, "finetuned": f}
        lines.append(f"| {stem} | {o['error_fraction']:.3f} | {f.get('error_fraction', float('nan')):.3f} | {o['error_nodes']}/{o['feature_nodes']} | {f.get('error_nodes','?')}/{f.get('feature_nodes','?')} |")
    (output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n")
    logger.info("Comparison written: %s", output_dir / "comparison.md")
    logger.info("Serve side by side:")
    logger.info("  uv run --extra viz circuit-tracer serve --graph_file_dir %s --port 8050", output_dir / "original")
    logger.info("  uv run --extra viz circuit-tracer serve --graph_file_dir %s --port 8051", output_dir / "finetuned")


if __name__ == "__main__":
    main()
