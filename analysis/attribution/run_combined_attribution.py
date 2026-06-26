"""Full-replacement base+adapter attribution: real error nodes in ONE graph.

This is the full-replacement pivot from the 5/28 notes (``MLP(x) = T_base(x) + T_finetune(x) + Err``).
Instead of the RelP path (which keeps the real base MLP and zero-pads its error nodes), this
replaces each MLP with ``T_base`` (GemmaScope) + ``T_adapter`` (our trained adapter) and runs
circuit-tracer's STANDARD linear attribution. Because the MLP is fully replaced by transcoders,
that attribution produces real reconstruction-error nodes (``error = MLP_true - T_base - T_adapter``)
for free -- the "error triangles" -- and no RelP is needed.

The result is ONE graph containing base features (combined feature index < n_base), adapter
features (>= n_base), and error nodes. ``tag_combined_graph`` labels each feature node base/adapter,
rewrites adapter feature indices to their source-local value so the frontend loads examples from
``/adapter_features`` (vs ``/base_features``), and keeps the most influential error nodes (rendered
as triangles by the comparison frontend).

Combined transcoder construction + the algebraic equivalence ``combined == T_base + T_adapter`` are
validated in tests/test_combined_transcoders.py and misc_scripts/validate_combined_transcoder.py.

Example:
  uv run --extra viz python -m analysis.attribution.run_combined_attribution \\
    --adapter_checkpoint siddharthmb/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860 \\
    --base_model google/gemma-2-2b \\
    --prompts analysis/attribution/prompts/interesting_small/capital_paris.txt \\
    --prompt_format chat --gemmascope_width width_16k --gemmascope_l0 average_l0_76 \\
    --base_feature_data_path mntss/gemma-scope-transcoders \\
    --max_feature_nodes 256 --batch_size 4 --max_n_logits 5
"""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path
from typing import Any, Literal

from analysis.attribution.combined_transcoders import (
    build_combined_transcoder_set,
    extract_adapter_transcoder_set,
    n_base_features,
)
from analysis.attribution.run_attribution import (
    _list_prompt_files,
    load_prompt_file,
)
from analysis.attribution.run_base_adapter_comparison import (
    ADAPTER_NODE_SHAPE,
    BASE_NODE_SHAPE,
    LOCAL_BASE_FEATURE_SCAN,
    LOCAL_FEATURE_SCAN,
    SOURCE_ADAPTER,
    SOURCE_BASE,
    SOURCE_SHARED,
    _call_create_graph_files,
    _graph_slug,
    _load_gemmascope_transcoders,
    _model_type_for_prompt_loader,
    _prompt_tokenizer_for_base,
    _safe_empty_cuda_cache,
    _save_raw_graph,
    _torch_device,
    _torch_dtype,
    build_gemmascope_transcoder_config,
    resolve_gemmascope_l0_values,
)
from analysis.attribution.run_circuit_tracer_pipeline import (
    is_hf_feature_ref,
    normalize_hf_feature_ref,
)
from helpers.log import logger, setup_logging
from helpers.paths.output_path import generate_output_path

ERROR_NODE_SHAPE = "error"
DEFAULT_MAX_ERROR_NODES = 32


def cantor_pair(x: int, y: int) -> int:
    return (x + y) * (x + y + 1) // 2 + y


def cantor_unpair(z: int) -> tuple[int, int]:
    w = int((math.isqrt(8 * z + 1) - 1) // 2)
    t = (w * w + w) // 2
    y = z - t
    x = w - y
    return x, y


def _is_feature_node(node: dict[str, Any]) -> bool:
    return str(node.get("feature_type") or "") == "cross layer transcoder"


def _is_error_node(node: dict[str, Any]) -> bool:
    return "error" in str(node.get("feature_type") or "")


def _within_layer_index(node: dict[str, Any]) -> tuple[int, int]:
    """Return (layer, within-layer combined feature index) for a feature node.

    circuit-tracer feature nodes carry node_id "layer_feature_ctx" whose middle component is
    the raw within-layer feature index (validated), and a cantor-paired ``feature`` field
    encoding (layer, within-layer feature). Prefer the node_id form (matches the existing
    ``_source_feature_index`` convention); fall back to unpairing ``feature``.
    """
    node_id = str(node.get("node_id") or "")
    parts = node_id.split("_")
    if len(parts) >= 3 and parts[0].lstrip("-").isdigit() and parts[1].isdigit():
        return int(parts[0]), int(parts[1])
    feature = node.get("feature")
    if isinstance(feature, int):
        return cantor_unpair(feature)
    raise ValueError(f"Cannot determine feature index for node {node_id!r}")


def _influence(node: dict[str, Any]) -> float:
    try:
        return abs(float(node.get("influence") or 0.0))
    except (TypeError, ValueError):
        return 0.0


def tag_combined_graph(
    payload: dict[str, Any],
    *,
    n_base: int,
    base_feature_scan: str | None,
    adapter_feature_scan: str | None,
    max_error_nodes: int | None = DEFAULT_MAX_ERROR_NODES,
) -> dict[str, Any]:
    """Tag a combined-attribution graph: base/adapter feature nodes + influence-ranked errors.

    - Feature nodes get ``source_model`` (base/adapter) and ``node_shape``. Adapter nodes
      (within-layer index >= n_base) have their ``feature`` rewritten to the source-local
      cantor id ``cantor(layer, within - n_base)`` so the frontend loads adapter examples
      from ``/adapter_features``; base nodes keep their (GemmaScope) index for ``/base_features``.
    - Error nodes are ranked by |influence| and only the top ``max_error_nodes`` are kept
      (the rest, and their links, are dropped); the frontend renders the survivors as triangles.
    """
    nodes = list(payload["nodes"])
    kept_error_ids: set[str] = set()

    if max_error_nodes is not None:
        error_nodes = [node for node in nodes if _is_error_node(node)]
        ranked = sorted(error_nodes, key=lambda node: -_influence(node))
        kept_error_ids = {str(node["node_id"]) for node in ranked[:max_error_nodes]}
        dropped_error_ids = {str(node["node_id"]) for node in ranked[max_error_nodes:]}
    else:
        dropped_error_ids = set()

    tagged_nodes: list[dict[str, Any]] = []
    base_count = adapter_count = error_count = 0
    for node in nodes:
        if _is_error_node(node):
            if str(node["node_id"]) in dropped_error_ids:
                continue
            node["source_model"] = SOURCE_SHARED
            node["node_shape"] = ERROR_NODE_SHAPE
            node["isError"] = True
            error_count += 1
        elif _is_feature_node(node):
            layer, within = _within_layer_index(node)
            if within < n_base:
                node["source_model"] = SOURCE_BASE
                node["node_shape"] = BASE_NODE_SHAPE
                node["source_feature_id"] = str(within)
                base_count += 1
            else:
                local = within - n_base
                node["source_model"] = SOURCE_ADAPTER
                node["node_shape"] = ADAPTER_NODE_SHAPE
                node["source_feature_id"] = str(local)
                node["feature"] = cantor_pair(layer, local)
                adapter_count += 1
        tagged_nodes.append(node)

    kept_ids = {str(node["node_id"]) for node in tagged_nodes}
    tagged_links = [
        link
        for link in payload["links"]
        if str(link.get("source")) in kept_ids and str(link.get("target")) in kept_ids
    ]

    metadata = dict(payload["metadata"])
    comparison = dict(metadata.get("comparison") or {})
    comparison.update(
        {
            "mode": "combined_full_replacement",
            "n_base_features": n_base,
            "base_feature_scan": base_feature_scan,
            "adapter_feature_scan": adapter_feature_scan,
            "node_shapes": {
                SOURCE_BASE: BASE_NODE_SHAPE,
                SOURCE_ADAPTER: ADAPTER_NODE_SHAPE,
                "error": ERROR_NODE_SHAPE,
            },
            "node_counts": {
                "base_features": base_count,
                "adapter_features": adapter_count,
                "error_nodes": error_count,
            },
            "max_error_nodes": max_error_nodes,
        }
    )
    metadata["comparison"] = comparison
    metadata["scan"] = "base-vs-adapter"
    return {
        **payload,
        "metadata": metadata,
        "nodes": tagged_nodes,
        "links": tagged_links,
    }


def build_combined_replacement_model(args: argparse.Namespace):
    """Load base model + (GemmaScope ⊕ adapter) combined transcoders into a ReplacementModel."""
    import torch
    from circuit_tracer import ReplacementModel
    from models.auto import AutoModelForCausalLMWithTranscoder

    device = _torch_device(args.device)
    dtype = _torch_dtype(args.dtype)

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
    logger.info(f"Loading GemmaScope base transcoders: {config['scan_name']}")
    base_set = _load_gemmascope_transcoders(config, device=device, dtype=dtype)
    n_base = n_base_features(base_set)

    logger.info(f"Loading adapter checkpoint for transcoder extraction: {args.adapter_checkpoint}")
    adapter_model = AutoModelForCausalLMWithTranscoder.from_pretrained(
        args.adapter_checkpoint,
        dtype=dtype,
        device_map={"": device},
    )
    adapter_set = extract_adapter_transcoder_set(
        adapter_model,
        feature_input_hook=args.feature_input_hook,
        feature_output_hook=args.feature_output_hook,
        device=device,
        dtype=dtype,
    )
    n_adapter = adapter_set[0].d_transcoder
    del adapter_model
    _safe_empty_cuda_cache()

    logger.info(f"Building combined transcoder set: {n_base} base + {n_adapter} adapter features/layer")
    combined_set = build_combined_transcoder_set(
        base_set,
        adapter_set,
        scan_name="base-vs-adapter",
        device=device,
        dtype=dtype,
    )

    logger.info(f"Loading combined ReplacementModel ({args.base_backend})")
    model = ReplacementModel.from_pretrained_and_transcoders(
        model_name=args.base_model,
        transcoders=combined_set,
        backend=args.base_backend,
        device=device,
        dtype=dtype,
    )
    return model, n_base


def _resolve_feature_scans(args: argparse.Namespace) -> tuple[str | None, str | None]:
    base_scan = None
    if args.base_feature_data_path:
        base_scan = (
            normalize_hf_feature_ref(args.base_feature_data_path)
            if is_hf_feature_ref(args.base_feature_data_path)
            else LOCAL_BASE_FEATURE_SCAN
        )
    adapter_scan = None
    if args.adapter_feature_data_path:
        adapter_scan = (
            normalize_hf_feature_ref(args.adapter_feature_data_path)
            if is_hf_feature_ref(args.adapter_feature_data_path)
            else LOCAL_FEATURE_SCAN
        )
    return base_scan, adapter_scan


def run_combined_attribution(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from circuit_tracer import attribute

    output_dir = args.output_dir or generate_output_path(
        "combined_attribution_graphs",
        f"{args.run_name}",
        consistent=True,
    )
    # Resolve to an absolute path: circuit_tracer's add_graph_metadata asserts on
    # os.path.dirname(output_dir), which is empty for a single-component relative dir
    # like ./graph_x (Path strips the "./"), so a bare --output_dir would crash on write.
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_paths = _list_prompt_files(args.prompts)
    if not prompt_paths:
        raise ValueError(f"No .txt prompt files found in {args.prompts}")

    base_feature_scan, adapter_feature_scan = _resolve_feature_scans(args)
    max_error_nodes = None if args.max_error_nodes < 0 else args.max_error_nodes

    model, n_base = build_combined_replacement_model(args)
    model_type = _model_type_for_prompt_loader(model, args.base_model)
    prompt_tokenizer = _prompt_tokenizer_for_base(
        model.tokenizer,
        prompt_format=args.prompt_format,
        prompt_tokenizer_model=args.prompt_tokenizer_model,
    )

    results: dict[str, str] = {}
    for index, prompt_path in enumerate(prompt_paths, 1):
        slug = _graph_slug(args.run_name, prompt_path)
        graph_json_path = output_dir / f"{slug}.json"
        if graph_json_path.exists():
            logger.info(f"[{index}/{len(prompt_paths)}] {slug}: already exists, skipping")
            results[prompt_path.stem] = "skipped"
            continue

        prompt_tokens, target_token, prompt_text = load_prompt_file(
            prompt_path,
            prompt_tokenizer,
            prompt_format=args.prompt_format,  # type: ignore[arg-type]
            model_type=model_type,
        )
        logger.info(f"[{index}/{len(prompt_paths)}] {slug}: {len(prompt_tokens)} prompt tokens")

        graph = None
        try:
            graph = attribute(
                prompt=prompt_tokens,
                model=model,
                max_n_logits=args.max_n_logits,
                batch_size=args.batch_size,
                max_feature_nodes=args.max_feature_nodes,
                offload=args.offload,
                verbose=True,
            )
            graph.to("cpu")
            _safe_empty_cuda_cache()
            _call_create_graph_files(
                graph=graph,
                slug=slug,
                output_dir=output_dir,
                scan="base-vs-adapter",
                node_threshold=args.node_threshold,
                edge_threshold=args.edge_threshold,
            )
            raw_payload = json.loads(graph_json_path.read_text())
            tagged = tag_combined_graph(
                raw_payload,
                n_base=n_base,
                base_feature_scan=base_feature_scan,
                adapter_feature_scan=adapter_feature_scan,
                max_error_nodes=max_error_nodes,
            )
            graph_json_path.write_text(json.dumps(tagged, indent=2) + "\n")
            counts = tagged["metadata"]["comparison"]["node_counts"]
            logger.info(
                f"  Tagged graph: {counts['base_features']} base + {counts['adapter_features']} "
                f"adapter features + {counts['error_nodes']} error nodes -> {graph_json_path}"
            )
            results[prompt_path.stem] = "success"
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if "out of memory" not in str(exc).lower() and "CUDA" not in str(exc):
                raise
            if graph is None:
                raise
            raw_graph_path = _save_raw_graph(graph, output_dir, slug)
            results[prompt_path.stem] = f"raw_saved: {raw_graph_path}"
            logger.error(f"  OOM during export; saved raw graph to {raw_graph_path}")

    manifest = {
        "run_name": args.run_name,
        "adapter_checkpoint": args.adapter_checkpoint,
        "base_model": args.base_model,
        "n_base_features": n_base,
        "output_dir": str(output_dir),
        "base_feature_scan": base_feature_scan,
        "adapter_feature_scan": adapter_feature_scan,
        "max_error_nodes": max_error_nodes,
        "results": results,
    }
    (output_dir / "combined_attribution_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info(f"Wrote manifest: {output_dir / 'combined_attribution_manifest.json'}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Full-replacement (T_base + T_adapter) attribution with real error nodes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--adapter_checkpoint", required=True, help="HF repo ID or local adapter checkpoint")
    parser.add_argument("--base_model", default="google/gemma-2-2b")
    parser.add_argument("--prompt_tokenizer_model", default="google/gemma-2-2b-it", help="Tokenizer for chat prompts when the base tokenizer has no chat_template.")
    parser.add_argument("--prompts", required=True, type=Path, help="Directory of .txt prompts, or one .txt file")
    parser.add_argument("--run_name", default="combined")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--base_feature_data_path", default="mntss/gemma-scope-transcoders", help="Base feature examples: HF feature repo, or a local collected dir served as /base_features.")
    parser.add_argument("--adapter_feature_data_path", default=None, help="Adapter feature examples: HF feature repo, or a local collected dir served as /adapter_features.")
    parser.add_argument("--gemmascope_repo", default="google/gemma-scope-2b-pt-transcoders")
    parser.add_argument("--gemmascope_width", required=True, help="GemmaScope width folder, e.g. width_16k")
    parser.add_argument("--gemmascope_l0", required=True, help="GemmaScope L0 folder/target, e.g. average_l0_76")
    parser.add_argument("--gemmascope_l0_match", choices=["exact", "nearest"], default="nearest")
    parser.add_argument("--gemmascope_n_layers", type=int, default=26)
    parser.add_argument("--feature_input_hook", default="ln2.hook_normalized")
    parser.add_argument("--feature_output_hook", default="hook_mlp_out")
    parser.add_argument("--prompt_format", choices=["auto", "raw", "chat"], default="auto")
    parser.add_argument("--max_n_logits", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_feature_nodes", type=int, default=4096)
    parser.add_argument(
        "--max_error_nodes",
        type=int,
        default=DEFAULT_MAX_ERROR_NODES,
        help="Keep this many highest-influence reconstruction-error nodes (triangles). -1 keeps all.",
    )
    parser.add_argument("--node_threshold", type=float, default=0.8)
    parser.add_argument("--edge_threshold", type=float, default=0.98)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32", "bf16", "fp16", "fp32"], default="bfloat16")
    parser.add_argument("--base_backend", choices=["transformerlens", "nnsight"], default="transformerlens")
    parser.add_argument("--offload", choices=["cpu", "disk"], default=None)
    parser.epilog = textwrap.dedent(__doc__.split("Example:")[-1]) if "Example:" in (__doc__ or "") else None
    return parser


def main() -> None:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args()
    try:
        run_combined_attribution(args)
    except Exception as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
