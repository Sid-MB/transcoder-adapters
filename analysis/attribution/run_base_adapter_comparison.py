"""Run matched base-vs-adapter circuit-tracer graphs and build an overlay.

This module keeps the comparison workflow repo-local.  It does not patch the
installed ``circuit_tracer`` package; instead it writes ordinary graph JSON for
the base and adapter runs, then writes a third overlay graph with additive node
metadata used by ``serve_comparison_graphs``.
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import textwrap
from pathlib import Path
from collections import defaultdict
from typing import Any, Literal

import yaml

from analysis.attribution.run_attribution import (
    _list_prompt_files,
    _prompt_content_hash,
    load_prompt_file,
)
from analysis.attribution.run_circuit_tracer_pipeline import (
    ensure_feature_data_conversion,
    is_hf_feature_ref,
    normalize_hf_feature_ref,
    run_pipeline,
    scan_name_for_feature_output,
)
from helpers.log import logger, setup_logging
from helpers.paths.output_path import generate_output_path


SOURCE_BASE = "base"
SOURCE_ADAPTER = "adapter"
SOURCE_SHARED = "shared"
BASE_NODE_SHAPE = "base_model"
ADAPTER_NODE_SHAPE = "adapter_model"
SHARED_NODE_SHAPE = "shared"
OVERLAY_SCHEMA_VERSION = 2
DEFAULT_COMPACT_BASE_FEATURE_NODES = 64
DEFAULT_COMPACT_BASE_ERROR_NODES = 0
LOCAL_FEATURE_SCAN = "/adapter_features"
LOCAL_BASE_FEATURE_SCAN = "/base_features"
STANDARD_GEMMA2_BASE_FEATURE_SCAN = "mntss/gemma-scope-transcoders"

_SLUG_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _slugify(value: str) -> str:
    return _SLUG_RE.sub("_", Path(value.rstrip("/")).name or value).strip("_") or "run"


def default_comparison_output_dir(*, adapter_checkpoint: str, run_name: str) -> Path:
    return generate_output_path(
        "base_adapter_comparisons",
        f"{_slugify(run_name)}_{_slugify(adapter_checkpoint)}",
        consistent=True,
    )


def default_base_feature_scan_for_gemmascope(
    *,
    base_model: str,
    gemmascope_repo: str,
    gemmascope_width: str,
) -> str | None:
    """Return the bundled HF feature scan for the standard Gemma-2 base setup."""
    if base_model.rstrip("/") != "google/gemma-2-2b":
        return None
    if gemmascope_repo.strip("/") != "google/gemma-scope-2b-pt-transcoders":
        return None
    if gemmascope_width.strip("/") != "width_16k":
        return None
    return STANDARD_GEMMA2_BASE_FEATURE_SCAN


def default_base_feature_scan_for_overlay_base_scan(base_scan: str | None) -> str | None:
    if not base_scan:
        return None
    normalized = base_scan.strip("/")
    if normalized.startswith("google/gemma-scope-2b-pt-transcoders/width_16k/"):
        return STANDARD_GEMMA2_BASE_FEATURE_SCAN
    return None


def gemmascope_scan_name(*, repo: str, width: str, l0: str, l0_match: str = "exact") -> str:
    l0_suffix = l0.strip("/")
    if l0_match != "exact":
        l0_suffix = f"{l0_suffix}_{l0_match}"
    return f"{repo.strip('/')}/{width.strip('/')}/{l0_suffix}"


def _parse_average_l0_value(l0: str) -> int:
    match = re.fullmatch(r"average_l0_(\d+)", l0.strip("/"))
    if match is None:
        raise ValueError(f"Expected GemmaScope L0 folder like average_l0_76, got {l0!r}")
    return int(match.group(1))


def _explicit_l0_values(l0: str, *, n_layers: int) -> list[str] | None:
    l0_path = Path(l0).expanduser()
    if l0_path.exists():
        payload = yaml.safe_load(l0_path.read_text())
        if isinstance(payload, list):
            values = [str(item) for item in payload]
        elif isinstance(payload, dict):
            values = []
            for layer in range(n_layers):
                if str(layer) in payload:
                    values.append(str(payload[str(layer)]))
                elif layer in payload:
                    values.append(str(payload[layer]))
                else:
                    raise ValueError(f"Missing layer {layer} in GemmaScope L0 map {l0_path}")
        else:
            raise ValueError(f"Unsupported GemmaScope L0 map in {l0_path}; expected list or dict")
    elif "," in l0:
        values = [item.strip() for item in l0.split(",") if item.strip()]
    else:
        return None

    if len(values) != n_layers:
        raise ValueError(f"Expected {n_layers} GemmaScope L0 values, got {len(values)}")
    for value in values:
        _parse_average_l0_value(value)
    return values


def _available_l0_values_for_layer(*, repo: str, width: str, layer: int) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi()
    path_in_repo = f"layer_{layer}/{width.strip('/')}"
    entries = api.list_repo_tree(
        repo_id=repo.strip("/"),
        path_in_repo=path_in_repo,
        recursive=False,
    )
    values = [
        Path(entry.path).name
        for entry in entries
        if Path(entry.path).name.startswith("average_l0_")
    ]
    if not values:
        raise ValueError(f"No average_l0_* directories found at {repo}/{path_in_repo}")
    return values


def resolve_gemmascope_l0_values(
    *,
    repo: str,
    width: str,
    l0: str,
    n_layers: int,
    l0_match: Literal["exact", "nearest"],
) -> list[str]:
    """Resolve a GemmaScope L0 spec to one concrete folder per layer."""
    explicit_values = _explicit_l0_values(l0, n_layers=n_layers)
    if explicit_values is not None:
        return explicit_values
    if l0_match == "exact":
        _parse_average_l0_value(l0)
        return [l0.strip("/")] * n_layers
    if l0_match != "nearest":
        raise ValueError(f"Unsupported GemmaScope L0 match mode: {l0_match}")

    target_l0 = _parse_average_l0_value(l0)
    resolved_values = []
    for layer in range(n_layers):
        available = _available_l0_values_for_layer(repo=repo, width=width, layer=layer)
        selected = min(available, key=lambda value: abs(_parse_average_l0_value(value) - target_l0))
        resolved_values.append(selected)
        if selected != l0:
            logger.info(
                f"Layer {layer}: requested {l0}, using nearest available GemmaScope L0 {selected}"
            )
    return resolved_values


def gemmascope_layer_refs(
    *,
    repo: str,
    width: str,
    l0: str,
    n_layers: int,
    l0_values: list[str] | None = None,
) -> list[str]:
    """Return explicit HF file refs for one GemmaScope transcoder per layer."""
    if n_layers <= 0:
        raise ValueError(f"n_layers must be positive, got {n_layers}")
    repo = repo.strip("/")
    width = width.strip("/")
    l0 = l0.strip("/")
    if not repo or not width or not l0:
        raise ValueError("repo, width, and l0 must be non-empty")
    if l0_values is None:
        l0_values = [l0] * n_layers
    if len(l0_values) != n_layers:
        raise ValueError(f"Expected {n_layers} GemmaScope L0 values, got {len(l0_values)}")
    return [
        f"hf://{repo}/layer_{layer}/{width}/{l0_values[layer].strip('/')}/params.npz"
        for layer in range(n_layers)
    ]


def build_gemmascope_transcoder_config(
    *,
    repo: str,
    width: str,
    l0: str,
    n_layers: int,
    model_name: str,
    feature_input_hook: str,
    feature_output_hook: str,
    l0_values: list[str] | None = None,
    l0_match: str = "exact",
) -> dict[str, Any]:
    """Build a circuit-tracer transcoder-set config for GemmaScope PT transcoders."""
    return {
        "model_name": model_name,
        "model_kind": "transcoder_set",
        "feature_input_hook": feature_input_hook,
        "feature_output_hook": feature_output_hook,
        "repo_id": repo.strip("/"),
        "scan_name": gemmascope_scan_name(repo=repo, width=width, l0=l0, l0_match=l0_match),
        "requested_l0": l0,
        "l0_match": l0_match,
        "l0_by_layer": l0_values or [l0.strip("/")] * n_layers,
        "transcoders": gemmascope_layer_refs(
            repo=repo,
            width=width,
            l0=l0,
            n_layers=n_layers,
            l0_values=l0_values,
        ),
    }


def write_gemmascope_transcoder_config(config: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    logger.info(f"Wrote GemmaScope transcoder config: {path}")


def _torch_dtype(dtype_name: str):
    import torch

    normalized = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
    }.get(dtype_name, dtype_name)
    try:
        return getattr(torch, normalized)
    except AttributeError as exc:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}") from exc


def _torch_device(device_name: str):
    import torch

    return torch.device(device_name)


def _safe_empty_cuda_cache() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _graph_slug(run_name: str, prompt_path: Path) -> str:
    return f"{run_name}__{prompt_path.stem}__h{_prompt_content_hash(prompt_path)}"


def _model_type_for_prompt_loader(model: Any, base_model: str) -> str | None:
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None)
    if model_type is not None:
        return str(model_type)
    if "gemma" in base_model.lower():
        return "gemma2"
    return None


def _prompt_tokenizer_for_base(
    model_tokenizer: Any,
    *,
    prompt_format: str,
    prompt_tokenizer_model: str | None,
) -> Any:
    """Return a tokenizer that can render chat prompts for base-model attribution."""
    if prompt_format == "raw" or getattr(model_tokenizer, "chat_template", None):
        return model_tokenizer
    if prompt_tokenizer_model is None:
        return model_tokenizer

    from transformers import AutoTokenizer

    logger.info(
        "Base tokenizer has no chat_template; loading prompt tokenizer from "
        f"{prompt_tokenizer_model}"
    )
    prompt_tokenizer = AutoTokenizer.from_pretrained(prompt_tokenizer_model)
    if not getattr(prompt_tokenizer, "chat_template", None):
        logger.warning(
            "Prompt tokenizer %s also has no chat_template; falling back to base tokenizer",
            prompt_tokenizer_model,
        )
        return model_tokenizer
    return prompt_tokenizer


def _call_create_graph_files(
    *,
    graph: Any,
    slug: str,
    output_dir: Path,
    scan: str,
    node_threshold: float,
    edge_threshold: float,
) -> None:
    from circuit_tracer.utils.create_graph_files import create_graph_files

    kwargs = {
        "graph_or_path": graph,
        "slug": slug,
        "output_path": output_dir,
        "node_threshold": node_threshold,
        "edge_threshold": edge_threshold,
    }
    if "scan" in inspect.signature(create_graph_files).parameters:
        kwargs["scan"] = scan
    else:
        kwargs["scan_name"] = scan
    create_graph_files(**kwargs)


def _load_gemmascope_transcoders(
    config: dict[str, Any],
    *,
    device: Any,
    dtype: Any,
):
    """Load GemmaScope 2B PT .npz transcoders without relying on repo-name heuristics."""
    from circuit_tracer.transcoder.single_layer_transcoder import load_transcoder_set
    from circuit_tracer.utils.hf_utils import resolve_transcoder_paths

    transcoder_paths = resolve_transcoder_paths(config)
    return load_transcoder_set(
        transcoder_paths,
        scan_name=config["scan_name"],
        feature_input_hook=config["feature_input_hook"],
        feature_output_hook=config["feature_output_hook"],
        special_load_fn="gemma-scope",
        device=device,
        dtype=dtype,
        lazy_encoder=False,
        lazy_decoder=False,
    )


def _save_raw_graph(graph: Any, output_dir: Path, slug: str) -> Path:
    raw_graph_path = output_dir / f"{slug}_raw.pt"
    graph.to("cpu")
    graph.to_pt(str(raw_graph_path))
    return raw_graph_path


def run_base_attribution(
    *,
    base_model: str,
    gemmascope_config_path: Path,
    prompts: Path,
    run_name: str,
    output_dir: Path,
    prompt_format: str,
    max_n_logits: int,
    batch_size: int,
    max_feature_nodes: int,
    node_threshold: float,
    edge_threshold: float,
    device: str,
    dtype: str,
    backend: Literal["nnsight", "transformerlens"],
    offload: Literal["cpu", "disk", None],
    prompt_tokenizer_model: str | None,
    finetuned_transcoder_dir: str | None = None,
    finetuned_layers: list[int] | None = None,
) -> dict[str, str]:
    """Run circuit-tracer attribution for the base model/GemmaScope side."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_paths = _list_prompt_files(prompts)
    if not prompt_paths:
        raise ValueError(f"No .txt prompt files found in {prompts}")

    existing_results = {
        prompt_path.stem: "skipped"
        for prompt_path in prompt_paths
        if (output_dir / f"{_graph_slug(run_name, prompt_path)}.json").exists()
    }
    missing_prompt_paths = [
        prompt_path
        for prompt_path in prompt_paths
        if prompt_path.stem not in existing_results
    ]
    if not missing_prompt_paths:
        logger.info("All base GemmaScope graph JSONs already exist; skipping base attribution.")
        return existing_results

    import torch
    from circuit_tracer import ReplacementModel, attribute

    config = yaml.safe_load(gemmascope_config_path.read_text())
    dtype_obj = _torch_dtype(dtype)
    device_obj = _torch_device(device)
    scan = config["scan_name"]

    logger.info(f"Loading GemmaScope transcoders for scan {scan}")
    transcoders = _load_gemmascope_transcoders(
        config,
        device=device_obj,
        dtype=dtype_obj,
    )
    if finetuned_transcoder_dir and finetuned_layers:
        from analysis.attribution.gemmascope_finetune import patch_finetuned_layers

        patch_finetuned_layers(transcoders, finetuned_transcoder_dir, finetuned_layers, device_obj, dtype_obj)
        logger.info("Applied fine-tuned transcoder weights (layers %s) from %s", finetuned_layers, finetuned_transcoder_dir)
    logger.info(f"Loading base replacement model: {base_model} ({backend})")
    model = ReplacementModel.from_pretrained_and_transcoders(
        model_name=base_model,
        transcoders=transcoders,
        backend=backend,
        device=device_obj,
        dtype=dtype_obj,
    )
    model_type = _model_type_for_prompt_loader(model, base_model)
    prompt_tokenizer = _prompt_tokenizer_for_base(
        model.tokenizer,
        prompt_format=prompt_format,
        prompt_tokenizer_model=prompt_tokenizer_model,
    )

    results: dict[str, str] = dict(existing_results)
    for index, prompt_path in enumerate(missing_prompt_paths, 1):
        slug = _graph_slug(run_name, prompt_path)
        logger.info(f"[base {index}/{len(missing_prompt_paths)}] {slug}")
        prompt_tokens, target_token, prompt_text = load_prompt_file(
            prompt_path,
            prompt_tokenizer,
            prompt_format=prompt_format,  # type: ignore[arg-type]
            model_type=model_type,
        )
        logger.info(f"  Prompt tokens: {len(prompt_tokens)}")
        logger.info(f"  Target token held out by loader: {prompt_tokenizer.decode([target_token])!r}")
        logger.info(f"  Last 60 chars: ...{prompt_text[-60:]!r}")

        graph = None
        try:
            graph = attribute(
                prompt=prompt_tokens,
                model=model,
                max_n_logits=max_n_logits,
                batch_size=batch_size,
                max_feature_nodes=max_feature_nodes,
                offload=offload,
                verbose=True,
            )
            graph.to("cpu")
            _safe_empty_cuda_cache()
            _call_create_graph_files(
                graph=graph,
                slug=slug,
                output_dir=output_dir,
                scan=scan,
                node_threshold=node_threshold,
                edge_threshold=edge_threshold,
            )
            results[prompt_path.stem] = "success"
            logger.info(f"  Wrote base graph: {output_dir / (slug + '.json')}")
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if "out of memory" not in str(exc).lower() and "CUDA" not in str(exc):
                raise
            if graph is None:
                raise
            raw_graph_path = _save_raw_graph(graph, output_dir, slug)
            results[prompt_path.stem] = f"raw_saved: {raw_graph_path}"
            logger.error(f"  OOM during base graph export; saved raw graph to {raw_graph_path}")

    return results


def _load_graph_payloads_by_prompt_tokens(graph_dir: Path) -> dict[tuple[str, ...], dict[str, Any]]:
    graphs: dict[tuple[str, ...], dict[str, Any]] = {}
    for graph_path in sorted(graph_dir.glob("*.json")):
        if graph_path.name in {"graph-metadata.json", "run_attribution_args.json"}:
            continue
        try:
            payload = json.loads(graph_path.read_text())
        except json.JSONDecodeError:
            continue
        if not {"metadata", "nodes", "links"}.issubset(payload):
            continue
        prompt_tokens = payload.get("metadata", {}).get("prompt_tokens")
        if not isinstance(prompt_tokens, list):
            continue
        graphs[tuple(str(token) for token in prompt_tokens)] = payload
    return graphs


def _is_shared_overlay_node(node: dict[str, Any]) -> bool:
    return node.get("feature_type") == "embedding"


def _is_direct_embedding_logit_link(
    link: dict[str, Any],
    *,
    nodes_by_original_id: dict[str, dict[str, Any]],
) -> bool:
    source_node = nodes_by_original_id.get(str(link["source"]))
    target_node = nodes_by_original_id.get(str(link["target"]))
    if source_node is None or target_node is None:
        return False
    endpoint_types = {source_node.get("feature_type"), target_node.get("feature_type")}
    return endpoint_types == {"embedding", "logit"}


def _prefixed_node_id(source: str, node_id: str) -> str:
    return f"{source}__{node_id}"


def _overlay_shape(source: str, *, shared: bool, feature_type: str | None = None) -> str:
    if shared:
        return SHARED_NODE_SHAPE
    if feature_type == "logit":
        return "logit"
    if feature_type and "error" in feature_type:
        return "error"
    if source == SOURCE_BASE:
        return BASE_NODE_SHAPE
    if source == SOURCE_ADAPTER:
        return ADAPTER_NODE_SHAPE
    raise ValueError(f"Unknown overlay source: {source}")


def _source_feature_index(node: dict[str, Any]) -> str:
    original_node_id = str(node.get("original_node_id") or node.get("node_id") or "")
    parts = original_node_id.split("_")
    if len(parts) >= 2 and parts[1]:
        return parts[1]
    return str(node.get("feature", ""))


def _token_label(prompt_tokens: list[Any], ctx_idx: Any) -> str:
    try:
        index = int(ctx_idx)
    except (TypeError, ValueError):
        return "unknown token"
    if index < 0 or index >= len(prompt_tokens):
        return f"token {index}"
    return str(prompt_tokens[index])


def _format_float(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{value:.3g}"


def _fallback_feature_clerp(
    node: dict[str, Any],
    *,
    source: str,
    prompt_tokens: list[Any],
) -> str:
    source_label = "Base GemmaScope" if source == SOURCE_BASE else "Adapter"
    feature_idx = _source_feature_index(node)
    token = _token_label(prompt_tokens, node.get("ctx_idx"))
    return (
        f"{source_label} feature L{node.get('layer')}/F{feature_idx} on token "
        f"{node.get('ctx_idx')} ({token}); act={_format_float(node.get('activation'))}, "
        f"influence={_format_float(node.get('influence'))}"
    )


def _overlay_node(
    node: dict[str, Any],
    *,
    source: str,
    source_graph_slug: str,
    shared: bool,
    prompt_tokens: list[Any],
) -> dict[str, Any]:
    new_node = dict(node)
    original_node_id = str(node["node_id"])
    new_node["original_node_id"] = original_node_id
    new_node["source_graph_slug"] = source_graph_slug
    new_node["source_model"] = SOURCE_SHARED if shared else source
    new_node["node_shape"] = _overlay_shape(
        source,
        shared=shared,
        feature_type=str(node.get("feature_type") or ""),
    )
    if not shared and node.get("feature_type") == "cross layer transcoder":
        new_node["source_feature_id"] = _source_feature_index(new_node)
        new_node["source_feature_label"] = (
            f"{new_node['source_model']} L{node.get('layer')}/F{new_node['source_feature_id']}"
        )
        if not new_node.get("clerp"):
            new_node["clerp"] = _fallback_feature_clerp(
                new_node,
                source=source,
                prompt_tokens=prompt_tokens,
            )
    if not shared:
        new_node["node_id"] = _prefixed_node_id(source, original_node_id)
        new_node["jsNodeId"] = _prefixed_node_id(source, str(node.get("jsNodeId", original_node_id)))
    return new_node


def _node_id_map(
    nodes: list[dict[str, Any]],
    *,
    source: str,
) -> dict[str, str]:
    mapping = {}
    for node in nodes:
        node_id = str(node["node_id"])
        mapping[node_id] = node_id if _is_shared_overlay_node(node) else _prefixed_node_id(source, node_id)
    return mapping


def build_overlay_payload(
    *,
    base_payload: dict[str, Any],
    adapter_payload: dict[str, Any],
    overlay_slug: str | None = None,
    hide_direct_embedding_logit_links: bool = True,
    base_feature_scan: str | None = None,
    adapter_feature_scan: str = LOCAL_FEATURE_SCAN,
) -> dict[str, Any]:
    """Merge matched base and adapter graph JSON payloads into one visual overlay."""
    base_metadata = base_payload["metadata"]
    adapter_metadata = adapter_payload["metadata"]
    if base_metadata.get("prompt_tokens") != adapter_metadata.get("prompt_tokens"):
        raise ValueError(
            "Cannot overlay graphs with different prompt tokens: "
            f"{base_metadata.get('slug')} vs {adapter_metadata.get('slug')}"
        )
    prompt_tokens = list(base_metadata.get("prompt_tokens") or [])

    overlay_slug = overlay_slug or f"{adapter_metadata['slug']}__overlay"
    base_slug = str(base_metadata["slug"])
    adapter_slug = str(adapter_metadata["slug"])
    base_nodes = list(base_payload["nodes"])
    adapter_nodes = list(adapter_payload["nodes"])
    base_map = _node_id_map(base_nodes, source=SOURCE_BASE)
    adapter_map = _node_id_map(adapter_nodes, source=SOURCE_ADAPTER)
    source_nodes_by_id = {
        SOURCE_BASE: {str(node["node_id"]): node for node in base_nodes},
        SOURCE_ADAPTER: {str(node["node_id"]): node for node in adapter_nodes},
    }

    nodes_by_id: dict[str, dict[str, Any]] = {}
    for source, source_slug, nodes in (
        (SOURCE_BASE, base_slug, base_nodes),
        (SOURCE_ADAPTER, adapter_slug, adapter_nodes),
    ):
        for node in nodes:
            shared = _is_shared_overlay_node(node)
            overlay_node = _overlay_node(
                node,
                source=source,
                source_graph_slug=source_slug,
                shared=shared,
                prompt_tokens=prompt_tokens,
            )
            nodes_by_id.setdefault(str(overlay_node["node_id"]), overlay_node)

    overlay_links: list[dict[str, Any]] = []
    skipped_direct_embedding_logit_links = {
        SOURCE_BASE: 0,
        SOURCE_ADAPTER: 0,
    }
    for source, source_slug, id_map, links in (
        (SOURCE_BASE, base_slug, base_map, base_payload["links"]),
        (SOURCE_ADAPTER, adapter_slug, adapter_map, adapter_payload["links"]),
    ):
        for link in links:
            if hide_direct_embedding_logit_links and _is_direct_embedding_logit_link(
                link,
                nodes_by_original_id=source_nodes_by_id[source],
            ):
                skipped_direct_embedding_logit_links[source] += 1
                continue
            source_id = str(link["source"])
            target_id = str(link["target"])
            if source_id not in id_map or target_id not in id_map:
                raise ValueError(f"Link references missing node in {source_slug}: {link}")
            overlay_link = dict(link)
            overlay_link["source"] = id_map[source_id]
            overlay_link["target"] = id_map[target_id]
            overlay_link["source_model"] = source
            overlay_link["source_graph_slug"] = source_slug
            overlay_links.append(overlay_link)

    overlay_metadata = dict(adapter_metadata)
    overlay_metadata.update(
        {
            "slug": overlay_slug,
            "scan": "base-vs-adapter",
            "transcoder_list": [],
            "schema_version": OVERLAY_SCHEMA_VERSION,
            "comparison": {
                "base_slug": base_slug,
                "adapter_slug": adapter_slug,
                "base_scan": base_metadata.get("scan"),
                "adapter_scan": adapter_metadata.get("scan"),
                "base_feature_scan": base_feature_scan,
                "adapter_feature_scan": adapter_feature_scan,
                "node_shapes": {
                    SOURCE_BASE: BASE_NODE_SHAPE,
                    SOURCE_ADAPTER: ADAPTER_NODE_SHAPE,
                    SOURCE_SHARED: SHARED_NODE_SHAPE,
                },
                "link_filter": {
                    "hide_direct_embedding_logit_links": hide_direct_embedding_logit_links,
                    "skipped_direct_embedding_logit_links": skipped_direct_embedding_logit_links,
                },
            },
        }
    )
    qparams = dict(adapter_payload.get("qParams") or {})
    qparams["clickedId"] = ""
    return {
        "metadata": overlay_metadata,
        "qParams": qparams,
        "nodes": list(nodes_by_id.values()),
        "links": overlay_links,
    }


def _is_compactable_feature_node(node: dict[str, Any], *, source: str) -> bool:
    return (
        node.get("source_model") == source
        and node.get("feature_type") == "cross layer transcoder"
    )


def _is_compactable_error_node(node: dict[str, Any], *, source: str) -> bool:
    return node.get("source_model") == source and "error" in str(node.get("feature_type") or "")


def normalize_overlay_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Add current display metadata to old or freshly-generated overlay payloads."""
    normalized = {
        **payload,
        "metadata": dict(payload["metadata"]),
        "qParams": dict(payload.get("qParams") or {}),
        "nodes": [dict(node) for node in payload["nodes"]],
        "links": [dict(link) for link in payload["links"]],
    }
    comparison = dict(normalized["metadata"].get("comparison") or {})
    if comparison and "adapter_feature_scan" not in comparison:
        comparison["adapter_feature_scan"] = LOCAL_FEATURE_SCAN
        normalized["metadata"]["comparison"] = comparison
    prompt_tokens = list(normalized["metadata"].get("prompt_tokens") or [])
    for node in normalized["nodes"]:
        feature_type = str(node.get("feature_type") or "")
        shared = node.get("source_model") == SOURCE_SHARED or feature_type == "embedding"
        source = str(node.get("source_model") or SOURCE_SHARED)
        if source == SOURCE_SHARED and not shared:
            source = SOURCE_ADAPTER
        if source in {SOURCE_BASE, SOURCE_ADAPTER} or shared:
            node["node_shape"] = _overlay_shape(
                SOURCE_ADAPTER if shared else source,
                shared=shared,
                feature_type=feature_type,
            )
        if source in {SOURCE_BASE, SOURCE_ADAPTER} and feature_type == "cross layer transcoder":
            node["source_feature_id"] = _source_feature_index(node)
            node["source_feature_label"] = (
                f"{source} L{node.get('layer')}/F{node['source_feature_id']}"
            )
            if not node.get("clerp"):
                node["clerp"] = _fallback_feature_clerp(
                    node,
                    source=source,
                    prompt_tokens=prompt_tokens,
                )
    return normalized


def _rank_nodes_by_incident_weight(payload: dict[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for link in payload.get("links", []):
        try:
            weight = abs(float(link.get("weight", 0.0)))
        except (TypeError, ValueError):
            weight = 0.0
        scores[str(link.get("source"))] += weight
        scores[str(link.get("target"))] += weight
    return scores


def compact_overlay_payload(
    payload: dict[str, Any],
    *,
    max_base_feature_nodes: int | None = DEFAULT_COMPACT_BASE_FEATURE_NODES,
    max_base_error_nodes: int | None = DEFAULT_COMPACT_BASE_ERROR_NODES,
    max_adapter_feature_nodes: int | None = None,
    base_feature_scan: str | None = None,
    adapter_feature_scan: str | None = None,
    slug_suffix: str | None = None,
) -> dict[str, Any]:
    """Return a display-focused overlay payload with optional source-node caps."""
    payload = normalize_overlay_payload(payload)
    if max_base_feature_nodes is not None and max_base_feature_nodes < 0:
        raise ValueError("max_base_feature_nodes must be non-negative or None")
    if max_base_error_nodes is not None and max_base_error_nodes < 0:
        raise ValueError("max_base_error_nodes must be non-negative or None")
    if max_adapter_feature_nodes is not None and max_adapter_feature_nodes < 0:
        raise ValueError("max_adapter_feature_nodes must be non-negative or None")

    scores = _rank_nodes_by_incident_weight(payload)
    kept_ids = {str(node["node_id"]) for node in payload["nodes"]}
    removed_counts = {
        SOURCE_BASE: 0,
        SOURCE_ADAPTER: 0,
    }
    removed_error_counts = {
        SOURCE_BASE: 0,
        SOURCE_ADAPTER: 0,
    }

    def apply_cap(source: str, cap: int | None, *, node_filter) -> int:
        if cap is None:
            return 0
        source_nodes = [
            node
            for node in payload["nodes"]
            if node_filter(node, source=source)
        ]
        ranked_nodes = sorted(
            source_nodes,
            key=lambda node: (
                -scores.get(str(node["node_id"]), 0.0),
                -abs(float(node.get("influence") or 0.0)),
                str(node["node_id"]),
            ),
        )
        selected = {str(node["node_id"]) for node in ranked_nodes[:cap]}
        for node in ranked_nodes[cap:]:
            kept_ids.discard(str(node["node_id"]))
        return max(0, len(ranked_nodes) - len(selected))

    removed_counts[SOURCE_BASE] += apply_cap(
        SOURCE_BASE,
        max_base_feature_nodes,
        node_filter=_is_compactable_feature_node,
    )
    removed_error_counts[SOURCE_BASE] += apply_cap(
        SOURCE_BASE,
        max_base_error_nodes,
        node_filter=_is_compactable_error_node,
    )
    removed_counts[SOURCE_ADAPTER] += apply_cap(
        SOURCE_ADAPTER,
        max_adapter_feature_nodes,
        node_filter=_is_compactable_feature_node,
    )

    compact_payload = {
        **payload,
        "metadata": dict(payload["metadata"]),
        "qParams": dict(payload.get("qParams") or {}),
        "nodes": [node for node in payload["nodes"] if str(node["node_id"]) in kept_ids],
        "links": [
            link
            for link in payload["links"]
            if str(link.get("source")) in kept_ids and str(link.get("target")) in kept_ids
        ],
    }
    original_slug = str(payload["metadata"]["slug"])
    if slug_suffix is None:
        suffix_parts = []
        if max_base_feature_nodes is not None:
            suffix_parts.append(f"base{max_base_feature_nodes}")
        if max_base_error_nodes is not None:
            suffix_parts.append(f"error{max_base_error_nodes}")
        if max_adapter_feature_nodes is not None:
            suffix_parts.append(f"adapter{max_adapter_feature_nodes}")
        slug_suffix = "compact_" + "_".join(suffix_parts or ["all"])
    compact_payload["metadata"]["slug"] = f"{original_slug}__{slug_suffix}"
    compact_payload["metadata"]["comparison"] = dict(
        compact_payload["metadata"].get("comparison") or {}
    )
    if base_feature_scan is not None:
        compact_payload["metadata"]["comparison"]["base_feature_scan"] = base_feature_scan
    if adapter_feature_scan is not None:
        compact_payload["metadata"]["comparison"]["adapter_feature_scan"] = adapter_feature_scan
    compact_payload["metadata"]["comparison"]["compact_view"] = {
        "source_slug": original_slug,
        "rank": "sum_abs_incident_link_weight",
        "max_base_feature_nodes": max_base_feature_nodes,
        "max_base_error_nodes": max_base_error_nodes,
        "max_adapter_feature_nodes": max_adapter_feature_nodes,
        "removed_feature_nodes": removed_counts,
        "removed_error_nodes": removed_error_counts,
        "original_node_count": len(payload["nodes"]),
        "compact_node_count": len(compact_payload["nodes"]),
        "original_link_count": len(payload["links"]),
        "compact_link_count": len(compact_payload["links"]),
    }
    compact_payload["qParams"]["clickedId"] = ""
    return compact_payload


def write_compact_overlay_graphs(
    *,
    overlay_graph_dir: Path,
    compact_overlay_graph_dir: Path,
    max_base_feature_nodes: int | None = DEFAULT_COMPACT_BASE_FEATURE_NODES,
    max_base_error_nodes: int | None = DEFAULT_COMPACT_BASE_ERROR_NODES,
    max_adapter_feature_nodes: int | None = None,
    base_feature_scan: str | None = None,
    adapter_feature_scan: str | None = None,
) -> list[Path]:
    """Write compact display variants for already-written overlay graph JSONs."""
    compact_overlay_graph_dir.mkdir(parents=True, exist_ok=True)
    written_paths = []
    for graph_path in sorted(overlay_graph_dir.glob("*.json")):
        if graph_path.name in {"graph-metadata.json", "run_attribution_args.json"}:
            continue
        payload = json.loads(graph_path.read_text())
        if not {"metadata", "nodes", "links"}.issubset(payload):
            continue
        compact_payload = compact_overlay_payload(
            payload,
            max_base_feature_nodes=max_base_feature_nodes,
            max_base_error_nodes=max_base_error_nodes,
            max_adapter_feature_nodes=max_adapter_feature_nodes,
            base_feature_scan=base_feature_scan,
            adapter_feature_scan=adapter_feature_scan,
        )
        compact_path = compact_overlay_graph_dir / f"{compact_payload['metadata']['slug']}.json"
        compact_path.write_text(json.dumps(compact_payload, indent=2) + "\n")
        _write_graph_metadata(compact_payload["metadata"], compact_overlay_graph_dir)
        written_paths.append(compact_path)
        logger.info(f"Wrote compact overlay graph: {compact_path}")
    return written_paths


def _write_graph_metadata(graph_metadata: dict[str, Any], output_dir: Path) -> None:
    metadata_path = output_dir / "graph-metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    else:
        metadata = {"graphs": []}
    metadata["graphs"] = [
        graph
        for graph in metadata.get("graphs", [])
        if graph.get("slug") != graph_metadata["slug"]
    ]
    metadata["graphs"].append(graph_metadata)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


def write_overlay_graphs(
    *,
    base_graph_dir: Path,
    adapter_graph_dir: Path,
    overlay_graph_dir: Path,
    hide_direct_embedding_logit_links: bool = True,
    base_feature_scan: str | None = None,
) -> list[Path]:
    """Write overlay graph JSONs for all matched base/adapter prompt-token sets."""
    base_graphs = _load_graph_payloads_by_prompt_tokens(base_graph_dir)
    adapter_graphs = _load_graph_payloads_by_prompt_tokens(adapter_graph_dir)
    shared_keys = sorted(set(base_graphs) & set(adapter_graphs))
    if not shared_keys:
        raise ValueError(
            f"No matching base/adapter graph JSONs found in {base_graph_dir} and {adapter_graph_dir}"
        )

    overlay_graph_dir.mkdir(parents=True, exist_ok=True)
    written_paths = []
    for prompt_key in shared_keys:
        overlay_payload = build_overlay_payload(
            base_payload=base_graphs[prompt_key],
            adapter_payload=adapter_graphs[prompt_key],
            hide_direct_embedding_logit_links=hide_direct_embedding_logit_links,
            base_feature_scan=base_feature_scan,
        )
        overlay_path = overlay_graph_dir / f"{overlay_payload['metadata']['slug']}.json"
        overlay_path.write_text(json.dumps(overlay_payload, indent=2) + "\n")
        _write_graph_metadata(overlay_payload["metadata"], overlay_graph_dir)
        written_paths.append(overlay_path)
        logger.info(f"Wrote overlay graph: {overlay_path}")
    return written_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run base GemmaScope and adapter attribution for matched prompts, then write overlay graph JSONs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--adapter_checkpoint", required=True, help="HF repo ID or local adapter checkpoint")
    parser.add_argument("--base_model", default="google/gemma-2-2b")
    parser.add_argument(
        "--prompt_tokenizer_model",
        default="google/gemma-2-2b-it",
        help=(
            "Tokenizer used only to render chat prompts for base attribution when "
            "the base tokenizer has no chat_template."
        ),
    )
    parser.add_argument("--prompts", required=True, type=Path, help="Directory of .txt prompts, or one .txt file")
    parser.add_argument("--run_name", default="base_adapter_overlay")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--feature_data_path", default=None, help="Optional adapter feature examples")
    parser.add_argument(
        "--base_feature_data_path",
        default=None,
        help=(
            "Optional base/GemmaScope feature examples. Use a Hugging Face feature repo "
            "for pre-collected features, or a local collected feature-data/packed "
            "feature directory to serve it as /base_features. Defaults to "
            f"{STANDARD_GEMMA2_BASE_FEATURE_SCAN} for the standard google/gemma-2-2b "
            "GemmaScope width_16k base setup."
        ),
    )
    parser.add_argument("--adapter_feature_output_dir", type=Path, default=None)
    parser.add_argument("--adapter_transcoder_output_dir", type=Path, default=None)
    parser.add_argument("--gemmascope_repo", default="google/gemma-scope-2b-pt-transcoders")
    parser.add_argument("--gemmascope_width", required=True, help="GemmaScope width folder, e.g. width_16k")
    parser.add_argument(
        "--gemmascope_l0",
        required=True,
        help=(
            "GemmaScope L0 folder, target L0, comma-separated per-layer folders, "
            "or a YAML/JSON list/dict path. Example: average_l0_76"
        ),
    )
    parser.add_argument(
        "--gemmascope_l0_match",
        choices=["exact", "nearest"],
        default="nearest",
        help=(
            "How to interpret a single --gemmascope_l0 value. "
            "'nearest' selects the closest available average_l0_* folder per layer."
        ),
    )
    parser.add_argument("--gemmascope_n_layers", type=int, default=26)
    parser.add_argument("--finetuned_transcoder_dir", type=str, default=None, help="Optional dir with finetuned_layer_*.safetensors: apply those layers' fine-tuned weights to the base GemmaScope transcoders before attribution/serving (from analysis/features/finetune_transcoder_shift.py). Use with --finetuned_layers.")
    parser.add_argument("--finetuned_layers", nargs="+", type=int, default=None, help="Layers to patch from --finetuned_transcoder_dir (e.g. 0 24 25).")
    parser.add_argument("--gemmascope_config_path", type=Path, default=None)
    parser.add_argument("--feature_input_hook", default="ln2.hook_normalized")
    parser.add_argument("--feature_output_hook", default="hook_mlp_out")
    parser.add_argument("--activation", default="relu", choices=["relu"])
    parser.add_argument("--prompt_format", choices=["auto", "raw", "chat"], default="auto")
    parser.add_argument("--max_n_logits", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_feature_nodes", type=int, default=10000)
    parser.add_argument(
        "--overlay_max_base_feature_nodes",
        type=int,
        default=DEFAULT_COMPACT_BASE_FEATURE_NODES,
        help=(
            "Write an additional compact overlay view keeping this many base "
            "cross-layer feature nodes per graph. Use -1 to skip compact views."
        ),
    )
    parser.add_argument(
        "--overlay_max_base_error_nodes",
        type=int,
        default=DEFAULT_COMPACT_BASE_ERROR_NODES,
        help=(
            "Write compact overlay views keeping this many base reconstruction-error "
            "nodes per graph. Use -1 for no base error cap."
        ),
    )
    parser.add_argument(
        "--overlay_max_adapter_feature_nodes",
        type=int,
        default=None,
        help="Optional adapter feature-node cap for the compact overlay view.",
    )
    parser.add_argument("--node_threshold", type=float, default=0.8)
    parser.add_argument("--edge_threshold", type=float, default=0.98)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32", "bf16", "fp16", "fp32"], default="bfloat16")
    parser.add_argument("--base_backend", choices=["nnsight", "transformerlens"], default="nnsight")
    parser.add_argument("--base_offload", choices=["cpu", "disk"], default=None)
    parser.add_argument(
        "--show_direct_embedding_logit_links",
        action="store_true",
        help=(
            "Keep raw direct embedding-to-logit skip links in the overlay. "
            "By default the overlay hides them so source paths are easier to read through feature nodes."
        ),
    )
    parser.add_argument("--serve", action="store_true", help="Serve overlay graph after writing outputs")
    parser.add_argument("--port", type=int, default=8044)
    parser.epilog = textwrap.dedent(
        """
        Example:
          uv run --extra viz python -m analysis.attribution.run_base_adapter_comparison \\
            --adapter_checkpoint siddharthmb/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860 \\
            --base_model google/gemma-2-2b \\
            --prompts analysis/attribution/prompts/interesting_small/capital_paris.txt \\
            --prompt_format chat \\
            --gemmascope_width width_16k \\
            --gemmascope_l0 average_l0_76 \\
            --max_feature_nodes 64 --batch_size 4 --max_n_logits 5
        """
    )
    return parser


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir or default_comparison_output_dir(
        adapter_checkpoint=args.adapter_checkpoint,
        run_name=args.run_name,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    base_graph_dir = output_dir / "base"
    adapter_graph_dir = output_dir / "adapter"
    overlay_graph_dir = output_dir / "overlay"
    compact_overlay_graph_dir = output_dir / "overlay_compact"
    gemmascope_config_path = args.gemmascope_config_path or output_dir / "gemmascope_config.yaml"

    l0_values = resolve_gemmascope_l0_values(
        repo=args.gemmascope_repo,
        width=args.gemmascope_width,
        l0=args.gemmascope_l0,
        n_layers=args.gemmascope_n_layers,
        l0_match=args.gemmascope_l0_match,
    )
    gemmascope_config = build_gemmascope_transcoder_config(
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
    write_gemmascope_transcoder_config(gemmascope_config, gemmascope_config_path)

    feature_output = ensure_feature_data_conversion(
        args.feature_data_path,
        args.adapter_feature_output_dir,
        n_layers=None,
        n_features=None,
    )
    adapter_feature_output_dir = feature_output if isinstance(feature_output, Path) else None
    adapter_scan = scan_name_for_feature_output(feature_output, args.run_name)
    adapter_features_dir = str(feature_output) if isinstance(feature_output, Path) else None
    if adapter_features_dir is not None:
        adapter_scan = LOCAL_FEATURE_SCAN
    # The overlay frontend loads adapter feature examples from `adapter_feature_scan`: the local
    # `/adapter_features` alias when features are on disk (served via --adapter_features_dir), or the
    # HF repo id when they live on the Hub (fetched remotely, like base_feature_scan). Mirror base so
    # HF adapter features don't 404 against an unmounted local alias.
    adapter_feature_scan = LOCAL_FEATURE_SCAN if adapter_features_dir is not None else adapter_scan
    base_features_dir = None
    base_feature_scan = None
    if args.base_feature_data_path:
        if is_hf_feature_ref(args.base_feature_data_path):
            base_feature_scan = normalize_hf_feature_ref(args.base_feature_data_path)
        else:
            base_feature_path = Path(args.base_feature_data_path).expanduser()
            if (base_feature_path / "circuit_tracer_features" / "index.json.gz").exists():
                base_feature_path = base_feature_path / "circuit_tracer_features"
            base_features_dir = str(base_feature_path)
            base_feature_scan = LOCAL_BASE_FEATURE_SCAN
    else:
        base_feature_scan = default_base_feature_scan_for_gemmascope(
            base_model=args.base_model,
            gemmascope_repo=args.gemmascope_repo,
            gemmascope_width=args.gemmascope_width,
        )
        if base_feature_scan is not None:
            logger.info(f"Using standard Gemma-2 base feature scan: {base_feature_scan}")

    adapter_args = argparse.Namespace(
        transcoder_model_path=args.adapter_checkpoint,
        base_model=args.base_model,
        prompts=args.prompts,
        feature_data_path=args.feature_data_path,
        run_name=args.run_name,
        transcoder_output_dir=args.adapter_transcoder_output_dir,
        feature_output_dir=adapter_feature_output_dir,
        graph_output_dir=adapter_graph_dir,
        prompt_format=args.prompt_format,
        max_n_logits=args.max_n_logits,
        batch_size=args.batch_size,
        max_feature_nodes=args.max_feature_nodes,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
        device=args.device,
        device_map=None,
        auto_shard_gpus=False,
        n_layers=None,
        n_features=None,
        feature_input_hook=args.feature_input_hook,
        feature_output_hook=args.feature_output_hook,
        activation=args.activation,
        port=args.port,
        serve=False,
    )

    logger.info("Running adapter attribution")
    adapter_results = run_pipeline(adapter_args)
    logger.info("Running base GemmaScope attribution")
    base_results = run_base_attribution(
        base_model=args.base_model,
        gemmascope_config_path=gemmascope_config_path,
        prompts=args.prompts,
        run_name=args.run_name,
        output_dir=base_graph_dir,
        prompt_format=args.prompt_format,
        max_n_logits=args.max_n_logits,
        batch_size=args.batch_size,
        max_feature_nodes=args.max_feature_nodes,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
        device=args.device,
        dtype=args.dtype,
        backend=args.base_backend,
        offload=args.base_offload,
        prompt_tokenizer_model=args.prompt_tokenizer_model,
        finetuned_transcoder_dir=args.finetuned_transcoder_dir,
        finetuned_layers=args.finetuned_layers,
    )
    overlay_paths = write_overlay_graphs(
        base_graph_dir=base_graph_dir,
        adapter_graph_dir=adapter_graph_dir,
        overlay_graph_dir=overlay_graph_dir,
        hide_direct_embedding_logit_links=not args.show_direct_embedding_logit_links,
        base_feature_scan=base_feature_scan,
        adapter_feature_scan=adapter_feature_scan,
    )
    compact_overlay_paths: list[Path] = []
    compact_base_cap = (
        None if args.overlay_max_base_feature_nodes < 0 else args.overlay_max_base_feature_nodes
    )
    compact_base_error_cap = (
        None if args.overlay_max_base_error_nodes < 0 else args.overlay_max_base_error_nodes
    )
    if (
        args.overlay_max_base_feature_nodes >= 0
        or args.overlay_max_base_error_nodes >= 0
        or args.overlay_max_adapter_feature_nodes is not None
    ):
        compact_overlay_paths = write_compact_overlay_graphs(
            overlay_graph_dir=overlay_graph_dir,
            compact_overlay_graph_dir=compact_overlay_graph_dir,
            max_base_feature_nodes=compact_base_cap,
            max_base_error_nodes=compact_base_error_cap,
            max_adapter_feature_nodes=args.overlay_max_adapter_feature_nodes,
        )

    manifest = {
        "adapter_checkpoint": args.adapter_checkpoint,
        "base_model": args.base_model,
        "prompt_tokenizer_model": args.prompt_tokenizer_model,
        "prompts": str(args.prompts),
        "run_name": args.run_name,
        "prompt_format": args.prompt_format,
        "output_dir": str(output_dir),
        "base_graph_dir": str(base_graph_dir),
        "adapter_graph_dir": str(adapter_graph_dir),
        "overlay_graph_dir": str(overlay_graph_dir),
        "overlay_graph_paths": [str(path) for path in overlay_paths],
        "compact_overlay_graph_dir": str(compact_overlay_graph_dir) if compact_overlay_paths else None,
        "compact_overlay_graph_paths": [str(path) for path in compact_overlay_paths],
        "gemmascope_config_path": str(gemmascope_config_path),
        "gemmascope": {
            "repo": args.gemmascope_repo,
            "width": args.gemmascope_width,
            "requested_l0": args.gemmascope_l0,
            "l0_match": args.gemmascope_l0_match,
            "l0_by_layer": l0_values,
            "n_layers": args.gemmascope_n_layers,
            "scan": gemmascope_config["scan_name"],
        },
        "adapter_features": {
            "feature_data_path": args.feature_data_path,
            "feature_output_dir": str(feature_output) if isinstance(feature_output, Path) else None,
            "scan": adapter_scan,
            "features_dir": adapter_features_dir,
        },
        "base_features": {
            "feature_data_path": args.base_feature_data_path,
            "scan": base_feature_scan,
            "features_dir": base_features_dir,
        },
        "overlay_options": {
            "hide_direct_embedding_logit_links": not args.show_direct_embedding_logit_links,
            "compact_max_base_feature_nodes": compact_base_cap if compact_overlay_paths else None,
            "compact_max_base_error_nodes": compact_base_error_cap if compact_overlay_paths else None,
            "compact_max_adapter_feature_nodes": (
                args.overlay_max_adapter_feature_nodes if compact_overlay_paths else None
            ),
        },
        "results": {
            "base": base_results,
            "adapter": adapter_results,
        },
        "later_todo": [
            "Add mixed feature-example routing for local adapter examples plus base GemmaScope examples.",
        ],
    }
    manifest_path = output_dir / "comparison-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info(f"Wrote comparison manifest: {manifest_path}")

    if args.serve:
        from analysis.attribution.serve_comparison_graphs import serve_until_interrupted

        serve_until_interrupted(
            graph_file_dir=compact_overlay_graph_dir if compact_overlay_paths else overlay_graph_dir,
            adapter_features_dir=adapter_features_dir,
            base_features_dir=base_features_dir,
            port=args.port,
        )

    return manifest


def main() -> None:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args()
    try:
        run_comparison(args)
    except Exception as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
