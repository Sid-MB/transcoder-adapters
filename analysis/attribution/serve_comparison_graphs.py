"""Serve comparison overlay graphs with repo-local circuit-tracer UI overrides."""

from __future__ import annotations

import argparse
import functools
import gzip
import json
import os
import time
from pathlib import Path

from analysis.attribution.comparison_frontend import prepare_comparison_frontend
from analysis.attribution.neuronpedia_descriptions import (
    NeuronpediaDescriptions,
    cantor_unpair,
)
from analysis.attribution.run_base_adapter_comparison import (
    LOCAL_BASE_FEATURE_SCAN,
    LOCAL_FEATURE_SCAN,
)
from analysis.attribution.run_circuit_tracer_pipeline import (
    is_hf_feature_ref,
    normalize_hf_feature_ref,
)
from helpers.log import logger, setup_logging


def resolve_graph_file_dir(graph_file_dir: str | Path) -> Path:
    """Resolve ``--graph_file_dir`` to a local directory of graph JSONs.

    Accepts either a local path, or a Hugging Face **dataset** repo id (optionally with a
    ``:subdir`` suffix, e.g. ``siddharthmb/2026.TA.hybrid_ft_overlay_ms100k_graphs:overlay_compact``)
    which is pulled with ``snapshot_download`` and the (sub)directory returned. Use the subdir form
    when a single repo bundles several graph sets (``overlay``/``overlay_compact``/``base``/...).
    """
    local = Path(graph_file_dir)
    if local.is_dir():
        return local
    ref = str(graph_file_dir)
    repo_part, _, subdir = ref.partition(":")  # HF repo ids never contain ':'
    if is_hf_feature_ref(repo_part):
        from huggingface_hub import snapshot_download

        repo_id = normalize_hf_feature_ref(repo_part)
        logger.info(
            "Resolving graph dir from Hugging Face dataset repo: %s%s",
            repo_id,
            f" (subdir {subdir})" if subdir else "",
        )
        snapshot = Path(snapshot_download(repo_id, repo_type="dataset"))
        return snapshot / subdir if subdir else snapshot
    raise FileNotFoundError(f"Graph file directory does not exist: {graph_file_dir}")


def _serve_local_feature_file(handler, *, root_dir: str, prefix: str) -> bool:
    rel_path = handler.path[len(prefix) :].split("?")[0]
    local_path = os.path.join(root_dir, rel_path)
    if not os.path.exists(local_path):
        handler.send_response(404)
        handler.end_headers()
        return True
    range_header = handler.headers.get("Range", "")
    with open(local_path, "rb") as f:
        if range_header.startswith("bytes="):
            file_size = os.path.getsize(local_path)
            start, end = range_header[6:].split("-")
            start = int(start)
            end = int(end) if end else file_size - 1
            f.seek(start)
            content = f.read(end - start + 1)
            handler.send_response(206)
            handler.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        else:
            content = f.read()
            handler.send_response(200)
    handler.send_header("Content-Type", "application/octet-stream")
    handler.send_header("Content-Length", str(len(content)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(content)
    return True


# Lazily-built, disk-cached Neuronpedia description fetcher shared across requests. Guarded by a
# lock because the HTTP server is threaded and NeuronpediaDescriptions mutates its cache dict.
import threading as _threading

_neuronpedia_lock = _threading.Lock()
_neuronpedia_descriptions: NeuronpediaDescriptions | None = None


def _get_neuronpedia_descriptions() -> NeuronpediaDescriptions:
    global _neuronpedia_descriptions
    if _neuronpedia_descriptions is None:
        _neuronpedia_descriptions = NeuronpediaDescriptions()
    return _neuronpedia_descriptions


def _serve_neuronpedia_description(handler) -> bool:
    """GET /neuronpedia_description?feature=<cantor> -> {layer, feature, description, url}.

    ``feature`` is the graph node's Cantor-paired ``(layer, within-layer index)`` id (node['feature']).
    The description is fetched live from Neuronpedia's auto-interp API on first request and disk-cached
    (see :mod:`analysis.attribution.neuronpedia_descriptions`), so repeat clicks are instant and offline.
    """
    from urllib.parse import parse_qs, urlparse

    if urlparse(handler.path).path != "/neuronpedia_description":
        return False
    params = parse_qs(urlparse(handler.path).query)
    raw = (params.get("feature") or [None])[0]
    try:
        cantor = int(raw)
    except (TypeError, ValueError):
        handler.send_response(400)
        handler.end_headers()
        return True
    layer, feat = cantor_unpair(cantor)
    try:
        with _neuronpedia_lock:
            npd = _get_neuronpedia_descriptions()
            description = npd.get_by_cantor(cantor)
            npd.save()
    except Exception as exc:
        logger.warning("Neuronpedia description fetch failed for %s: %s", cantor, exc)
        description = ""
    payload = json.dumps(
        {
            "layer": layer,
            "feature": feat,
            "description": description,
            "url": f"https://www.neuronpedia.org/gemma-2-2b/{layer}-gemmascope-transcoder-16k/{feat}",
        }
    ).encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(payload)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(payload)
    return True


def _serve_graph_data_file(handler) -> bool:
    if not handler.path.startswith(("/data/", "/graph_data/")):
        return False
    if handler.path.startswith("/data/"):
        rel_path = handler.path[len("/data/") :].split("?")[0]
    else:
        rel_path = handler.path[len("/graph_data/") :].split("?")[0]
    local_path = os.path.join(handler.data_dir, rel_path)
    if not os.path.exists(local_path):
        handler.send_response(404)
        handler.end_headers()
        return True
    with open(local_path, "rb") as f:
        content = f.read()
    handler.send_response(200)
    if len(content) > 1024**2:
        content = gzip.compress(content, compresslevel=3)
        handler.send_header("Content-Encoding", "gzip")
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(content)))
    handler.end_headers()
    handler.wfile.write(content)
    return True


def _start_comparison_server(
    *,
    graph_file_dir: Path,
    frontend_dir: Path,
    adapter_features_dir: str | None,
    base_features_dir: str | None,
    port: int,
):
    from circuit_tracer.frontend.local_server import ReusableTCPServer, Server
    from http.server import SimpleHTTPRequestHandler
    import threading

    local_feature_dirs = {}
    if adapter_features_dir is not None:
        local_feature_dirs[LOCAL_FEATURE_SCAN] = adapter_features_dir
        local_feature_dirs["/features"] = adapter_features_dir
    if base_features_dir is not None:
        local_feature_dirs[LOCAL_BASE_FEATURE_SCAN] = base_features_dir

    class ComparisonGraphHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, frontend_dir, data_dir, **kwargs):
            self.data_dir = data_dir
            super().__init__(*args, directory=str(frontend_dir), **kwargs)

        def do_GET(self):
            try:
                for scan, root_dir in local_feature_dirs.items():
                    prefix = f"{scan.rstrip('/')}/"
                    if self.path.startswith(prefix):
                        _serve_local_feature_file(self, root_dir=root_dir, prefix=prefix)
                        return
                if _serve_neuronpedia_description(self):
                    return
                if _serve_graph_data_file(self):
                    return
                super().do_GET()
            except Exception as exc:
                logger.exception(f"Error handling GET request: {exc}")
                self.send_response(500)
                self.end_headers()

        def do_POST(self):
            if not self.path.startswith("/save_graph/"):
                self.send_response(404)
                return
            try:
                slug = self.path.split("?")[0].strip("/").split("/")[-1]
                content_length = int(self.headers["Content-Length"])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode("utf-8"))
                save_path = os.path.join(self.data_dir, f"{slug}.json")
                with open(save_path) as f:
                    graph = json.load(f)
                    graph["qParams"] = data["qParams"]
                with open(save_path, "w") as f:
                    json.dump(graph, f, indent=2)
                self.send_response(200)
                self.end_headers()
            except Exception as exc:
                logger.exception(f"Error saving graph: {exc}")
                self.send_response(500)
                self.end_headers()

    handler = functools.partial(
        ComparisonGraphHandler,
        frontend_dir=str(frontend_dir),
        data_dir=str(graph_file_dir),
    )
    httpd = ReusableTCPServer(("", port), handler)
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    logger.info(f"Serving at http://localhost:{port}")
    return Server(httpd, server_thread)


def serve_until_interrupted(
    *,
    graph_file_dir: Path,
    adapter_features_dir: str | Path | None = None,
    base_features_dir: str | Path | None = None,
    port: int,
) -> None:
    server = start_comparison_server(
        graph_file_dir=graph_file_dir,
        adapter_features_dir=adapter_features_dir,
        base_features_dir=base_features_dir,
        port=port,
    )
    try:
        logger.info("Press Ctrl+C to stop the server.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping comparison graph server...")
        server.stop()


def start_comparison_server(
    *,
    graph_file_dir: Path,
    adapter_features_dir: str | Path | None = None,
    base_features_dir: str | Path | None = None,
    port: int,
):
    graph_file_dir = resolve_graph_file_dir(graph_file_dir)
    resolved_adapter_features_dir = (
        str(Path(adapter_features_dir).resolve()) if adapter_features_dir is not None else None
    )
    resolved_base_features_dir = (
        str(Path(base_features_dir).resolve()) if base_features_dir is not None else None
    )
    frontend_dir = prepare_comparison_frontend(graph_file_dir)

    logger.info(f"Starting comparison graph server on port {port}")
    logger.info(f"Serving graph directory: {graph_file_dir.resolve()}")
    if resolved_adapter_features_dir is not None:
        logger.info(f"Serving adapter feature directory at {LOCAL_FEATURE_SCAN}: {resolved_adapter_features_dir}")
    if resolved_base_features_dir is not None:
        logger.info(f"Serving base feature directory at {LOCAL_BASE_FEATURE_SCAN}: {resolved_base_features_dir}")

    server = _start_comparison_server(
        graph_file_dir=graph_file_dir,
        frontend_dir=frontend_dir,
        adapter_features_dir=resolved_adapter_features_dir,
        base_features_dir=resolved_base_features_dir,
        port=port,
    )
    return server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve base-vs-adapter overlay graph JSONs with node-shape UI overrides."
    )
    parser.add_argument(
        "--graph_file_dir",
        required=True,
        help=(
            "Local directory of graph JSONs, or a Hugging Face dataset repo id to download and serve "
            "(optionally 'org/name:subdir' to serve one subdir of a multi-set repo, e.g. "
            "'siddharthmb/2026.TA.hybrid_ft_overlay_ms100k_graphs:overlay_compact')."
        ),
    )
    parser.add_argument(
        "--features_dir",
        default=None,
        help="Deprecated alias for --adapter_features_dir.",
    )
    parser.add_argument(
        "--adapter_features_dir",
        default=None,
        help="Optional adapter circuit_tracer_features directory for local feature examples.",
    )
    parser.add_argument(
        "--base_features_dir",
        default=None,
        help="Optional base/GemmaScope circuit_tracer_features directory for local feature examples.",
    )
    parser.add_argument("--port", type=int, default=8044)
    return parser


def main() -> None:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args()
    try:
        serve_until_interrupted(
            graph_file_dir=args.graph_file_dir,
            adapter_features_dir=args.adapter_features_dir or args.features_dir,
            base_features_dir=args.base_features_dir,
            port=args.port,
        )
    except Exception as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
