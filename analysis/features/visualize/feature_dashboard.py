"""
Interactive dashboard for collect_feature_activations.py outputs.

Serves a browser UI that loads ``feature_metadata.json`` and per-feature JSON files
from a collection directory (``features/{cantor_id}.json``). Supports multi-domain
runs: per-domain densities/fractions and per-domain example quantiles from the
collector are shown side by side.

Usage:
    python -m analysis.features.visualize.feature_dashboard --data_dir /path/to/output_dir

Then open the printed URL (default http://127.0.0.1:8765/). Stop with Ctrl+C.

This does not require ``pack_features.py``; it reads the same JSON the packer uses.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

from helpers.log import logger, setup_logging

_STATIC_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _STATIC_DIR.parents[2]
_DEFAULT_PROMPT_OUTPUT_DIR = _REPO_ROOT / "analysis" / "attribution" / "prompts"


def _safe_path_component(
    value: object,
    fallback: str,
    max_length: int = 96,
    lower: bool = False,
) -> str:
    text = str(value or "")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    if lower:
        text = text.lower()
    if not text:
        text = fallback
    return text[:max_length].strip("._-") or fallback


def _next_available_path(directory: Path, stem: str, suffix: str = ".txt") -> Path:
    candidate = directory / f"{stem}{suffix}"
    if not candidate.exists():
        return candidate
    for i in range(2, 10000):
        candidate = directory / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find an available filename for {stem}{suffix}")


def _save_prompt_example(
    prompt_output_dir: Path,
    data_dir: Path,
    payload: dict,
) -> dict:
    transcript = payload.get("transcript")
    if not isinstance(transcript, str) or not transcript:
        raise ValueError("transcript must be a non-empty string")

    layer = _safe_path_component(payload.get("layer"), fallback="L")
    feature = _safe_path_component(payload.get("feature"), fallback="F")
    cantor_id = _safe_path_component(payload.get("cantor_id"), fallback="cantor")
    quantile = _safe_path_component(
        payload.get("quantile_name"),
        fallback="example",
        lower=True,
    )
    example_index = payload.get("example_index", 0)
    try:
        example_number = int(example_index) + 1
    except (TypeError, ValueError):
        example_number = 1

    folder_name = _safe_path_component(data_dir.name, fallback="feature_run")
    prompt_dir = prompt_output_dir / folder_name
    prompt_dir.mkdir(parents=True, exist_ok=True)

    stem = f"L{layer}_F{feature}_{cantor_id}_{quantile}_{example_number:02d}"
    path = _next_available_path(prompt_dir, stem)
    path.write_text(transcript)

    try:
        display_path = str(path.relative_to(_REPO_ROOT))
    except ValueError:
        display_path = str(path)

    return {
        "ok": True,
        "path": str(path),
        "display_path": display_path,
        "prompt_format": "raw",
    }


def _load_jsonl_row(path: Path, row_idx: int) -> dict:
    with path.open() as f:
        for i, line in enumerate(f):
            if i == row_idx:
                return json.loads(line)
    raise IndexError(f"Row {row_idx} not found in {path}")


def _download_hf_jsonl_row(source_path: str, row_idx: int) -> dict:
    from huggingface_hub import hf_hub_download

    hf_path = source_path[len("hf://") :]
    parts = hf_path.split("/", 2)
    if len(parts) != 3:
        raise ValueError(f"Expected hf://org/repo/path.jsonl, got {source_path!r}")
    repo_id = f"{parts[0]}/{parts[1]}"
    filename = parts[2]
    local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    return _load_jsonl_row(Path(local_path), row_idx)


def _load_hf_dataset_row(source_path: str, row_idx: int) -> tuple[dict, str]:
    from datasets import load_dataset

    dataset = load_dataset(source_path, trust_remote_code=True)
    for split_name in ("val", "validation", "test"):
        if split_name in dataset:
            split = dataset[split_name]
            return dict(split[row_idx]), split_name
    available = list(dataset.keys())
    raise ValueError(
        f"No validation split found in {source_path!r}; available splits: {available}"
    )


def _normalize_conversation_value(value: object) -> list[dict]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError("Conversation column is not a list")
    conversation = []
    for msg in value:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role") or msg.get("from") or msg.get("speaker") or "unknown"
        content = msg.get("content")
        if content is None:
            content = msg.get("value", "")
        conversation.append({"role": str(role), "content": str(content)})
    return conversation


def _format_source_row_transcript(row: dict) -> tuple[str, str]:
    for conv_col in ("conversation", "conversations"):
        if conv_col not in row:
            continue
        conversation = _normalize_conversation_value(row[conv_col])
        lines = []
        for msg in conversation:
            lines.append(f"{msg['role']}:\n{msg['content']}")
        return "\n\n".join(lines), conv_col

    for text_col in ("text", "content", "prompt"):
        value = row.get(text_col)
        if value is not None:
            return str(value), text_col

    return json.dumps(row, indent=2, sort_keys=True), "json"


def _load_source_transcript(source_metadata: dict) -> dict:
    source_path = source_metadata.get("source_path")
    if not isinstance(source_path, str) or not source_path:
        raise ValueError("source_metadata.source_path is required")
    try:
        row_idx = int(source_metadata["dataset_row_idx"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("source_metadata.dataset_row_idx is required") from exc

    split_name = None
    if source_path.startswith("hf://"):
        row = _download_hf_jsonl_row(source_path, row_idx)
        source_kind = "hf_jsonl"
    elif source_path.endswith(".jsonl") or Path(source_path).is_file():
        row = _load_jsonl_row(Path(source_path), row_idx)
        source_kind = "jsonl"
    else:
        row, split_name = _load_hf_dataset_row(source_path, row_idx)
        source_kind = "hf_dataset"

    transcript, transcript_field = _format_source_row_transcript(row)
    ids = {
        key: row[key]
        for key in ("conversation_id", "id")
        if key in row and row[key] is not None
    }
    return {
        "ok": True,
        "source_kind": source_kind,
        "source_path": source_path,
        "split": split_name,
        "dataset_row_idx": row_idx,
        "transcript_field": transcript_field,
        "transcript": transcript,
        "ids": ids,
    }


def _load_annotations(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected annotations object in {path}")
    return data


def _save_annotations(path: Path, annotations: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(annotations, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp_path.replace(path)


def _load_feature_histogram_payload(
    *,
    data_dir: Path,
    histograms_file: str,
    feature_meta: dict,
    tokens_per_domain: dict[str, int],
    layer_cache: dict[int, dict],
) -> dict:
    hist_path = data_dir / histograms_file
    if not hist_path.is_file():
        raise FileNotFoundError(f"Missing activation histogram sidecar: {hist_path}")

    layer_idx = int(feature_meta["layer"])
    feature_idx = int(feature_meta["feature"])
    if layer_idx not in layer_cache:
        with np.load(hist_path, allow_pickle=False) as data:
            layer_cache[layer_idx] = {
                "bin_lower_bounds": data["bin_lower_bounds"].astype(float),
                "domain_names": data["domain_names"].astype(str).tolist(),
                "feature_hist_total": data[f"feature_hist_total_layer_{layer_idx}"],
                "feature_hist_by_domain": data[f"feature_hist_by_domain_layer_{layer_idx}"],
            }
    layer_data = layer_cache[layer_idx]
    bins = layer_data["bin_lower_bounds"].tolist()
    domain_names = layer_data["domain_names"]
    total_arr = layer_data["feature_hist_total"][feature_idx]
    by_domain_arr = layer_data["feature_hist_by_domain"][:, feature_idx, :]
    by_domain_counts = {
        domain: by_domain_arr[i].astype(int).tolist()
        for i, domain in enumerate(domain_names)
    }
    by_domain_token_density = {}
    by_domain_activation_fraction = {}
    for i, domain in enumerate(domain_names):
        counts = by_domain_arr[i].astype(float)
        domain_tokens = int(tokens_per_domain.get(domain) or 0)
        domain_total = float(counts.sum())
        by_domain_token_density[domain] = (
            (counts / domain_tokens).tolist()
            if domain_tokens > 0
            else [0.0] * len(counts)
        )
        by_domain_activation_fraction[domain] = (
            (counts / domain_total).tolist()
            if domain_total > 0
            else [0.0] * len(counts)
        )
    return {
        "bin_lower_bounds": bins,
        "domain_names": domain_names,
        "total_counts": total_arr.astype(int).tolist(),
        "by_domain_counts": by_domain_counts,
        "by_domain_token_density": by_domain_token_density,
        "by_domain_activation_fraction": by_domain_activation_fraction,
    }


def _build_top_logit_index(data_dir: Path) -> list[dict]:
    features_dir = data_dir / "features"
    if not features_dir.is_dir():
        return []
    index = []
    for feature_path in sorted(features_dir.glob("*.json")):
        try:
            feature_data = json.loads(feature_path.read_text())
        except Exception as exc:
            logger.warning("Could not load feature JSON for logit search: %s", exc)
            continue
        top_logits = feature_data.get("top_logits")
        if not isinstance(top_logits, list):
            continue
        try:
            cantor_id = int(feature_path.stem)
        except ValueError:
            continue
        index.append({
            "cantor_id": cantor_id,
            "layer": feature_data.get("layer"),
            "feature": feature_data.get("feature"),
            "top_logits": [
                str(token)
                for token in top_logits
                if token is not None
            ],
        })
    return index


def _search_top_logits(index: list[dict], query: str) -> dict:
    needle = query.casefold()
    if not needle:
        return {"query": query, "count": 0, "matches": []}
    matches = []
    for entry in index:
        matched_logits = [
            token
            for token in entry["top_logits"]
            if needle in token.casefold()
        ]
        if matched_logits:
            matches.append({
                "cantor_id": entry["cantor_id"],
                "layer": entry["layer"],
                "feature": entry["feature"],
                "matched_top_logits": matched_logits,
            })
    return {"query": query, "count": len(matches), "matches": matches}


def _json_response(handler: BaseHTTPRequestHandler, payload: bytes, status: int = 200) -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(payload)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(payload)


def _html_response(handler: BaseHTTPRequestHandler, body: bytes) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _file_response(handler: BaseHTTPRequestHandler, path: Path) -> None:
    if not path.is_file():
        handler.send_error(404, "Not found")
        return
    ctype, _ = mimetypes.guess_type(str(path))
    if ctype is None:
        ctype = "application/octet-stream"
    data = path.read_bytes()
    handler.send_response(200)
    handler.send_header("Content-Type", ctype)
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "public, max-age=3600")
    handler.end_headers()
    handler.wfile.write(data)


def make_handler_class(
    data_dir: Path,
    annotations_file: Path | None = None,
    prompt_output_dir: Path | None = None,
):
    data_dir = data_dir.resolve()
    annotations_path = (annotations_file or (data_dir / "feature_annotations.json")).resolve()
    prompt_output_dir = (prompt_output_dir or _DEFAULT_PROMPT_OUTPUT_DIR).resolve()
    annotations_lock = threading.Lock()
    prompt_save_lock = threading.Lock()
    feature_index_by_cantor: dict[str, dict] = {}
    tokens_per_domain: dict[str, int] = {}
    histogram_layer_cache: dict[int, dict] = {}
    top_logit_index: list[dict] | None = None
    top_logit_index_lock = threading.Lock()
    metadata_path = data_dir / "feature_metadata.json"
    histograms_file = "activation_histograms.npz"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text())
            histograms_file = metadata.get("activation_histograms_file") or histograms_file
            tokens_per_domain = metadata.get("tokens_per_domain") or {}
            feature_index_by_cantor = {
                str(feature["cantor_id"]): feature
                for feature in metadata.get("features") or []
            }
        except Exception as exc:
            logger.warning("Could not pre-load feature metadata index: %s", exc)

    class DashboardHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args) -> None:
            logger.debug("%s - %s", self.address_string(), fmt % args)

        def do_GET(self) -> None:  # noqa: N802
            nonlocal top_logit_index
            parsed = urlparse(self.path)
            path = unquote(parsed.path)

            if path == "/" or path == "/index.html" or re.match(r"^/cantor/\d+$", path):
                html_path = _STATIC_DIR / "dashboard.html"
                if not html_path.is_file():
                    _html_response(
                        self,
                        b"<h1>Missing dashboard.html next to feature_dashboard.py</h1>",
                    )
                    return
                _html_response(self, html_path.read_bytes())
                return

            if path == "/api/metadata":
                meta_path = data_dir / "feature_metadata.json"
                if not meta_path.is_file():
                    _json_response(
                        self,
                        json.dumps({"error": f"Missing {meta_path}"}).encode(),
                        status=404,
                    )
                    return
                _json_response(self, meta_path.read_bytes())
                return

            if path == "/api/annotations":
                try:
                    with annotations_lock:
                        annotations = _load_annotations(annotations_path)
                    _json_response(self, json.dumps(annotations).encode())
                except Exception as exc:
                    _json_response(
                        self,
                        json.dumps({"error": str(exc)}).encode(),
                        status=500,
                    )
                return

            if path == "/api/logit_search":
                query = parse_qs(parsed.query).get("q", [""])[0]
                try:
                    with top_logit_index_lock:
                        if top_logit_index is None:
                            top_logit_index = _build_top_logit_index(data_dir)
                    payload = _search_top_logits(top_logit_index, query)
                    _json_response(self, json.dumps(payload).encode())
                except Exception as exc:
                    _json_response(
                        self,
                        json.dumps({"error": str(exc)}).encode(),
                        status=500,
                    )
                return

            m = re.match(r"^/api/feature/(\d+)$", path)
            if m:
                cantor_id = m.group(1)
                feat_path = data_dir / "features" / f"{cantor_id}.json"
                if not feat_path.is_file():
                    _json_response(
                        self,
                        json.dumps({"error": f"No feature file {feat_path.name}"}).encode(),
                        status=404,
                    )
                    return
                _json_response(self, feat_path.read_bytes())
                return

            m = re.match(r"^/api/feature_hist/(\d+)$", path)
            if m:
                cantor_id = m.group(1)
                feature_meta = feature_index_by_cantor.get(cantor_id)
                if feature_meta is None:
                    _json_response(
                        self,
                        json.dumps({"error": f"No metadata for feature {cantor_id}"}).encode(),
                        status=404,
                    )
                    return
                try:
                    payload = _load_feature_histogram_payload(
                        data_dir=data_dir,
                        histograms_file=histograms_file,
                        feature_meta=feature_meta,
                        tokens_per_domain=tokens_per_domain,
                        layer_cache=histogram_layer_cache,
                    )
                    _json_response(self, json.dumps(payload).encode())
                except Exception as exc:
                    _json_response(
                        self,
                        json.dumps({"error": str(exc)}).encode(),
                        status=404,
                    )
                return

            # Optional: serve other static files from _STATIC_DIR
            if path.startswith("/static/"):
                rel = path[len("/static/") :].lstrip("/")
                candidate = (_STATIC_DIR / rel).resolve()
                if candidate.is_file() and str(candidate).startswith(str(_STATIC_DIR)):
                    _file_response(self, candidate)
                    return

            self.send_error(404, "Not found")

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = unquote(parsed.path)

            if path == "/api/save_prompt":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(length) or b"{}")
                    if not isinstance(payload, dict):
                        raise ValueError("Expected JSON object")
                    with prompt_save_lock:
                        result = _save_prompt_example(
                            prompt_output_dir=prompt_output_dir,
                            data_dir=data_dir,
                            payload=payload,
                        )
                    _json_response(self, json.dumps(result).encode())
                except Exception as exc:
                    _json_response(
                        self,
                        json.dumps({"error": str(exc)}).encode(),
                        status=400,
                    )
                return

            if path == "/api/logit_search":
                try:
                    nonlocal top_logit_index
                    length = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(length) or b"{}")
                    if not isinstance(payload, dict):
                        raise ValueError("Expected JSON object")
                    query = str(payload.get("query") or "")
                    with top_logit_index_lock:
                        if top_logit_index is None:
                            top_logit_index = _build_top_logit_index(data_dir)
                    result = _search_top_logits(top_logit_index, query)
                    _json_response(self, json.dumps(result).encode())
                except Exception as exc:
                    _json_response(
                        self,
                        json.dumps({"error": str(exc)}).encode(),
                        status=400,
                    )
                return

            if path == "/api/source_transcript":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(length) or b"{}")
                    if not isinstance(payload, dict):
                        raise ValueError("Expected JSON object")
                    source_metadata = payload.get("source_metadata")
                    if not isinstance(source_metadata, dict):
                        raise ValueError("source_metadata must be an object")
                    result = _load_source_transcript(source_metadata)
                    _json_response(self, json.dumps(result).encode())
                except Exception as exc:
                    _json_response(
                        self,
                        json.dumps({"error": str(exc)}).encode(),
                        status=400,
                    )
                return

            if path != "/api/annotations":
                self.send_error(404, "Not found")
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                cantor_id = str(payload["cantor_id"])
                tags = payload.get("tags", [])
                notes = payload.get("notes", "")
                if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
                    raise ValueError("tags must be a list of strings")
                if not isinstance(notes, str):
                    raise ValueError("notes must be a string")
                clean_tags = sorted({t.strip() for t in tags if t.strip()})
                with annotations_lock:
                    annotations = _load_annotations(annotations_path)
                    entry = {
                        **(annotations.get(cantor_id) or {}),
                        "tags": clean_tags,
                        "notes": notes,
                    }
                    annotations[cantor_id] = entry
                    _save_annotations(annotations_path, annotations)
                _json_response(self, json.dumps({"ok": True, "annotation": entry}).encode())
            except Exception as exc:
                _json_response(
                    self,
                    json.dumps({"error": str(exc)}).encode(),
                    status=400,
                )

    return DashboardHandler


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Browse feature activation collection outputs in a local dashboard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing feature_metadata.json and features/",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind address")
    parser.add_argument("--port", type=int, default=8765, help="Port")
    parser.add_argument(
        "--annotations_file",
        type=str,
        default=None,
        help="JSON file for persistent feature annotations. Defaults to [data_dir]/feature_annotations.json",
    )
    parser.add_argument(
        "--prompt_output_dir",
        type=str,
        default=str(_DEFAULT_PROMPT_OUTPUT_DIR),
        help="Directory where saved raw attribution prompts are written",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open a browser tab automatically",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"data_dir is not a directory: {data_dir}")

    annotations_file = Path(args.annotations_file) if args.annotations_file else None
    prompt_output_dir = Path(args.prompt_output_dir)
    handler = make_handler_class(data_dir, annotations_file, prompt_output_dir)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/"
    logger.info("Feature dashboard serving %s at %s", data_dir, url)
    logger.info("Annotations file: %s", (annotations_file or (data_dir / "feature_annotations.json")).resolve())
    logger.info("Saved prompts directory: %s", prompt_output_dir.resolve())
    logger.info("Press Ctrl+C to stop.")

    if not args.no_open:

        def _open() -> None:
            webbrowser.open(url)

        threading.Timer(0.35, _open).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
