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
from urllib.parse import unquote, urlparse

from helpers.log import logger, setup_logging

_STATIC_DIR = Path(__file__).resolve().parent


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


def make_handler_class(data_dir: Path, annotations_file: Path | None = None):
    data_dir = data_dir.resolve()
    annotations_path = (annotations_file or (data_dir / "feature_annotations.json")).resolve()
    annotations_lock = threading.Lock()

    class DashboardHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args) -> None:
            logger.debug("%s - %s", self.address_string(), fmt % args)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = unquote(parsed.path)

            if path == "/" or path == "/index.html":
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
        "--no-open",
        action="store_true",
        help="Do not open a browser tab automatically",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"data_dir is not a directory: {data_dir}")

    annotations_file = Path(args.annotations_file) if args.annotations_file else None
    handler = make_handler_class(data_dir, annotations_file)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/"
    logger.info("Feature dashboard serving %s at %s", data_dir, url)
    logger.info("Annotations file: %s", (annotations_file or (data_dir / "feature_annotations.json")).resolve())
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
