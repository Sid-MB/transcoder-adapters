"""Color tokens by next-token probability change from transcoder adapters."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import mimetypes
import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import torch

from analysis.attribution.run_attribution import _parse_chat_prompt_text
from helpers.log import logger, setup_logging
from helpers.paths import PRODUCTS_DIR
from models.auto import AutoModelForCausalLMWithTranscoder, load_tokenizer
from models.tokens import _input_ids_from_chat_template_output


_SLUG_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class TokenDelta:
    index: int
    token_id: int
    text: str
    logprob_with_adapter: float
    logprob_without_adapter: float
    prob_with_adapter: float
    prob_without_adapter: float
    delta_logprob: float
    delta_prob: float
    in_assistant: bool


def _slugify(value: str) -> str:
    return _SLUG_RE.sub("_", Path(value.rstrip("/")).stem or value).strip("_") or "run"


def _prompt_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def default_dashboard_cache_dir(
    *,
    model_path: str,
    prompts_dir: Path,
    prompt_path: Path,
    prompt_format: str,
    all_tokens: bool = False,
) -> Path:
    scope = "all_tokens" if all_tokens else "assistant_tokens"
    return (
        PRODUCTS_DIR
        / "adapter_token_deltas"
        / f"{_slugify(model_path)}_{_slugify(str(prompts_dir))}_{prompt_format}"
        / f"{_slugify(prompt_path.name)}_{_prompt_hash(prompt_path)}_{scope}"
    )


def _dashboard_cache_parent(*, model_path: str, prompts_dir: Path, prompt_format: str) -> Path:
    return (
        PRODUCTS_DIR
        / "adapter_token_deltas"
        / f"{_slugify(model_path)}_{_slugify(str(prompts_dir))}_{prompt_format}"
    )


def _cleanup_stale_dashboard_caches(
    *,
    model_path: str,
    prompts_dir: Path,
    prompt_path: Path,
    prompt_format: str,
    all_tokens: bool,
    current_output_dir: Path,
) -> list[Path]:
    parent = _dashboard_cache_parent(
        model_path=model_path,
        prompts_dir=prompts_dir,
        prompt_format=prompt_format,
    )
    if not parent.exists():
        return []
    scope = "all_tokens" if all_tokens else "assistant_tokens"
    prefix = f"{_slugify(prompt_path.name)}_"
    removed = []
    for candidate in parent.glob(f"{prefix}*_{scope}"):
        if candidate == current_output_dir or not candidate.is_dir():
            continue
        import shutil

        shutil.rmtree(candidate)
        removed.append(candidate)
    return removed


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _model_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    return next(model.parameters()).device


def _set_transcoder_disabled(model: Any, disabled: bool) -> None:
    if not hasattr(model, "_transcoder_mlps"):
        raise ValueError("Model does not expose _transcoder_mlps(); cannot disable adapters")
    for mlp in model._transcoder_mlps():
        mlp.disable_transcoder = disabled


@contextmanager
def _transcoder_disabled(model: Any, disabled: bool):
    mlps = list(model._transcoder_mlps())
    previous = [getattr(mlp, "disable_transcoder", False) for mlp in mlps]
    try:
        for mlp in mlps:
            mlp.disable_transcoder = disabled
        yield
    finally:
        for mlp, value in zip(mlps, previous, strict=True):
            mlp.disable_transcoder = value


def _parse_prompt_file(path: Path, prompt_format: str, tokenizer: Any) -> tuple[list[int], list[bool], str]:
    text = path.read_text()
    if prompt_format == "raw":
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        return token_ids, [True] * len(token_ids), tokenizer.decode(token_ids)

    parsed = _parse_chat_prompt_text(text)
    if parsed is None:
        raise ValueError(
            f"Prompt format 'chat' requires marked user/assistant text: {path}"
        )
    user_content, assistant_content = parsed
    prompt_ids = _input_ids_from_chat_template_output(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    assistant_ids = tokenizer.encode(assistant_content, add_special_tokens=False)
    token_ids = list(prompt_ids) + list(assistant_ids)
    assistant_mask = [False] * len(prompt_ids) + [True] * len(assistant_ids)
    return token_ids, assistant_mask, tokenizer.decode(token_ids)


def _next_token_logprobs(model: Any, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
    logprobs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    next_ids = input_ids[:, 1:].unsqueeze(-1)
    return logprobs.gather(dim=-1, index=next_ids).squeeze(0).squeeze(-1).cpu()


def compute_token_deltas(
    model: Any,
    tokenizer: Any,
    token_ids: list[int],
    assistant_mask: list[bool],
) -> list[TokenDelta]:
    if len(token_ids) < 2:
        raise ValueError("Need at least two tokens to compute next-token deltas")

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=_model_device(model))
    _set_transcoder_disabled(model, False)
    logprobs_with = _next_token_logprobs(model, input_ids)
    with _transcoder_disabled(model, True):
        logprobs_without = _next_token_logprobs(model, input_ids)

    deltas: list[TokenDelta] = []
    for offset, token_id in enumerate(token_ids[1:]):
        logprob_with = float(logprobs_with[offset].item())
        logprob_without = float(logprobs_without[offset].item())
        prob_with = math.exp(logprob_with)
        prob_without = math.exp(logprob_without)
        deltas.append(
            TokenDelta(
                index=offset + 1,
                token_id=int(token_id),
                text=tokenizer.decode([token_id], skip_special_tokens=False),
                logprob_with_adapter=logprob_with,
                logprob_without_adapter=logprob_without,
                prob_with_adapter=prob_with,
                prob_without_adapter=prob_without,
                delta_logprob=logprob_with - logprob_without,
                delta_prob=prob_with - prob_without,
                in_assistant=assistant_mask[offset + 1],
            )
        )
    return deltas


def _color_for_delta(delta: float, scale: float) -> str:
    if scale <= 0:
        return "rgba(245, 245, 245, 1)"
    intensity = min(abs(delta) / scale, 1.0)
    alpha = 0.12 + 0.72 * intensity
    if delta >= 0:
        return f"rgba(194, 65, 12, {alpha:.3f})"
    return f"rgba(37, 99, 235, {alpha:.3f})"


def _auto_scale(deltas: list[TokenDelta]) -> float:
    values = sorted(abs(delta.delta_logprob) for delta in deltas)
    if not values:
        return 1.0
    index = min(len(values) - 1, max(0, int(0.95 * (len(values) - 1))))
    return max(values[index], 1e-6)


def write_outputs(
    deltas: list[TokenDelta],
    output_dir: Path,
    *,
    prompt_path: Path,
    model_path: str,
    prompt_format: str,
    assistant_only: bool,
    scale: float | None = None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    visible_deltas = [delta for delta in deltas if delta.in_assistant or not assistant_only]
    color_scale = scale if scale is not None else _auto_scale(visible_deltas)

    json_path = output_dir / "token_probability_deltas.json"
    json_path.write_text(
        json.dumps(
            {
                "model_path": model_path,
                "prompt_file": str(prompt_path),
                "prompt_format": prompt_format,
                "assistant_only": assistant_only,
                "color_scale_delta_logprob": color_scale,
                "tokens": [delta.__dict__ for delta in deltas],
            },
            indent=2,
        )
    )

    spans = []
    for delta in deltas:
        if assistant_only and not delta.in_assistant:
            continue
        title = (
            f"token_id={delta.token_id}\\n"
            f"logp adapter={delta.logprob_with_adapter:.4f}\\n"
            f"logp disabled={delta.logprob_without_adapter:.4f}\\n"
            f"delta logp={delta.delta_logprob:+.4f}\\n"
            f"p adapter={delta.prob_with_adapter:.4g}\\n"
            f"p disabled={delta.prob_without_adapter:.4g}\\n"
            f"delta p={delta.delta_prob:+.4g}"
        )
        token_text = html.escape(delta.text).replace("\n", "↵\n")
        spans.append(
            "<span class=\"tok\" "
            f"style=\"background:{_color_for_delta(delta.delta_logprob, color_scale)}\" "
            f"title=\"{html.escape(title)}\">{token_text}</span>"
        )

    html_path = output_dir / "token_probability_deltas.html"
    html_path.write_text(
        f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Adapter Token Probability Deltas</title>
<style>
body {{
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  margin: 24px;
  color: #171717;
  background: #fafafa;
}}
.meta {{
  font-size: 13px;
  color: #525252;
  margin-bottom: 16px;
  line-height: 1.45;
}}
.legend {{
  display: flex;
  gap: 14px;
  align-items: center;
  margin: 12px 0 18px;
  font-size: 13px;
}}
.chip {{
  display: inline-block;
  width: 32px;
  height: 14px;
  border: 1px solid #d4d4d4;
  vertical-align: middle;
}}
.text {{
  white-space: pre-wrap;
  line-height: 1.9;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  font-size: 15px;
}}
.tok {{
  border-radius: 3px;
  padding: 2px 1px;
}}
</style>
</head>
<body>
<h1>Adapter Token Probability Deltas</h1>
<div class="meta">
  <div><strong>Model:</strong> {html.escape(model_path)}</div>
  <div><strong>Prompt:</strong> {html.escape(str(prompt_path))}</div>
  <div><strong>Prompt format:</strong> {html.escape(prompt_format)}</div>
  <div><strong>Shown tokens:</strong> {"assistant only" if assistant_only else "all tokens"}</div>
  <div><strong>Color value:</strong> log P(token | prefix, adapter on) - log P(token | prefix, adapter off)</div>
  <div><strong>Color scale:</strong> +/- {color_scale:.4f} logprob</div>
</div>
<div class="legend">
  <span><span class="chip" style="background:rgba(194,65,12,.75)"></span> adapter raises probability</span>
  <span><span class="chip" style="background:rgba(37,99,235,.75)"></span> adapter lowers probability</span>
</div>
<div class="text">{''.join(spans)}</div>
</body>
</html>
"""
    )
    return html_path, json_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare next-token probabilities with transcoder adapters enabled vs disabled "
            "and write an HTML token-color view."
        )
    )
    parser.add_argument("model_path", help="HF repo ID or local transcoder checkpoint")
    parser.add_argument("--prompt_file", type=Path, help="Prompt .txt file to score")
    parser.add_argument("--prompt_format", choices=["raw", "chat"], default="chat")
    parser.add_argument("--tokenizer", default=None, help="Optional tokenizer override")
    parser.add_argument("--output_dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--device_map", default="auto", help="Model device_map passed to from_pretrained")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument(
        "--all_tokens",
        action="store_true",
        help="Show prompt/template tokens too. By default chat prompts show assistant tokens only.",
    )
    parser.add_argument(
        "--color_scale",
        type=float,
        default=None,
        help="Absolute delta-logprob value used for full color saturation. Defaults to p95.",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Serve a local dashboard for on-demand rendering of a prompt directory.",
    )
    parser.add_argument(
        "--prompts_dir",
        type=Path,
        default=None,
        help="Prompt directory for --dashboard. Defaults to --prompt_file parent.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host for --dashboard")
    parser.add_argument("--port", type=int, default=8765, help="Port for --dashboard")
    return parser


class _DashboardModelCache:
    def __init__(
        self,
        *,
        model_path: str,
        tokenizer_path: str | None,
        dtype_name: str,
        device_map: str,
    ):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.dtype_name = dtype_name
        self.device_map = device_map
        self._lock = threading.Lock()
        self._model = None
        self._tokenizer = None

    def load(self):
        with self._lock:
            if self._model is not None and self._tokenizer is not None:
                return self._model, self._tokenizer
            logger.info(f"Loading model: {self.model_path}")
            self._model = AutoModelForCausalLMWithTranscoder.from_pretrained(
                self.model_path,
                torch_dtype=_dtype_from_name(self.dtype_name),
                device_map=self.device_map,
            )
            self._model.eval()
            self._tokenizer = load_tokenizer(self.model_path, tokenizer_path=self.tokenizer_path)
            return self._model, self._tokenizer


def render_prompt_to_cache(
    *,
    model_cache: _DashboardModelCache,
    prompt_path: Path,
    prompts_dir: Path,
    prompt_format: str,
    all_tokens: bool,
    color_scale: float | None,
) -> dict[str, str | bool]:
    output_dir = default_dashboard_cache_dir(
        model_path=model_cache.model_path,
        prompts_dir=prompts_dir,
        prompt_path=prompt_path,
        prompt_format=prompt_format,
        all_tokens=all_tokens,
    )
    stale_dirs = _cleanup_stale_dashboard_caches(
        model_path=model_cache.model_path,
        prompts_dir=prompts_dir,
        prompt_path=prompt_path,
        prompt_format=prompt_format,
        all_tokens=all_tokens,
        current_output_dir=output_dir,
    )
    html_path = output_dir / "token_probability_deltas.html"
    json_path = output_dir / "token_probability_deltas.json"
    if html_path.exists() and json_path.exists():
        logger.info(
            f"Token-delta cache skipped unchanged: {prompt_path.name} "
            f"h{_prompt_hash(prompt_path)} -> {output_dir}"
        )
        return {
            "cached": True,
            "output_dir": str(output_dir),
            "html_path": str(html_path),
            "json_path": str(json_path),
        }

    if stale_dirs:
        logger.info(
            f"Token-delta prompt content changed; removed stale cache dir(s) for "
            f"{prompt_path.name}, new hash h{_prompt_hash(prompt_path)}: "
            + ", ".join(path.name for path in stale_dirs)
        )
    else:
        logger.info(
            f"Token-delta new prompt render: {prompt_path.name} "
            f"h{_prompt_hash(prompt_path)} -> {output_dir}"
        )

    model, tokenizer = model_cache.load()
    token_ids, assistant_mask, decoded_text = _parse_prompt_file(
        prompt_path,
        prompt_format,
        tokenizer,
    )
    logger.info(f"Rendering {prompt_path}: {len(token_ids)} tokens")
    logger.info(f"Decoded text tail: {decoded_text[-120:]!r}")
    deltas = compute_token_deltas(model, tokenizer, token_ids, assistant_mask)
    assistant_only = prompt_format == "chat" and not all_tokens
    html_path, json_path = write_outputs(
        deltas,
        output_dir,
        prompt_path=prompt_path,
        model_path=model_cache.model_path,
        prompt_format=prompt_format,
        assistant_only=assistant_only,
        scale=color_scale,
    )
    return {
        "cached": False,
        "output_dir": str(output_dir),
        "html_path": str(html_path),
        "json_path": str(json_path),
    }


def _json_response(handler: BaseHTTPRequestHandler, payload: dict, status: int = 200) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _html_response(handler: BaseHTTPRequestHandler, body: str, status: int = 200) -> None:
    encoded = body.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(encoded)))
    handler.end_headers()
    handler.wfile.write(encoded)


def _file_response(handler: BaseHTTPRequestHandler, path: Path) -> None:
    if not path.exists() or not path.is_file():
        _json_response(handler, {"error": f"File not found: {path}"}, status=404)
        return
    body = path.read_bytes()
    content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    if path.suffix == ".html":
        content_type = "text/html; charset=utf-8"
    elif path.suffix == ".json":
        content_type = "application/json"
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _dashboard_shell(*, model_path: str, prompts_dir: Path, prompt_format: str) -> str:
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Adapter Token Delta Dashboard</title>
<style>
body {{ margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #171717; background: #f7f7f7; }}
.top {{ display: grid; grid-template-columns: minmax(240px, 360px) 1fr; gap: 14px; padding: 14px; border-bottom: 1px solid #d4d4d4; background: #fff; }}
.meta {{ font-size: 12px; color: #525252; line-height: 1.4; overflow-wrap: anywhere; }}
.controls {{ display: flex; align-items: end; gap: 10px; flex-wrap: wrap; }}
label {{ display: grid; gap: 4px; font-size: 12px; color: #404040; }}
select, button {{ font: inherit; font-size: 14px; }}
select {{ min-width: 340px; max-width: 680px; padding: 6px 8px; }}
button {{ padding: 7px 10px; cursor: pointer; }}
#status {{ font-size: 13px; color: #404040; min-height: 20px; margin-top: 6px; }}
#frame {{ width: 100%; height: calc(100vh - 96px); border: 0; background: #fafafa; }}
</style>
</head>
<body>
<div class="top">
  <div class="meta">
    <div><strong>Model:</strong> {html.escape(model_path)}</div>
    <div><strong>Prompts:</strong> {html.escape(str(prompts_dir))}</div>
    <div><strong>Format:</strong> {html.escape(prompt_format)}</div>
  </div>
  <div>
    <div class="controls">
      <label>Prompt <select id="prompt"></select></label>
      <button id="render">Render</button>
      <button id="json">JSON</button>
      <label><span><input type="checkbox" id="allTokens"> all tokens</span></label>
    </div>
    <div id="status"></div>
  </div>
</div>
<iframe id="frame"></iframe>
<script>
const promptSelect = document.getElementById('prompt');
const statusEl = document.getElementById('status');
const frame = document.getElementById('frame');
const allTokens = document.getElementById('allTokens');
let lastJsonUrl = null;

async function loadPrompts() {{
  const res = await fetch('/api/prompts');
  const data = await res.json();
  promptSelect.innerHTML = '';
  for (const item of data.prompts) {{
    const opt = document.createElement('option');
    opt.value = item.name;
    opt.textContent = item.name;
    promptSelect.appendChild(opt);
  }}
}}

async function renderSelected() {{
  const name = promptSelect.value;
  if (!name) return;
  statusEl.textContent = 'Rendering...';
  frame.removeAttribute('src');
  const params = new URLSearchParams({{ prompt: name }});
  if (allTokens.checked) params.set('all_tokens', '1');
  const res = await fetch('/api/render?' + params.toString(), {{ method: 'POST' }});
  const data = await res.json();
  if (!res.ok) {{
    statusEl.textContent = data.error || 'Render failed';
    return;
  }}
  statusEl.textContent = data.cached ? 'Loaded cached render' : 'Rendered and cached';
  frame.src = data.html_url;
  lastJsonUrl = data.json_url;
}}

document.getElementById('render').addEventListener('click', renderSelected);
document.getElementById('json').addEventListener('click', () => {{ if (lastJsonUrl) window.open(lastJsonUrl, '_blank'); }});
promptSelect.addEventListener('change', renderSelected);
allTokens.addEventListener('change', renderSelected);
loadPrompts().then(renderSelected).catch(err => {{ statusEl.textContent = String(err); }});
</script>
</body>
</html>
"""


def serve_dashboard(
    *,
    model_path: str,
    prompts_dir: Path,
    prompt_format: str,
    tokenizer_path: str | None,
    dtype_name: str,
    device_map: str,
    port: int,
    host: str,
    color_scale: float | None,
) -> ThreadingHTTPServer:
    prompts_dir = prompts_dir.resolve()
    if not prompts_dir.exists() or not prompts_dir.is_dir():
        raise ValueError(f"Prompt directory does not exist: {prompts_dir}")
    model_cache = _DashboardModelCache(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        dtype_name=dtype_name,
        device_map=device_map,
    )

    class DashboardHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            logger.info("%s - %s", self.address_string(), format % args)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    _html_response(
                        self,
                        _dashboard_shell(
                            model_path=model_path,
                            prompts_dir=prompts_dir,
                            prompt_format=prompt_format,
                        ),
                    )
                    return
                if parsed.path == "/api/prompts":
                    prompts = [{"name": path.name, "path": str(path)} for path in sorted(prompts_dir.glob("*.txt"))]
                    _json_response(self, {"prompts": prompts})
                    return
                if parsed.path.startswith("/cache/"):
                    relative = parsed.path.removeprefix("/cache/")
                    _file_response(self, (PRODUCTS_DIR / relative).resolve())
                    return
                _json_response(self, {"error": "Not found"}, status=404)
            except Exception as exc:
                logger.exception("Dashboard GET failed")
                _json_response(self, {"error": str(exc)}, status=500)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path != "/api/render":
                    _json_response(self, {"error": "Not found"}, status=404)
                    return
                params = parse_qs(parsed.query)
                prompt_name = params.get("prompt", [""])[0]
                prompt_path = (prompts_dir / prompt_name).resolve()
                if prompt_path.parent != prompts_dir or prompt_path.suffix != ".txt" or not prompt_path.exists():
                    _json_response(self, {"error": f"Invalid prompt: {prompt_name}"}, status=400)
                    return
                result = render_prompt_to_cache(
                    model_cache=model_cache,
                    prompt_path=prompt_path,
                    prompts_dir=prompts_dir,
                    prompt_format=prompt_format,
                    all_tokens=params.get("all_tokens", ["0"])[0] == "1",
                    color_scale=color_scale,
                )
                html_path = Path(str(result["html_path"]))
                json_path = Path(str(result["json_path"]))
                _json_response(
                    self,
                    {
                        **result,
                        "html_url": f"/cache/{html_path.relative_to(PRODUCTS_DIR)}",
                        "json_url": f"/cache/{json_path.relative_to(PRODUCTS_DIR)}",
                    },
                )
            except Exception as exc:
                logger.exception("Dashboard render failed")
                _json_response(self, {"error": str(exc)}, status=500)

    server = ThreadingHTTPServer((host, port), DashboardHandler)
    logger.info(f"Adapter token-delta dashboard serving {prompts_dir} at http://{host}:{port}/")
    return server


def run_once(args: argparse.Namespace) -> None:
    if args.prompt_file is None:
        raise ValueError("--prompt_file is required unless --dashboard is set")
    if args.output_dir is None:
        model_cache = _DashboardModelCache(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer,
            dtype_name=args.dtype,
            device_map=args.device_map,
        )
        result = render_prompt_to_cache(
            model_cache=model_cache,
            prompt_path=args.prompt_file,
            prompts_dir=args.prompt_file.parent,
            prompt_format=args.prompt_format,
            all_tokens=args.all_tokens,
            color_scale=args.color_scale,
        )
        logger.info(f"Wrote HTML: {result['html_path']}")
        logger.info(f"Wrote JSON: {result['json_path']}")
        return

    logger.info(f"Loading model: {args.model_path}")
    model = AutoModelForCausalLMWithTranscoder.from_pretrained(
        args.model_path,
        torch_dtype=_dtype_from_name(args.dtype),
        device_map=args.device_map,
    )
    model.eval()
    tokenizer = load_tokenizer(args.model_path, tokenizer_path=args.tokenizer)

    token_ids, assistant_mask, decoded_text = _parse_prompt_file(
        args.prompt_file,
        args.prompt_format,
        tokenizer,
    )
    logger.info(f"Loaded {len(token_ids)} tokens from {args.prompt_file}")
    logger.info(f"Decoded text tail: {decoded_text[-120:]!r}")

    deltas = compute_token_deltas(model, tokenizer, token_ids, assistant_mask)
    assistant_only = args.prompt_format == "chat" and not args.all_tokens
    output_dir = args.output_dir
    html_path, json_path = write_outputs(
        deltas,
        output_dir,
        prompt_path=args.prompt_file,
        model_path=args.model_path,
        prompt_format=args.prompt_format,
        assistant_only=assistant_only,
        scale=args.color_scale,
    )
    logger.info(f"Wrote HTML: {html_path}")
    logger.info(f"Wrote JSON: {json_path}")


def main() -> None:
    setup_logging()
    args = build_parser().parse_args()
    if args.dashboard:
        prompts_dir = args.prompts_dir or (args.prompt_file.parent if args.prompt_file is not None else None)
        if prompts_dir is None:
            raise ValueError("--dashboard requires --prompts_dir or --prompt_file")
        server = serve_dashboard(
            model_path=args.model_path,
            prompts_dir=prompts_dir,
            prompt_format=args.prompt_format,
            tokenizer_path=args.tokenizer,
            dtype_name=args.dtype,
            device_map=args.device_map,
            port=args.port,
            host=args.host,
            color_scale=args.color_scale,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Stopping dashboard...")
        finally:
            server.server_close()
        return

    run_once(args)


if __name__ == "__main__":
    main()
