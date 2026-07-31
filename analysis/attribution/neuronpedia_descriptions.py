"""Fetch Neuronpedia LLM feature descriptions, with a persistent on-disk cache.

<!-- Created by Claude Code session "implement: new 07/02 gemmascope transcoder experiments". 2026-07-09. -->

Neuronpedia hosts auto-interp (LLM-written) descriptions for the GemmaScope gemma-2-2b
transcoders under sources named ``{layer}-gemmascope-transcoder-16k``. Each feature's description
lives in ``SAEFeature.get(...).jsonData['explanations'][0]['description']``. This wraps that behind
a disk cache so we never re-fetch the same feature (there are up to 26x16384 of them), and so a
`retag` run over graphs stays fast and offline after the first pass.

Requires an API key: ``$NEURONPEDIA_API_KEY`` or ``~/.shell/secrets/neuronpedia_api_key``.
``neuronpedia`` is in the ``viz`` extra (``uv add neuronpedia --optional viz``).

Feature index convention: circuit-tracer graph nodes store ``node['feature']`` as the
Cantor pairing of ``(layer, per_layer_feature_index)`` — use :func:`cantor_unpair` to recover
``(layer, feat)``; the Neuronpedia index is ``feat`` and the source encodes ``layer``.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

from helpers.log import logger

DEFAULT_SOURCE_TEMPLATE = "{layer}-gemmascope-transcoder-16k"


def _default_cache_path() -> Path:
    root = os.environ.get("LARGE_ARTIFACTS_DIR") or f"/nlp/scr/{os.environ.get('USER', '')}"
    return Path(root) / "transcoder-adapters" / "neuronpedia_cache" / "descriptions.json"


def _load_api_key() -> str | None:
    key = os.environ.get("NEURONPEDIA_API_KEY")
    if key:
        return key.strip()
    secret = Path.home() / ".shell" / "secrets" / "neuronpedia_api_key"
    if secret.exists():
        return secret.read_text().strip()
    return None


def cantor_unpair(z: int) -> tuple[int, int]:
    """Inverse of the Cantor pairing used by circuit-tracer: z -> (layer, feature)."""
    w = (math.isqrt(8 * z + 1) - 1) // 2
    t = w * (w + 1) // 2
    feat = z - t
    layer = w - feat
    return layer, feat


class NeuronpediaDescriptions:
    """Disk-cached fetcher. Cached value is the description string, or "" for a known miss."""

    def __init__(self, *, model: str = "gemma-2-2b", source_template: str = DEFAULT_SOURCE_TEMPLATE, cache_path: Path | None = None) -> None:
        self.model = model
        self.source_template = source_template
        self.cache_path = Path(cache_path) if cache_path else _default_cache_path()
        self._cache: dict[str, str] = {}
        if self.cache_path.exists():
            self._cache = json.loads(self.cache_path.read_text())
        self._dirty = False
        self._api_key = _load_api_key()
        self._ctx = None  # lazy neuronpedia api_key context
        self.hits = self.fetched = self.misses = self.errors = 0

    # -- fetch one feature's description (network), returns "" if none/unavailable --
    def _fetch(self, source: str, index: int) -> str:
        try:
            import neuronpedia
            from neuronpedia.np_sae_feature import SAEFeature

            if self._ctx is None:
                if not self._api_key:
                    raise RuntimeError("No Neuronpedia API key ($NEURONPEDIA_API_KEY or ~/.shell/secrets/neuronpedia_api_key)")
                self._ctx = neuronpedia.api_key(self._api_key)
                self._ctx.__enter__()
            f = SAEFeature.get(self.model, source, str(index))
            data = f.jsonData
            data = json.loads(data) if isinstance(data, str) else data
            exps = data.get("explanations") or []
            return (exps[0].get("description") or "").strip() if exps else ""
        except Exception as exc:  # 500s on missing features, network hiccups, etc.
            self.errors += 1
            logger.debug("Neuronpedia miss %s/%s: %s", source, index, exc)
            return ""

    def get_by_layer_feature(self, layer: int, feat: int) -> str:
        source = self.source_template.format(layer=layer)
        key = f"{self.model}/{source}/{feat}"
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        desc = self._fetch(source, feat)
        self._cache[key] = desc
        self._dirty = True
        self.fetched += 1
        if not desc:
            self.misses += 1
        return desc

    def get_by_cantor(self, feature_index: int) -> str:
        layer, feat = cantor_unpair(int(feature_index))
        return self.get_by_layer_feature(layer, feat)

    def save(self) -> None:
        if self._dirty:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._cache))
            logger.info("Saved %d Neuronpedia descriptions to %s", len(self._cache), self.cache_path)

    def close(self) -> None:
        self.save()
        if self._ctx is not None:
            try:
                self._ctx.__exit__(None, None, None)
            finally:
                self._ctx = None
