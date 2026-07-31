"""Model registry for transcoder-adapted architectures.

Provides a unified interface to look up the correct Config and Model classes
for any supported architecture (qwen2, gemma2, gemma4, etc.).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from models.gemma2_transcoder import Gemma2ConfigWithTranscoder, Gemma2ForCausalLMWithTranscoder
    from models.gemma4_transcoder import Gemma4ConfigWithTranscoder, Gemma4ForCausalLMWithTranscoder
    from models.qwen2_transcoder import Qwen2ConfigWithTranscoder, Qwen2ForCausalLMWithTranscoder
    from transformers import Gemma2Config, Gemma2ForCausalLM


ModelOutputTypes = tuple[type[Any], type[Any]]

_KNOWN_ARCHITECTURES = ("qwen2", "gemma2", "gemma2-orig", "gemma4")
_ARCHITECTURE_ALIASES = {
    "gemma4_text": "gemma4",
}
_GEMMA4_LANGUAGE_MODEL_KEY_MAPPING = {
    r"^model\.language_model\.": "model.",
}
_REGISTRY: dict[str, ModelOutputTypes] = {}


def canonical_architecture(arch: str) -> str:
    return _ARCHITECTURE_ALIASES.get(arch, arch)


def validate_architecture_name(arch: str) -> str:
    if arch in _ARCHITECTURE_ALIASES:
        canonical = _ARCHITECTURE_ALIASES[arch]
        raise ValueError(
            f"Use model_arch: '{canonical}', not HF checkpoint model_type '{arch}'. "
            f"Available model_arch values: {', '.join(available_architectures())}"
        )
    if arch not in _KNOWN_ARCHITECTURES:
        available = ", ".join(available_architectures())
        raise ValueError(f"Unknown architecture: '{arch}'. Available: {available}")
    return arch


def _architecture_unavailable_error(arch: str, exc: ImportError) -> ValueError:
    detail = f"Architecture '{arch}' is unavailable because its backend could not be imported"
    if arch == "gemma4":
        detail += "; install a transformers build with Gemma4 support"
    return ValueError(f"{detail}: {exc}")


def _load_architecture(arch: str) -> ModelOutputTypes:
    arch = canonical_architecture(arch)
    if arch not in _REGISTRY:
        try:
            match arch:
                case "qwen2":
                    from models.qwen2_transcoder import Qwen2ConfigWithTranscoder, Qwen2ForCausalLMWithTranscoder

                    _REGISTRY[arch] = (Qwen2ConfigWithTranscoder, Qwen2ForCausalLMWithTranscoder)
                case "gemma2":
                    from models.gemma2_transcoder import Gemma2ConfigWithTranscoder, Gemma2ForCausalLMWithTranscoder

                    _REGISTRY[arch] = (Gemma2ConfigWithTranscoder, Gemma2ForCausalLMWithTranscoder)
                case "gemma2-orig":
                    from transformers import Gemma2Config, Gemma2ForCausalLM

                    _REGISTRY[arch] = (Gemma2Config, Gemma2ForCausalLM)
                case "gemma4":
                    from models.gemma4_transcoder import Gemma4ConfigWithTranscoder, Gemma4ForCausalLMWithTranscoder

                    _REGISTRY[arch] = (Gemma4ConfigWithTranscoder, Gemma4ForCausalLMWithTranscoder)
                case _:
                    raise ValueError(f"Unknown architecture: '{arch}'. Available: {', '.join(available_architectures())}")
        except ImportError as exc:
            raise _architecture_unavailable_error(arch, exc) from exc
    return _REGISTRY[arch]


def get_transcoder_classes(arch: str) -> ModelOutputTypes:
    return _load_architecture(validate_architecture_name(arch))


def get_transcoder_classes_for_model_type(model_type: str) -> ModelOutputTypes:
    canonical_arch = canonical_architecture(model_type)
    if canonical_arch not in _KNOWN_ARCHITECTURES:
        available = ", ".join(available_architectures())
        aliases = ", ".join(f"{alias} -> {canonical}" for alias, canonical in _ARCHITECTURE_ALIASES.items())
        raise ValueError(
            f"Unknown checkpoint model_type: '{model_type}'. Available model_arch values: {available}. "
            f"Accepted checkpoint aliases: {aliases}"
        )

    return _load_architecture(canonical_arch)


def checkpoint_load_kwargs_for_model_type(model_type: str) -> dict[str, Any]:
    """Return compatibility kwargs for loading saved transcoder checkpoints."""
    if canonical_architecture(model_type) == "gemma4":
        return {"key_mapping": dict(_GEMMA4_LANGUAGE_MODEL_KEY_MAPPING)}
    return {}


def detect_architecture(model_name: str) -> str:
    name_lower = model_name.lower()

    if "qwen" in name_lower:
        return "qwen2"
    if "google/gemma-4" in name_lower or "gemma-4" in name_lower or "gemma4" in name_lower:
        _load_architecture("gemma4")
        return "gemma4"
    if "google/gemma-2" in name_lower:
        from helpers.log import logger

        logger.info(f"Using original (non-transcoder) Gemma2 architecture for {model_name}")
        return "gemma2-orig"
    if "gemma2" in name_lower:
        return "gemma2"
    raise ValueError(
        f"Cannot auto-detect architecture for '{model_name}'. "
        "Set 'model_arch' explicitly in your config (e.g. model_arch: qwen2)."
    )


def available_architectures() -> list[str]:
    available: list[str] = []
    for arch in _KNOWN_ARCHITECTURES:
        try:
            _load_architecture(arch)
        except ValueError as exc:
            if not isinstance(exc.__cause__, ImportError):
                raise
            continue
        available.append(arch)
    return sorted(available)
