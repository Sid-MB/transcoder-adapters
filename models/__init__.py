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
_REGISTRY: dict[str, ModelOutputTypes] = {}


def _load_architecture(arch: str) -> ModelOutputTypes:
    if arch not in _REGISTRY:
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
    return _REGISTRY[arch]


def get_transcoder_classes(arch: str) -> ModelOutputTypes:
    if arch not in _KNOWN_ARCHITECTURES:
        available = ", ".join(available_architectures())
        raise ValueError(f"Unknown architecture: '{arch}'. Available: {available}")

    return _load_architecture(arch)


def detect_architecture(model_name: str) -> str:
    name_lower = model_name.lower()

    if "qwen" in name_lower:
        return "qwen2"
    if "google/gemma-4" in name_lower or "gemma-4" in name_lower or "gemma4" in name_lower:
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
    return sorted(_KNOWN_ARCHITECTURES)
