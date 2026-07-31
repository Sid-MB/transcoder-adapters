"""Auto-dispatching loader for RelP transcoder attribution models."""

from __future__ import annotations

from typing import Any


_RELP_REGISTRY: dict[str, type] = {}


def _ensure_relp_registered():
    if _RELP_REGISTRY:
        return

    from models.gemma2_with_transcoder_relp import Gemma2ForCausalLMWithTranscoderRelP
    from models.qwen2_with_transcoder_relp import Qwen2ForCausalLMWithTranscoderRelP

    _RELP_REGISTRY["gemma2"] = Gemma2ForCausalLMWithTranscoderRelP
    _RELP_REGISTRY["qwen2"] = Qwen2ForCausalLMWithTranscoderRelP


def get_relp_model_class(model_type: str) -> type:
    _ensure_relp_registered()
    if model_type not in _RELP_REGISTRY:
        available = ", ".join(sorted(_RELP_REGISTRY))
        raise ValueError(f"Unsupported RelP model_type '{model_type}'. Available: {available}")
    return _RELP_REGISTRY[model_type]


class AutoModelForCausalLMWithTranscoderRelP:
    """Auto-dispatching loader for RelP transcoder attribution models."""

    def __init__(self):
        raise EnvironmentError(
            "AutoModelForCausalLMWithTranscoderRelP is not meant to be instantiated. "
            "Use AutoModelForCausalLMWithTranscoderRelP.from_pretrained() instead."
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs: Any):
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
        model_cls = get_relp_model_class(hf_config.model_type)
        return model_cls.from_pretrained(pretrained_model_name_or_path, **kwargs)
