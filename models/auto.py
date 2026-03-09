"""Auto-dispatching loader for transcoder-adapted causal LM models."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

from models.tokens import SpecialTokenIds, detect_special_tokens


class AutoModelForCausalLMWithTranscoder:
    """Auto-dispatching loader for transcoder-adapted causal LM models.

    Works like ``transformers.AutoModelForCausalLM``: reads the saved config
    to detect the architecture, then delegates to the correct model class.
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoModelForCausalLMWithTranscoder is not meant to be instantiated. "
            "Use AutoModelForCausalLMWithTranscoder.from_pretrained() instead."
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs: Any) -> "PreTrainedModel":
        from transformers import AutoConfig
        from models import _ensure_registered, _REGISTRY

        hf_config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
        arch = hf_config.model_type  # e.g. "qwen2", "gemma2"

        _ensure_registered()
        if arch not in _REGISTRY:
            available = ", ".join(sorted(_REGISTRY.keys()))
            raise ValueError(
                f"Unsupported model_type '{arch}' for transcoder model at "
                f"'{pretrained_model_name_or_path}'. Available: {available}"
            )

        _config_cls, model_cls = _REGISTRY[arch]
        return model_cls.from_pretrained(pretrained_model_name_or_path, **kwargs)

    @staticmethod
    def detect_special_tokens(
        tokenizer: "PreTrainedTokenizerBase",
        model_type: str | None = None,
    ) -> SpecialTokenIds:
        """Detect special token IDs for a given tokenizer and architecture."""
        return detect_special_tokens(tokenizer, model_type=model_type)
