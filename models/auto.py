"""Auto-dispatching loader for transcoder-adapted causal LM models."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

from helpers.log import logger
from models.tokens import SpecialTokenIds, detect_special_tokens

# Fallback tokenizer source for each model_type, used only for legacy
# checkpoints that were uploaded without tokenizer files.
_BASE_TOKENIZER: dict[str, str] = {
    "gemma2": "google/gemma-2-2b-it",
    "qwen2": "Qwen/Qwen2-0.5B",
    "gemma4": "google/gemma-4-E2B-it",
}


def load_tokenizer(
    model_path: str,
    tokenizer_path: str | None = None,
) -> "PreTrainedTokenizerBase":
    """Load a tokenizer for a transcoder checkpoint.

    Tries ``model_path`` first (works for checkpoints that include tokenizer
    files).  Falls back to the canonical base-model tokenizer for the
    architecture when the checkpoint lacks tokenizer files.

    Args:
        model_path: HF repo ID or local path to the transcoder checkpoint.
        tokenizer_path: Explicit override — when set, loaded directly.
    """
    from transformers import AutoConfig, AutoTokenizer

    if tokenizer_path is not None:
        logger.info(f"Loading tokenizer from explicit path: {tokenizer_path}")
        return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    try:
        logger.info(f"Loading tokenizer from checkpoint: {model_path}")
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except (OSError, AttributeError, KeyError, TypeError) as exc:
        logger.warning(f"Could not load tokenizer from checkpoint ({exc}); trying base tokenizer fallback")

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type: str = getattr(config, "model_type", "")

    if model_type not in _BASE_TOKENIZER:
        raise RuntimeError(
            f"Checkpoint {model_path!r} has no tokenizer files and model_type "
            f"{model_type!r} has no known base tokenizer. "
            f"Pass --tokenizer explicitly."
        )

    source = _BASE_TOKENIZER[model_type]
    logger.info(f"No tokenizer in checkpoint, falling back to base model: {source}")
    return AutoTokenizer.from_pretrained(source, trust_remote_code=True)


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
