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


def _base_tokenizer_for(model_type: str) -> str | None:
    from models import canonical_architecture

    return _BASE_TOKENIZER.get(model_type) or _BASE_TOKENIZER.get(canonical_architecture(model_type))


def ensure_fast_tokenizer(tokenizer: "PreTrainedTokenizerBase", source: str) -> "PreTrainedTokenizerBase":
    """Return ``tokenizer`` if it is a fast (Rust) tokenizer, else raise.

    We require the fast tokenizer everywhere. The slow (pure-Python, sentencepiece-backed)
    tokenizer is only ever reached as a silent fallback when the fast ``tokenizer.json`` is
    missing/unbuildable — which masks real problems (e.g. a checkpoint uploaded without
    tokenizer files). Fail loudly instead of degrading to it.
    """
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            f"Loaded a SLOW (non-fast) tokenizer from {source!r}: {type(tokenizer).__name__}. "
            "Refusing to use the slow tokenizer. Ensure a fast tokenizer.json is available "
            "(e.g. re-upload the checkpoint's tokenizer files), or pass an explicit tokenizer "
            "path to a repo that has one."
        )
    return tokenizer


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

    # use_fast=True asks for the Rust tokenizer; ensure_fast_tokenizer then errors if a slow
    # tokenizer was returned anyway (e.g. when only sentencepiece files are present).
    if tokenizer_path is not None:
        logger.info(f"Loading tokenizer from explicit path: {tokenizer_path}")
        return ensure_fast_tokenizer(
            AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=True),
            tokenizer_path,
        )

    config = None
    try:
        logger.info(f"Loading tokenizer from checkpoint: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
        if getattr(tokenizer, "chat_template", None) is not None:
            return ensure_fast_tokenizer(tokenizer, model_path)

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model_type: str = getattr(config, "model_type", "")
        source = _base_tokenizer_for(model_type)
        if source is not None:
            logger.warning(
                f"Tokenizer from checkpoint has no chat_template; "
                f"falling back to base model tokenizer: {source}"
            )
            return ensure_fast_tokenizer(
                AutoTokenizer.from_pretrained(source, trust_remote_code=True, use_fast=True), source
            )

        logger.warning(
            "Tokenizer from checkpoint has no chat_template and model_type "
            f"{model_type!r} has no known base tokenizer. Use --tokenizer for chat data."
        )
        return ensure_fast_tokenizer(tokenizer, model_path)
    except (OSError, AttributeError, KeyError, TypeError, ValueError, ImportError) as exc:
        logger.warning(f"Could not load tokenizer from checkpoint ({exc}); trying base tokenizer fallback")

    if config is None:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type: str = getattr(config, "model_type", "")
    source = _base_tokenizer_for(model_type)

    if source is None:
        raise RuntimeError(
            f"Checkpoint {model_path!r} has no tokenizer files and model_type "
            f"{model_type!r} has no known base tokenizer. "
            f"Pass --tokenizer explicitly."
        )

    logger.info(f"No tokenizer in checkpoint, falling back to base model: {source}")
    return ensure_fast_tokenizer(
        AutoTokenizer.from_pretrained(source, trust_remote_code=True, use_fast=True), source
    )


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
        from models import checkpoint_load_kwargs_for_model_type, get_transcoder_classes_for_model_type

        hf_config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
        arch = hf_config.model_type  # e.g. "qwen2", "gemma2"

        try:
            _config_cls, model_cls = get_transcoder_classes_for_model_type(arch)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported model_type '{arch}' for transcoder model at "
                f"'{pretrained_model_name_or_path}': {exc}"
            ) from exc
        for key, value in checkpoint_load_kwargs_for_model_type(arch).items():
            kwargs.setdefault(key, value)
        return model_cls.from_pretrained(pretrained_model_name_or_path, **kwargs)

    @staticmethod
    def detect_special_tokens(
        tokenizer: "PreTrainedTokenizerBase",
        model_type: str | None = None,
    ) -> SpecialTokenIds:
        """Detect special token IDs for a given tokenizer and architecture."""
        return detect_special_tokens(tokenizer, model_type=model_type)
