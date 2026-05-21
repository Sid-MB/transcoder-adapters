"""Load validation data from various sources for feature collection.

Supports:
- Local JSONL files or hf:// JSONL URIs (OpenThoughts format)
- HF dataset IDs with chat conversation columns (LMSYS-style)
- HF datasets with pre-tokenized input_ids
- HF datasets with raw text columns
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from helpers.log import logger


@dataclass(frozen=True)
class FeatureDataSourceSettings:
    """Settings needed to reproduce feature-collection tokenization for one source."""

    source_path: str
    max_length: int
    model_type: str | None = None
    domain: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "FeatureDataSourceSettings":
        return cls(
            source_path=str(payload["source_path"]),
            max_length=int(payload["max_length"]),
            model_type=payload.get("model_type"),
            domain=payload.get("domain"),
        )


def load_val_data(
    val_data: str,
    tokenizer: Any,
    max_length: int,
    domain: str | None = None,
    model_type: str | None = None,
) -> tuple[Any, list[dict] | None]:
    """Load validation data from JSONL (local/hf://) or HF dataset ID.

    Args:
        val_data: Path to data (local JSONL, hf:// URI, or HF dataset ID).
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum sequence length in tokens.
        domain: If provided and the dataset has no per-example domain metadata,
                use this as the domain label for all examples.
        model_type: Optional architecture name. Used to choose JSONL formatting.

    Returns:
        (dataset, examples_meta) where examples_meta is a list of dicts with
        metadata (e.g. 'domain') or None if not available.
    """
    settings = FeatureDataSourceSettings(
        source_path=val_data,
        max_length=max_length,
        model_type=model_type,
        domain=domain,
    )
    return load_val_data_from_settings(settings, tokenizer)


def load_val_data_from_settings(
    settings: FeatureDataSourceSettings,
    tokenizer: Any,
) -> tuple[Any, list[dict] | None]:
    """Load validation data from a reproducible source settings object."""
    # Case 1: JSONL file (local path or hf:// URI)
    if settings.source_path.endswith(".jsonl") or settings.source_path.startswith("hf://"):
        dataset, examples_meta = _load_jsonl(
            settings.source_path,
            tokenizer,
            settings.max_length,
            model_type=settings.model_type,
        )
    else:
        # Case 2: HF dataset ID
        dataset, examples_meta = _load_hf_dataset(
            settings.source_path,
            tokenizer,
            settings.max_length,
        )

    # If a domain override is given and no per-example metadata exists, synthesize it
    if settings.domain is not None and examples_meta is None:
        examples_meta = [{"domain": settings.domain} for _ in range(len(dataset))]

    return dataset, examples_meta


def tokenize_source_row(
    settings: FeatureDataSourceSettings,
    tokenizer: Any,
    row_idx: int,
) -> list[int]:
    """Tokenize one source row with the exact feature-collection data path."""
    dataset, _ = load_val_data_from_settings(settings, tokenizer)
    item = dataset[row_idx]
    tokens = item["input_ids"]
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    return list(tokens)


def _load_jsonl(
    val_data: str,
    tokenizer: Any,
    max_length: int,
    model_type: str | None = None,
) -> tuple[Any, list[dict] | None]:
    """Load JSONL data via OpenThoughtsDataset."""
    from training.dataset import OpenThoughtsDataset

    dataset_format = "tokenizer" if model_type == "gemma2" else "deepseek"
    logger.info(
        f"Loading JSONL validation data as OpenThoughtsDataset "
        f"(format={dataset_format}) from {val_data}..."
    )
    dataset = OpenThoughtsDataset(
        data_path=val_data,
        tokenizer=tokenizer,
        max_length=max_length,
        format=dataset_format,
        truncate=True,
        loss_on_prompt=False,
    )
    return dataset, dataset.examples


def _load_hf_dataset(
    val_data: str,
    tokenizer: Any,
    max_length: int,
) -> tuple[Any, list[dict] | None]:
    """Load an HF dataset, auto-detecting the right column format."""
    from datasets import load_dataset

    hf_dataset = load_dataset(val_data, trust_remote_code=True)

    # Find the validation split
    split = None
    split_name = None
    for candidate in ("val", "validation", "test"):
        if candidate in hf_dataset:
            split = hf_dataset[candidate]
            split_name = candidate
            break

    if split is None:
        available = list(hf_dataset.keys())
        raise ValueError(
            f"No validation split found in '{val_data}'. "
            f"Available splits: {available}"
        )

    logger.info(f"Using split '{split_name}' with {len(split)} examples")
    cols = split.column_names
    examples_meta = _build_source_id_metadata(split)

    # Already tokenized
    if "input_ids" in cols:
        return split, examples_meta

    # Chat conversation column (LMSYS "conversation" or OpenThoughts "conversations")
    conv_col = None
    for candidate in ("conversation", "conversations"):
        if candidate in cols:
            conv_col = candidate
            break

    if conv_col is not None:
        return _load_chat_dataset(val_data, tokenizer, max_length, split_name, conv_col)

    # Generic text column
    text_col = None
    for col in ("text", "content", "prompt"):
        if col in cols:
            text_col = col
            break

    if text_col is None:
        raise ValueError(
            f"Cannot find tokenizable column in '{val_data}'. "
            f"Columns: {cols}. "
            f"Expected 'input_ids', 'conversation', 'conversations', "
            f"'text', 'content', or 'prompt'."
        )

    logger.info(f"Tokenizing text column '{text_col}'")

    def tokenize_fn(examples: dict) -> dict:
        return tokenizer(
            examples[text_col],
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )

    split = split.map(tokenize_fn, batched=True, remove_columns=cols)
    split.set_format("torch")
    return split, examples_meta


def _build_source_id_metadata(split: Any) -> list[dict] | None:
    """Keep stable source IDs before tokenization drops raw dataset columns."""
    id_columns = [col for col in ("conversation_id", "id") if col in split.column_names]
    if not id_columns:
        return None
    column_values = {col: split[col] for col in id_columns}
    metadata: list[dict] = []
    for row_idx in range(len(split)):
        row_meta = {}
        for col, values in column_values.items():
            value = values[row_idx]
            if value is not None:
                row_meta[col] = value
        metadata.append(row_meta)
    return metadata


def _load_chat_dataset(
    val_data: str,
    tokenizer: Any,
    max_length: int,
    split_name: str,
    conv_col: str,
) -> tuple[Any, list[dict] | None]:
    """Load chat conversation data via LMSYSChatDataset."""
    from training.dataset.gemma2.lmsys_chat import LMSYSChatDataset

    logger.info(f"Detected chat column '{conv_col}', loading via LMSYSChatDataset")
    dataset = LMSYSChatDataset(
        data_path=val_data,
        tokenizer=tokenizer,
        max_length=max_length,
        truncate=True,
        loss_on_prompt=True,
        split=split_name,
        conversation_field=conv_col,
    )
    return dataset, None
