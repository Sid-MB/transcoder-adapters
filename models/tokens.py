"""Architecture-aware special token detection and region classification.

Provides a unified interface for detecting chat structure tokens (user/assistant
markers, thinking tags, BOS) across different model families, and classifying
token positions into semantic regions.
"""

from __future__ import annotations

from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase


@dataclass
class SpecialTokenIds:
    """Special token IDs for a given tokenizer/architecture."""
    bos: int | None = None
    user_marker: int | None = None
    assistant_marker: int | None = None
    think_start: int | None = None
    think_end: int | None = None

    def as_dict(self) -> dict[str, int | None]:
        return {
            'bos': self.bos,
            'user_marker': self.user_marker,
            'assistant_marker': self.assistant_marker,
            'think_start': self.think_start,
            'think_end': self.think_end,
        }


def _safe_encode_single(tokenizer: PreTrainedTokenizerBase, text: str) -> int | None:
    """Encode a string and return its token ID, or None if it doesn't encode to a single token."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]
    return None


# Architecture-specific token detection strategies
_TOKEN_STRATEGIES: dict[str, list[dict[str, str]]] = {
    "qwen2": [
        {"user_marker": "<｜User｜>", "assistant_marker": "<｜Assistant｜>",
         "think_start": "<think>", "think_end": "</think>"},
    ],
    "gemma2": [
        # Gemma 2 uses <start_of_turn>user / <start_of_turn>model (no thinking tags)
        {"user_marker": "<start_of_turn>user", "assistant_marker": "<start_of_turn>model"},
    ],
}

# Generic fallback strategies tried for any architecture
_GENERIC_STRATEGIES: list[dict[str, str]] = [
    {"user_marker": "<|user|>", "assistant_marker": "<|assistant|>",
     "think_start": "<think>", "think_end": "</think>"},
    {"think_start": "<think>", "think_end": "</think>"},
]


def detect_special_tokens(
    tokenizer: PreTrainedTokenizerBase,
    model_type: str | None = None,
) -> SpecialTokenIds:
    """Detect special token IDs from a tokenizer.

    Tries architecture-specific strategies first, then generic fallbacks.
    Tokens that can't be detected are left as None.

    Args:
        tokenizer: The tokenizer to detect tokens for.
        model_type: Architecture name (e.g. "qwen2", "gemma2"). If None,
                     tries all strategies.
    """
    bos_id: int | None = tokenizer.bos_token_id  # type: ignore[assignment]
    result = SpecialTokenIds(bos=bos_id)

    # Collect strategies to try
    strategies: list[dict[str, str]] = []
    if model_type and model_type in _TOKEN_STRATEGIES:
        strategies.extend(_TOKEN_STRATEGIES[model_type])
    else:
        for strats in _TOKEN_STRATEGIES.values():
            strategies.extend(strats)
    strategies.extend(_GENERIC_STRATEGIES)

    for strategy in strategies:
        found_all = True
        candidates: dict[str, int | None] = {}

        for key, text in strategy.items():
            token_id = _safe_encode_single(tokenizer, text)
            if token_id is None:
                # Try looking in added_tokens / special tokens
                token_id: int | None = tokenizer.convert_tokens_to_ids(text)  # type: ignore[assignment]
                if token_id == tokenizer.unk_token_id:
                    token_id = None
            candidates[key] = token_id
            if token_id is None:
                found_all = False

        if found_all:
            for key, token_id in candidates.items():
                if getattr(result, key) is None:
                    setattr(result, key, token_id)
            break
    else:
        # No strategy matched fully; apply best-effort from all strategies
        for strategy in strategies:
            for key, text in strategy.items():
                if getattr(result, key) is not None:
                    continue
                token_id = _safe_encode_single(tokenizer, text)
                if token_id is None:
                    token_id: int | None = tokenizer.convert_tokens_to_ids(text)  # type: ignore[assignment]
                    if token_id == tokenizer.unk_token_id:
                        token_id = None
                if token_id is not None:
                    setattr(result, key, token_id)

    return result


def find_token_positions(tokens: list[int], special: SpecialTokenIds) -> dict[str, int | None]:
    """Find first positions of special tokens in a sequence."""
    ids = special.as_dict()
    positions: dict[str, int | None] = {k: None for k in ids}

    remaining = {k for k, v in ids.items() if v is not None}

    for i, tok in enumerate(tokens):
        matched = [k for k in remaining if tok == ids[k]]
        for k in matched:
            positions[k] = i
            remaining.discard(k)
        if not remaining:
            break

    return positions


def classify_position(position: int, markers: dict[str, int | None]) -> tuple[str, float | None]:
    """Classify which region a token position belongs to.

    Returns: (region_name, thinking_position_or_none)
        - thinking_position is 0.0-1.0 for tokens in the thinking region, None otherwise.
        - Regions: bos, user_marker, assistant_marker, think_start, think_end,
                   question, thinking, answer, unknown
    """
    # Single-token special markers
    for marker_name in ('bos', 'user_marker', 'assistant_marker', 'think_start', 'think_end'):
        if markers.get(marker_name) is not None and position == markers[marker_name]:
            return marker_name, None

    assistant_pos = markers.get('assistant_marker')
    think_start_pos = markers.get('think_start')
    think_end_pos = markers.get('think_end')

    # Before assistant marker = question
    if assistant_pos is not None and position < assistant_pos:
        return 'question', None

    # Inside thinking tags
    if think_start_pos is not None and think_end_pos is not None:
        if think_start_pos < position < think_end_pos:
            thinking_content_start = think_start_pos + 1
            thinking_content_end = think_end_pos - 1
            thinking_length = thinking_content_end - thinking_content_start + 1
            if thinking_length > 0:
                relative_pos = (position - thinking_content_start) / thinking_length
            else:
                relative_pos = 0.5
            return 'thinking', relative_pos

    # After think_end = answer
    if think_end_pos is not None and position > think_end_pos:
        return 'answer', None

    # Between assistant_marker and think_start (if thinking exists)
    if assistant_pos is not None and position > assistant_pos:
        if think_start_pos is None:
            return 'answer', None
        if position < think_start_pos:
            return 'answer', None

    return 'unknown', None


def precompute_regions(
    tokens: list[int], markers: dict[str, int | None]
) -> tuple[list[str], list[float | None]]:
    """Precompute region classification for all positions in a sequence."""
    regions = []
    thinking_positions = []
    for pos in range(len(tokens)):
        region, think_pos = classify_position(pos, markers)
        regions.append(region)
        thinking_positions.append(think_pos)
    return regions, thinking_positions
