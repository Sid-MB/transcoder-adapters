"""Architecture-aware special token detection and region classification.

Provides a unified interface for detecting chat structure tokens (user/assistant
markers, thinking tags, BOS) across different model families, and classifying
token positions into semantic regions.
"""

from __future__ import annotations

from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase

TokenPattern = tuple[int, ...]
TokenPatterns = tuple[TokenPattern, ...]


@dataclass
class SpecialTokenIds:
    """Special token patterns for a given tokenizer/architecture.

    Most markers are single tokens, but Gemma role markers are a token
    sequence: ``<start_of_turn>`` followed by the role name.
    """
    bos: TokenPatterns | None = None
    user_marker: TokenPatterns | None = None
    assistant_marker: TokenPatterns | None = None
    think_start: TokenPatterns | None = None
    think_end: TokenPatterns | None = None

    def as_dict(self) -> dict[str, TokenPatterns | None]:
        return {
            'bos': self.bos,
            'user_marker': self.user_marker,
            'assistant_marker': self.assistant_marker,
            'think_start': self.think_start,
            'think_end': self.think_end,
        }


def _as_patterns(patterns: list[TokenPattern]) -> TokenPatterns | None:
    """Normalize and deduplicate token patterns while preserving order."""
    out = []
    seen = set()
    for pattern in patterns:
        if not pattern or pattern in seen:
            continue
        out.append(pattern)
        seen.add(pattern)
    return tuple(out) if out else None


def _safe_encode_patterns(
    tokenizer: PreTrainedTokenizerBase,
    text_or_texts: str | list[str],
) -> TokenPatterns | None:
    """Encode marker text into one or more token patterns."""
    texts = [text_or_texts] if isinstance(text_or_texts, str) else text_or_texts
    patterns: list[TokenPattern] = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids and not (len(ids) == 1 and ids[0] == tokenizer.unk_token_id):
            patterns.append(tuple(ids))

        # Try looking in added_tokens / special tokens. This is mostly useful
        # for single-token chat markers such as <think>.
        token_id: int | None = tokenizer.convert_tokens_to_ids(text)  # type: ignore[assignment]
        if token_id is not None and token_id != tokenizer.unk_token_id:
            patterns.append((token_id,))
    return _as_patterns(patterns)


def _single_pattern(token_id: int | None) -> TokenPatterns | None:
    if token_id is None:
        return None
    return ((token_id,),)


def _patterns_match(tokens: list[int], position: int, patterns: TokenPatterns) -> bool:
    """Return True if any marker pattern starts at ``position``."""
    for pattern in patterns:
        end = position + len(pattern)
        if end <= len(tokens) and tuple(tokens[position:end]) == pattern:
            return True
    return False


# Architecture-specific token detection strategies
_TOKEN_STRATEGIES: dict[str, list[dict[str, str | list[str]]]] = {
    "qwen2": [
        {"user_marker": "<｜User｜>", "assistant_marker": "<｜Assistant｜>",
         "think_start": "<think>", "think_end": "</think>"},
    ],
    "gemma2": [
        # Gemma 2 uses <start_of_turn> plus role text (no thinking tags).
        # Keep newline/no-newline variants because chat templates include a
        # newline after the role, and tokenizers may bind it to the role token.
        {
            "user_marker": ["<start_of_turn>user\n", "<start_of_turn>user"],
            "assistant_marker": ["<start_of_turn>model\n", "<start_of_turn>model"],
        },
    ],
}

# Generic fallback strategies tried for any architecture
_GENERIC_STRATEGIES: list[dict[str, str | list[str]]] = [
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
    result = SpecialTokenIds(bos=_single_pattern(bos_id))

    # Collect strategies to try
    strategies: list[dict[str, str | list[str]]] = []
    if model_type and model_type in _TOKEN_STRATEGIES:
        strategies.extend(_TOKEN_STRATEGIES[model_type])
    else:
        for strats in _TOKEN_STRATEGIES.values():
            strategies.extend(strats)
    strategies.extend(_GENERIC_STRATEGIES)

    for strategy in strategies:
        found_all = True
        candidates: dict[str, TokenPatterns | None] = {}

        for key, text in strategy.items():
            patterns = _safe_encode_patterns(tokenizer, text)
            candidates[key] = patterns
            if patterns is None:
                found_all = False

        if found_all:
            for key, patterns in candidates.items():
                if getattr(result, key) is None:
                    setattr(result, key, patterns)
            break
    else:
        # No strategy matched fully; apply best-effort from all strategies
        for strategy in strategies:
            for key, text in strategy.items():
                if getattr(result, key) is not None:
                    continue
                patterns = _safe_encode_patterns(tokenizer, text)
                if patterns is not None:
                    setattr(result, key, patterns)

    return result


def find_token_positions(tokens: list[int], special: SpecialTokenIds) -> dict[str, int | None]:
    """Find first positions of special tokens in a sequence."""
    patterns_by_name = special.as_dict()
    positions: dict[str, int | None] = {k: None for k in patterns_by_name}
    all_positions: dict[str, list[int]] = {k: [] for k in patterns_by_name}

    for i in range(len(tokens)):
        matched = [
            k for k, patterns in patterns_by_name.items()
            if patterns_by_name[k] is not None and _patterns_match(tokens, i, patterns_by_name[k])
        ]
        for k in matched:
            if positions[k] is None:
                positions[k] = i
            all_positions[k].append(i)

    for key, values in all_positions.items():
        positions[f"{key}_positions"] = values  # type: ignore[assignment]

    return positions


def _marker_positions(markers: dict[str, int | None], name: str) -> list[int]:
    positions = markers.get(f"{name}_positions")
    if isinstance(positions, list):
        return positions
    first = markers.get(name)
    return [first] if isinstance(first, int) else []


def _last_before(positions: list[int], position: int) -> int | None:
    prior = [p for p in positions if p < position]
    return max(prior) if prior else None


def classify_position(position: int, markers: dict[str, int | None]) -> tuple[str, float | None]:
    """Classify which region a token position belongs to.

    Returns: (region_name, thinking_position_or_none)
        - thinking_position is 0.0-1.0 for tokens in the thinking region, None otherwise.
        - Regions: bos, user_marker, assistant_marker, think_start, think_end,
                   question, thinking, answer, unknown
    """
    # Single-token special markers
    for marker_name in ('bos', 'user_marker', 'assistant_marker', 'think_start', 'think_end'):
        if position in _marker_positions(markers, marker_name):
            return marker_name, None

    user_positions = _marker_positions(markers, 'user_marker')
    assistant_positions = _marker_positions(markers, 'assistant_marker')
    think_start_positions = _marker_positions(markers, 'think_start')
    think_end_positions = _marker_positions(markers, 'think_end')

    # Inside thinking tags. Pair each start with the first following end.
    for think_start_pos in think_start_positions:
        think_end_pos = next((p for p in think_end_positions if p > think_start_pos), None)
        if think_end_pos is not None and think_start_pos < position < think_end_pos:
            thinking_content_start = think_start_pos + 1
            thinking_content_end = think_end_pos - 1
            thinking_length = thinking_content_end - thinking_content_start + 1
            if thinking_length > 0:
                relative_pos = (position - thinking_content_start) / thinking_length
            else:
                relative_pos = 0.5
            return 'thinking', relative_pos

    last_user = _last_before(user_positions, position)
    last_assistant = _last_before(assistant_positions, position)

    # Classify by the most recent chat role marker, which handles multi-turn
    # conversations while preserving the old single-turn behavior.
    if last_assistant is not None and (last_user is None or last_assistant > last_user):
        return 'answer', None

    if last_user is not None and (last_assistant is None or last_user > last_assistant):
        return 'question', None

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
