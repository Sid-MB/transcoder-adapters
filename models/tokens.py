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
TokenSpan = tuple[int, int]
TokenMarkerValue = int | list[int] | list[TokenSpan] | None
TokenMarkers = dict[str, TokenMarkerValue]


@dataclass
class SpecialTokenIds:
    """Special token patterns for a given tokenizer/architecture.

    Most markers are single tokens, but Gemma2 role markers are a token
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
        if text.strip().lower() in {"user", "model", "assistant"}:
            continue

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


def _matched_pattern_length(
    tokens: list[int], position: int, patterns: TokenPatterns
) -> int | None:
    """Return the longest marker pattern length matching at ``position``."""
    matched_lengths = []
    for pattern in patterns:
        end = position + len(pattern)
        if end <= len(tokens) and tuple(tokens[position:end]) == pattern:
            matched_lengths.append(len(pattern))
    return max(matched_lengths) if matched_lengths else None


# Architecture-specific token detection strategies
_TOKEN_STRATEGIES: dict[str, list[dict[str, str | list[str]]]] = {
    "qwen2": [
        {"user_marker": "<｜User｜>", "assistant_marker": "<｜Assistant｜>",
         "think_start": "<think>", "think_end": "</think>"},
    ],
    "gemma2": [
        # Gemma2 uses <start_of_turn> plus role text (no thinking tags).
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


def find_token_positions(tokens: list[int], special: SpecialTokenIds) -> TokenMarkers:
    """Find first positions, all start positions, and spans of special tokens."""
    patterns_by_name = special.as_dict()
    positions: TokenMarkers = {k: None for k in patterns_by_name}
    all_positions: dict[str, list[int]] = {k: [] for k in patterns_by_name}
    all_spans: dict[str, list[TokenSpan]] = {k: [] for k in patterns_by_name}

    for i in range(len(tokens)):
        for k, patterns in patterns_by_name.items():
            if patterns is None:
                continue
            matched_length = _matched_pattern_length(tokens, i, patterns)
            if matched_length is None:
                continue
            if positions[k] is None:
                positions[k] = i
            all_positions[k].append(i)
            all_spans[k].append((i, i + matched_length))

    for key, values in all_positions.items():
        positions[f"{key}_positions"] = values
    for key, values in all_spans.items():
        positions[f"{key}_spans"] = values

    return positions


def _marker_positions(markers: TokenMarkers, name: str) -> list[int]:
    positions = markers.get(f"{name}_positions")
    if isinstance(positions, list):
        return [p for p in positions if isinstance(p, int)]
    first = markers.get(name)
    return [first] if isinstance(first, int) else []


def _marker_spans(markers: TokenMarkers, name: str) -> list[TokenSpan]:
    spans = markers.get(f"{name}_spans")
    if isinstance(spans, list):
        return [
            span for span in spans
            if (
                isinstance(span, tuple)
                and len(span) == 2
                and isinstance(span[0], int)
                and isinstance(span[1], int)
            )
        ]
    return [(position, position + 1) for position in _marker_positions(markers, name)]


def _in_marker_span(position: int, spans: list[TokenSpan]) -> bool:
    return any(start <= position < end for start, end in spans)


def _last_before(positions: list[int], position: int) -> int | None:
    prior = [p for p in positions if p < position]
    return max(prior) if prior else None


def classify_position(position: int, markers: TokenMarkers) -> tuple[str, float | None]:
    """Classify which region a token position belongs to.

    Returns: (region_name, thinking_position_or_none)
        - thinking_position is 0.0-1.0 for tokens in the thinking region, None otherwise.
        - Regions: bos, user_marker, assistant_marker, think_start, think_end,
                   question, thinking, answer, unknown
    """
    # Special marker spans. Gemma2 role markers span multiple tokens, e.g.
    # <start_of_turn>, role text, and sometimes the following newline.
    for marker_name in ('bos', 'user_marker', 'assistant_marker', 'think_start', 'think_end'):
        if _in_marker_span(position, _marker_spans(markers, marker_name)):
            return marker_name, None

    user_positions = _marker_positions(markers, 'user_marker')
    assistant_positions = _marker_positions(markers, 'assistant_marker')
    think_start_spans = _marker_spans(markers, 'think_start')
    think_end_spans = _marker_spans(markers, 'think_end')

    # Inside thinking tags. Pair each start with the first following end.
    for think_start_span in think_start_spans:
        think_end_span = next(
            (span for span in think_end_spans if span[0] > think_start_span[0]),
            None,
        )
        if think_end_span is not None and think_start_span[1] <= position < think_end_span[0]:
            thinking_content_start = think_start_span[1]
            thinking_content_end = think_end_span[0] - 1
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
    tokens: list[int], markers: TokenMarkers
) -> tuple[list[str], list[float | None]]:
    """Precompute region classification for all positions in a sequence."""
    regions = []
    thinking_positions = []
    for pos in range(len(tokens)):
        region, think_pos = classify_position(pos, markers)
        regions.append(region)
        thinking_positions.append(think_pos)
    return regions, thinking_positions
