"""Tokenizer-driven special token detection and region classification.

Provides a unified interface for detecting chat structure tokens (user/assistant
markers, thinking tags, BOS) across different model families, and classifying
token positions into semantic regions.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import TypedDict
from transformers import PreTrainedTokenizerBase

TokenPattern = tuple[int, ...]
TokenPatterns = tuple[TokenPattern, ...]
TokenSpan = tuple[int, int]
TokenMarkerValue = int | list[int] | list[TokenSpan] | None
TokenMarkers = dict[str, TokenMarkerValue]


class SpecialTokenName(StrEnum):
    BOS = "bos"
    USER_MARKER = "user_marker"
    ASSISTANT_MARKER = "assistant_marker"
    THINK_START = "think_start"
    THINK_END = "think_end"


PatternIndex = dict[int, list[tuple[SpecialTokenName, TokenPattern]]]


_REGION_MARKER_NAMES = (
    SpecialTokenName.BOS,
    SpecialTokenName.USER_MARKER,
    SpecialTokenName.ASSISTANT_MARKER,
    SpecialTokenName.THINK_START,
    SpecialTokenName.THINK_END,
)


class ChatRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class HFChatMessage(TypedDict):
    role: str
    content: str


@dataclass(frozen=True)
class ChatMessage:
    role: ChatRole
    content: str

    def to_hf(self) -> HFChatMessage:
        return {"role": self.role.value, "content": self.content}


@dataclass(frozen=True)
class RegionContext:
    marker_regions: dict[int, str]
    thinking_regions: dict[int, float]
    user_positions: list[int]
    assistant_positions: list[int]


@dataclass
class SpecialTokenIds:
    """Special token patterns for a tokenizer.

    Markers may be single tokens or multi-token patterns from the tokenizer's
    chat template.
    """
    bos: TokenPatterns | None = None
    user_marker: TokenPatterns | None = None
    assistant_marker: TokenPatterns | None = None
    think_start: TokenPatterns | None = None
    think_end: TokenPatterns | None = None

    def items(self) -> tuple[tuple[SpecialTokenName, TokenPatterns | None], ...]:
        return tuple(
            (name, getattr(self, name.value)) for name in _REGION_MARKER_NAMES
        )

    def as_dict(self) -> dict[str, TokenPatterns | None]:
        return {name.value: patterns for name, patterns in self.items()}


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


def _build_pattern_index(special: SpecialTokenIds) -> PatternIndex:
    pattern_index: PatternIndex = {}
    for name, patterns in special.items():
        if patterns is None:
            continue
        for pattern in patterns:
            if not pattern:
                continue
            pattern_index.setdefault(pattern[0], []).append((name, pattern))
    return pattern_index


def _input_ids_from_chat_template_output(output) -> list[int]:
    """Normalize tokenizer.apply_chat_template(tokenize=True) outputs."""
    if isinstance(output, dict) or hasattr(output, "input_ids"):
        output = output["input_ids"]
    if hasattr(output, "tolist"):
        output = output.tolist()
    if output and isinstance(output[0], list):
        if len(output) != 1:
            raise ValueError("Expected a single chat-template sequence.")
        output = output[0]
    if not isinstance(output, list) or not all(isinstance(token, int) for token in output):
        raise TypeError(
            "tokenizer.apply_chat_template(..., tokenize=True) must return token IDs."
        )
    return output


def _apply_chat_template_ids(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[ChatMessage],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("Tokenizer does not provide apply_chat_template().")

    output = tokenizer.apply_chat_template(
        [message.to_hf() for message in messages],
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    return _input_ids_from_chat_template_output(output)


def _find_subsequence(tokens: list[int], pattern: TokenPattern) -> list[int]:
    if not pattern:
        return []
    return [
        i for i in range(len(tokens) - len(pattern) + 1)
        if tuple(tokens[i:i + len(pattern)]) == pattern
    ]


def _strip_leading_pattern(
    tokens: list[int],
    pattern: TokenPattern | None,
) -> list[int]:
    if pattern is None:
        return tokens
    if tuple(tokens[:len(pattern)]) == pattern:
        return tokens[len(pattern):]
    return tokens


def _derive_role_markers_from_chat_template(
    tokenizer: PreTrainedTokenizerBase,
    bos: TokenPatterns | None,
) -> tuple[TokenPatterns, TokenPatterns]:
    """Infer user/assistant markers from the tokenizer chat template."""
    user_content = "TOKEN_MARKER_USER_SENTINEL_314159"
    assistant_content = "TOKEN_MARKER_ASSISTANT_SENTINEL_271828"
    user_content_ids = tuple(tokenizer.encode(user_content, add_special_tokens=False))
    assistant_content_ids = tuple(
        tokenizer.encode(assistant_content, add_special_tokens=False)
    )
    if not user_content_ids:
        raise ValueError("Could not encode user sentinel for chat-template detection.")
    if not assistant_content_ids:
        raise ValueError(
            "Could not encode assistant sentinel for chat-template detection."
        )

    user_messages = [ChatMessage(ChatRole.USER, user_content)]
    user_ids = _apply_chat_template_ids(
        tokenizer,
        user_messages,
        add_generation_prompt=False,
    )
    user_content_positions = _find_subsequence(user_ids, user_content_ids)
    if not user_content_positions:
        raise ValueError(
            "Could not locate user content inside tokenizer chat template output."
        )

    user_prefix = user_ids[:user_content_positions[0]]
    bos_pattern = bos[0] if bos else None
    user_marker = tuple(_strip_leading_pattern(user_prefix, bos_pattern))
    if not user_marker:
        raise ValueError("Could not derive a non-empty user marker from chat template.")

    user_with_generation_ids = _apply_chat_template_ids(
        tokenizer,
        user_messages,
        add_generation_prompt=True,
    )
    assistant_marker: TokenPattern = ()
    if user_with_generation_ids[:len(user_ids)] == user_ids:
        assistant_marker = tuple(user_with_generation_ids[len(user_ids):])
    if not assistant_marker:
        assistant_marker = _derive_assistant_marker_from_message_template(
            tokenizer,
            user_content,
            user_content_ids,
            assistant_content,
            assistant_content_ids,
            user_ids,
            user_content_positions[0],
        )

    return (user_marker,), (assistant_marker,)


def _derive_assistant_marker_from_message_template(
    tokenizer: PreTrainedTokenizerBase,
    user_content: str,
    user_content_ids: TokenPattern,
    assistant_content: str,
    assistant_content_ids: TokenPattern,
    user_ids: list[int],
    user_content_position: int,
) -> TokenPattern:
    """Infer assistant marker from a completed assistant message template."""
    user_suffix = user_ids[user_content_position + len(user_content_ids):]
    full_ids = _apply_chat_template_ids(
        tokenizer,
        [
            ChatMessage(ChatRole.USER, user_content),
            ChatMessage(ChatRole.ASSISTANT, assistant_content),
        ],
        add_generation_prompt=False,
    )
    assistant_content_positions = _find_subsequence(full_ids, assistant_content_ids)
    if not assistant_content_positions:
        raise ValueError(
            "Could not locate assistant content inside tokenizer chat template output."
        )

    assistant_content_position = assistant_content_positions[0]
    user_content_positions = [
        position for position in _find_subsequence(full_ids, user_content_ids)
        if position < assistant_content_position
    ]
    if not user_content_positions:
        raise ValueError(
            "Could not locate user content before assistant content in chat template."
        )

    segment_start = user_content_positions[0] + len(user_content_ids)
    pre_assistant_segment = full_ids[segment_start:assistant_content_position]
    if pre_assistant_segment[:len(user_suffix)] == user_suffix:
        pre_assistant_segment = pre_assistant_segment[len(user_suffix):]

    assistant_marker = tuple(pre_assistant_segment)
    if not assistant_marker:
        raise ValueError(
            "Could not derive a non-empty assistant marker from chat template."
        )
    return assistant_marker


def _special_token_strings(tokenizer: PreTrainedTokenizerBase) -> set[str]:
    """Collect declared special-token strings from tokenizer metadata."""
    tokens: set[str] = set()

    def token_string(value) -> str | None:
        if isinstance(value, str):
            return value
        content = getattr(value, "content", None)
        return content if isinstance(content, str) else None

    for token in getattr(tokenizer, "all_special_tokens", []) or []:
        token_text = token_string(token)
        if token_text is not None:
            tokens.add(token_text)

    def add_from_value(value) -> None:
        token_text = token_string(value)
        if token_text is not None:
            tokens.add(token_text)
        elif isinstance(value, Iterable):
            for item in value:
                token_text = token_string(item)
                if token_text is not None:
                    tokens.add(token_text)

    for value in (getattr(tokenizer, "special_tokens_map", {}) or {}).values():
        add_from_value(value)

    return tokens


def _require_encoded_patterns(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    description: str,
) -> TokenPatterns:
    patterns = _safe_encode_patterns(tokenizer, text)
    if patterns is None:
        raise ValueError(f"Tokenizer declares {description}, but {text!r} is not encodable.")
    return patterns


def _detect_thinking_markers(
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[TokenPatterns | None, TokenPatterns | None]:
    special_tokens = _special_token_strings(tokenizer)

    has_xml_start = "<think>" in special_tokens
    has_xml_end = "</think>" in special_tokens
    if has_xml_start != has_xml_end:
        raise ValueError("Tokenizer declares only one side of the <think> marker pair.")
    if has_xml_start:
        return (
            _require_encoded_patterns(tokenizer, "<think>", "the <think> start marker"),
            _require_encoded_patterns(tokenizer, "</think>", "the </think> end marker"),
        )

    has_channel_start = "<|channel>" in special_tokens
    has_channel_end = "<channel|>" in special_tokens
    if has_channel_start != has_channel_end:
        raise ValueError(
            "Tokenizer declares only one side of the channel thinking marker pair."
        )
    if has_channel_start:
        return (
            _require_encoded_patterns(
                tokenizer,
                "<|channel>thought\n",
                "the thought-channel start marker",
            ),
            _require_encoded_patterns(tokenizer, "<channel|>", "the channel end marker"),
        )

    return None, None


def detect_special_tokens(
    tokenizer: PreTrainedTokenizerBase,
    model_type: str | None = None,
) -> SpecialTokenIds:
    """Detect chat-structure token IDs from tokenizer metadata/templates.

    Args:
        tokenizer: The tokenizer to detect tokens for.
        model_type: Optional architecture name used only for error context.
    """
    bos_id: int | None = tokenizer.bos_token_id  # type: ignore[assignment]
    result = SpecialTokenIds(bos=_single_pattern(bos_id))

    try:
        result.user_marker, result.assistant_marker = (
            _derive_role_markers_from_chat_template(tokenizer, result.bos)
        )
        result.think_start, result.think_end = _detect_thinking_markers(tokenizer)
    except Exception as exc:
        context = f" for model_type={model_type!r}" if model_type else ""
        raise ValueError(f"Could not detect special tokens{context}: {exc}") from exc

    return result


def find_token_positions(tokens: list[int], special: SpecialTokenIds) -> TokenMarkers:
    """Find first positions, all start positions, and spans of special tokens."""
    token_items = special.items()
    pattern_index = _build_pattern_index(special)
    positions: TokenMarkers = {name.value: None for name, _ in token_items}
    all_positions: dict[SpecialTokenName, list[int]] = {
        name: [] for name, _ in token_items
    }
    all_spans: dict[SpecialTokenName, list[TokenSpan]] = {
        name: [] for name, _ in token_items
    }

    for i in range(len(tokens)):
        matches: dict[SpecialTokenName, int] = {}
        for name, pattern in pattern_index.get(tokens[i], []):
            end = i + len(pattern)
            if end <= len(tokens) and tuple(tokens[i:end]) == pattern:
                matches[name] = max(matches.get(name, 0), len(pattern))

        for name, matched_length in matches.items():
            if positions[name.value] is None:
                positions[name.value] = i
            all_positions[name].append(i)
            all_spans[name].append((i, i + matched_length))

    for key, values in all_positions.items():
        positions[f"{key.value}_positions"] = values
    for key, values in all_spans.items():
        positions[f"{key.value}_spans"] = values

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


def _last_before(positions: list[int], position: int) -> int | None:
    index = bisect_left(positions, position)
    return positions[index - 1] if index else None


def _region_context(markers: TokenMarkers) -> RegionContext:
    marker_spans = {
        name: sorted(_marker_spans(markers, name.value))
        for name in _REGION_MARKER_NAMES
    }
    marker_regions: dict[int, str] = {}
    for name in _REGION_MARKER_NAMES:
        for start, end in marker_spans[name]:
            for position in range(start, end):
                marker_regions.setdefault(position, name.value)

    thinking_regions: dict[int, float] = {}
    for think_start_span in marker_spans[SpecialTokenName.THINK_START]:
        think_end_span = next(
            (
                span for span in marker_spans[SpecialTokenName.THINK_END]
                if span[0] > think_start_span[0]
            ),
            None,
        )
        if think_end_span is None:
            continue

        thinking_content_start = think_start_span[1]
        thinking_content_end = think_end_span[0] - 1
        thinking_length = thinking_content_end - thinking_content_start + 1
        for position in range(thinking_content_start, thinking_content_end + 1):
            relative_pos = (
                (position - thinking_content_start) / thinking_length
                if thinking_length > 0
                else 0.5
            )
            thinking_regions.setdefault(position, relative_pos)

    return RegionContext(
        marker_regions=marker_regions,
        thinking_regions=thinking_regions,
        user_positions=sorted(
            _marker_positions(markers, SpecialTokenName.USER_MARKER.value)
        ),
        assistant_positions=sorted(
            _marker_positions(markers, SpecialTokenName.ASSISTANT_MARKER.value)
        ),
    )


def _classify_position_with_context(
    position: int, context: RegionContext
) -> tuple[str, float | None]:
    marker_region = context.marker_regions.get(position)
    if marker_region is not None:
        return marker_region, None

    thinking_position = context.thinking_regions.get(position)
    if thinking_position is not None:
        return 'thinking', thinking_position

    last_user = _last_before(context.user_positions, position)
    last_assistant = _last_before(context.assistant_positions, position)

    if last_assistant is not None and (last_user is None or last_assistant > last_user):
        return 'answer', None

    if last_user is not None and (last_assistant is None or last_user > last_assistant):
        return 'question', None

    return 'unknown', None


def classify_position(position: int, markers: TokenMarkers) -> tuple[str, float | None]:
    """Classify which region a token position belongs to.

    Returns: (region_name, thinking_position_or_none)
        - thinking_position is 0.0-1.0 for tokens in the thinking region, None otherwise.
        - Regions: bos, user_marker, assistant_marker, think_start, think_end,
                   question, thinking, answer, unknown
    """
    return _classify_position_with_context(position, _region_context(markers))


def precompute_regions(
    tokens: list[int], markers: TokenMarkers
) -> tuple[list[str], list[float | None]]:
    """Precompute region classification for all positions in a sequence."""
    context = _region_context(markers)
    regions = []
    thinking_positions = []
    for pos in range(len(tokens)):
        region, think_pos = _classify_position_with_context(pos, context)
        regions.append(region)
        thinking_positions.append(think_pos)
    return regions, thinking_positions
