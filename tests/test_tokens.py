import unittest
from dataclasses import dataclass
from enum import StrEnum

from models.tokens import (
    SpecialTokenIds,
    SpecialTokenName,
    detect_special_tokens,
    find_token_positions,
    precompute_regions,
)


class ThinkingFamily(StrEnum):
    NONE = "none"
    XML = "xml"
    CHANNEL = "channel"


@dataclass(frozen=True)
class TokenizerFixture:
    model_type: str
    user_marker: str
    assistant_marker: str
    turn_end: str
    thinking_family: ThinkingFamily
    extra_special_tokens: tuple[str, ...] = ()


@dataclass(frozen=True)
class FakeAddedToken:
    content: str


MODEL_FIXTURES = (
    TokenizerFixture(
        model_type="qwen2",
        user_marker="<｜User｜>",
        assistant_marker="<｜Assistant｜>",
        turn_end="<｜end▁of▁sentence｜>",
        thinking_family=ThinkingFamily.XML,
        extra_special_tokens=("<think>", "</think>"),
    ),
    TokenizerFixture(
        model_type="gemma2",
        user_marker="<start_of_turn>user\n",
        assistant_marker="<start_of_turn>model\n",
        turn_end="<end_of_turn>\n",
        thinking_family=ThinkingFamily.NONE,
        extra_special_tokens=("<start_of_turn>", "<end_of_turn>"),
    ),
    TokenizerFixture(
        model_type="gemma4",
        user_marker="<|turn>user\n",
        assistant_marker="<|turn>model\n",
        turn_end="<|end_of_turn|>\n",
        thinking_family=ThinkingFamily.CHANNEL,
        extra_special_tokens=("<|turn>", "<|channel>", "<channel|>", "<|end_of_turn|>"),
    ),
    TokenizerFixture(
        model_type="gemma4_text",
        user_marker="<|turn>user\n",
        assistant_marker="<|turn>model\n",
        turn_end="<|end_of_turn|>\n",
        thinking_family=ThinkingFamily.CHANNEL,
        extra_special_tokens=("<|turn>", "<|channel>", "<channel|>", "<|end_of_turn|>"),
    ),
)


class TemplateTokenizer:
    bos_token_id = 2
    eos_token_id = 1
    pad_token_id = 0
    unk_token_id = -1

    def __init__(self, fixture: TokenizerFixture):
        self.fixture = fixture
        self.special_token_to_id = {
            "<bos>": self.bos_token_id,
            "<eos>": self.eos_token_id,
            "<pad>": self.pad_token_id,
        }
        for token in (
            fixture.user_marker,
            fixture.assistant_marker,
            fixture.turn_end,
            *fixture.extra_special_tokens,
        ):
            if token not in self.special_token_to_id:
                self.special_token_to_id[token] = 100 + len(self.special_token_to_id)

        self.all_special_tokens = list(self.special_token_to_id)
        self.special_tokens_map = {
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "pad_token": "<pad>",
            "additional_special_tokens": list(self.special_token_to_id)[3:],
        }

    def _char_id(self, char: str) -> int:
        return 1000 + ord(char)

    def encode(self, text, add_special_tokens=False):
        ids = [self.bos_token_id] if add_special_tokens else []
        special_tokens = sorted(self.special_token_to_id, key=len, reverse=True)
        position = 0
        while position < len(text):
            matched = next(
                (
                    token for token in special_tokens
                    if text.startswith(token, position)
                ),
                None,
            )
            if matched is not None:
                ids.append(self.special_token_to_id[matched])
                position += len(matched)
            else:
                ids.append(self._char_id(text[position]))
                position += 1
        return ids

    def convert_tokens_to_ids(self, text):
        return self.special_token_to_id.get(text, self.unk_token_id)

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
    ):
        ids = [self.bos_token_id]
        for message in messages:
            role = message["role"]
            if role == "user":
                marker = self.fixture.user_marker
            elif role in ("assistant", "model"):
                marker = self.fixture.assistant_marker
            else:
                raise ValueError(f"Unsupported role: {role}")
            ids.extend(self.encode(marker, add_special_tokens=False))
            ids.extend(self.encode(message["content"], add_special_tokens=False))
            ids.extend(self.encode(self.fixture.turn_end, add_special_tokens=False))

        if add_generation_prompt:
            ids.extend(self.encode(self.fixture.assistant_marker, add_special_tokens=False))

        return ids if tokenize else ""


class AddedTokenMetadataTokenizer(TemplateTokenizer):
    def __init__(self, fixture: TokenizerFixture):
        super().__init__(fixture)
        self.all_special_tokens = [
            FakeAddedToken(token) for token in self.special_token_to_id
        ]
        self.special_tokens_map = {
            "bos_token": FakeAddedToken("<bos>"),
            "eos_token": FakeAddedToken("<eos>"),
            "pad_token": FakeAddedToken("<pad>"),
            "additional_special_tokens": [
                FakeAddedToken(token)
                for token in list(self.special_token_to_id)[3:]
            ],
        }


class NonPrefixGenerationTokenizer(TemplateTokenizer):
    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
    ):
        if not add_generation_prompt:
            return super().apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
            )

        ids = [self.bos_token_id]
        for message in messages:
            if message["role"] != "user":
                raise ValueError("This test tokenizer only supports user prompts.")
            ids.extend(self.encode(self.fixture.user_marker, add_special_tokens=False))
            ids.extend(self.encode(message["content"], add_special_tokens=False))

        ids.extend(self.encode(self.fixture.assistant_marker, add_special_tokens=False))
        return ids if tokenize else ""


def find_subsequence(tokens, pattern):
    return any(
        tuple(tokens[position:position + len(pattern)]) == pattern
        for position in range(len(tokens) - len(pattern) + 1)
    )


def thinking_probe(family: ThinkingFamily) -> str | None:
    if family == ThinkingFamily.XML:
        return "<think>reasoning</think>"
    if family == ThinkingFamily.CHANNEL:
        return "<|channel>thought\nreasoning<channel|>"
    return None


class TokenRegionTests(unittest.TestCase):
    def test_detected_special_tokens_match_tokenizer_sources_for_all_models(self):
        for fixture in MODEL_FIXTURES:
            tokenizer = TemplateTokenizer(fixture)
            with self.subTest(model_type=fixture.model_type):
                special = detect_special_tokens(tokenizer, model_type=fixture.model_type)
                user_chat = tokenizer.apply_chat_template(
                    [{"role": "user", "content": "TOKEN_MARKER_USER_SENTINEL_314159"}],
                    tokenize=True,
                    add_generation_prompt=False,
                )
                generation_chat = tokenizer.apply_chat_template(
                    [{"role": "user", "content": "TOKEN_MARKER_USER_SENTINEL_314159"}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
                thinking_text = thinking_probe(fixture.thinking_family)
                thinking_ids = (
                    tokenizer.encode(thinking_text, add_special_tokens=False)
                    if thinking_text is not None
                    else []
                )

                sources = {
                    SpecialTokenName.BOS: [[tokenizer.bos_token_id]],
                    SpecialTokenName.USER_MARKER: [user_chat],
                    SpecialTokenName.ASSISTANT_MARKER: [generation_chat],
                    SpecialTokenName.THINK_START: [thinking_ids],
                    SpecialTokenName.THINK_END: [thinking_ids],
                }
                for name, patterns in special.items():
                    if patterns is None:
                        continue
                    for pattern in patterns:
                        self.assertTrue(
                            any(find_subsequence(source, pattern) for source in sources[name]),
                            f"{fixture.model_type} {name.value} pattern {pattern} not found",
                        )

                has_thinking = fixture.thinking_family != ThinkingFamily.NONE
                self.assertEqual(special.think_start is not None, has_thinking)
                self.assertEqual(special.think_end is not None, has_thinking)

    def test_detect_special_tokens_raises_without_chat_template(self):
        class NoChatTemplateTokenizer:
            bos_token_id = 2
            unk_token_id = 0
            all_special_tokens = []
            special_tokens_map = {}

            def encode(self, text, add_special_tokens=False):
                del add_special_tokens
                return [ord(char) for char in text]

            def convert_tokens_to_ids(self, text):
                del text
                return self.unk_token_id

        with self.assertRaisesRegex(ValueError, "apply_chat_template"):
            detect_special_tokens(NoChatTemplateTokenizer(), model_type="unknown")

    def test_detect_thinking_markers_from_added_token_objects(self):
        fixture = TokenizerFixture(
            model_type="added-token-metadata",
            user_marker="<|user|>",
            assistant_marker="<|assistant|>",
            turn_end="<|end|>",
            thinking_family=ThinkingFamily.XML,
            extra_special_tokens=("<think>", "</think>"),
        )
        special = detect_special_tokens(
            AddedTokenMetadataTokenizer(fixture),
            model_type=fixture.model_type,
        )

        self.assertIsNotNone(special.think_start)
        self.assertIsNotNone(special.think_end)

    def test_detect_assistant_marker_when_generation_template_is_not_prefix(self):
        fixture = TokenizerFixture(
            model_type="non-prefix-generation",
            user_marker="<|user|>",
            assistant_marker="<|assistant|>",
            turn_end="<|end|>",
            thinking_family=ThinkingFamily.NONE,
            extra_special_tokens=("<|user|>", "<|assistant|>", "<|end|>"),
        )
        tokenizer = NonPrefixGenerationTokenizer(fixture)
        special = detect_special_tokens(tokenizer, model_type=fixture.model_type)

        expected = tuple(tokenizer.encode(fixture.assistant_marker, add_special_tokens=False))
        self.assertEqual(special.assistant_marker, (expected,))

    def test_detect_special_tokens_raises_for_incomplete_thinking_pair(self):
        fixture = TokenizerFixture(
            model_type="incomplete-thinking",
            user_marker="<|user|>",
            assistant_marker="<|assistant|>",
            turn_end="<|end|>",
            thinking_family=ThinkingFamily.NONE,
            extra_special_tokens=("<think>",),
        )

        with self.assertRaisesRegex(ValueError, "only one side"):
            detect_special_tokens(TemplateTokenizer(fixture), model_type=fixture.model_type)

    def test_gemma4_thought_channel_regions(self):
        special = SpecialTokenIds(
            bos=((2,),),
            user_marker=((200, 1001, 108),),
            assistant_marker=((200, 1002, 108),),
            think_start=((201, 1003, 108),),
            think_end=((202,),),
        )
        tokens = [
            2,
            200, 1001, 108,
            501,
            200, 1002, 108,
            201, 1003, 108,
            601, 602,
            202,
            701,
        ]

        markers = find_token_positions(tokens, special)
        regions, thinking_positions = precompute_regions(tokens, markers)

        self.assertEqual(markers["think_start_spans"], [(8, 11)])
        self.assertEqual(markers["think_end_spans"], [(13, 14)])
        self.assertEqual(regions[8:11], ["think_start"] * 3)
        self.assertEqual(regions[11:13], ["thinking"] * 2)
        self.assertEqual(thinking_positions[11], 0.0)
        self.assertEqual(thinking_positions[12], 0.5)
        self.assertEqual(regions[13], "think_end")
        self.assertEqual(regions[14], "answer")

    def test_gemma_marker_spans_and_multiturn_regions(self):
        special = SpecialTokenIds(
            bos=((2,),),
            user_marker=((106, 1645, 108),),
            assistant_marker=((106, 2516, 108),),
        )
        tokens = [
            2,
            106, 1645, 108,
            501, 502,
            106, 2516, 108,
            601, 602,
            106, 1645, 108,
            701,
            106, 2516, 108,
            801,
        ]

        markers = find_token_positions(tokens, special)
        regions, _ = precompute_regions(tokens, markers)

        self.assertEqual(markers["user_marker_positions"], [1, 11])
        self.assertEqual(markers["assistant_marker_positions"], [6, 15])
        self.assertEqual(markers["user_marker_spans"], [(1, 4), (11, 14)])
        self.assertEqual(markers["assistant_marker_spans"], [(6, 9), (15, 18)])

        self.assertEqual(regions[0], "bos")
        self.assertEqual(regions[1:4], ["user_marker"] * 3)
        self.assertEqual(regions[4:6], ["question"] * 2)
        self.assertEqual(regions[6:9], ["assistant_marker"] * 3)
        self.assertEqual(regions[9:11], ["answer"] * 2)
        self.assertEqual(regions[11:14], ["user_marker"] * 3)
        self.assertEqual(regions[14], "question")
        self.assertEqual(regions[15:18], ["assistant_marker"] * 3)
        self.assertEqual(regions[18], "answer")


if __name__ == "__main__":
    unittest.main()
