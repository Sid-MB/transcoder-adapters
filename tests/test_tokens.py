import unittest

import models.tokens as token_module
from models.tokens import (
    SpecialTokenIds,
    detect_special_tokens,
    find_token_positions,
    precompute_regions,
)


class FakeTokenizer:
    bos_token_id = 2
    unk_token_id = 0

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return {
            "user": [11],
            "model": [12],
            "assistant": [13],
        }.get(text, [self.unk_token_id])

    def convert_tokens_to_ids(self, text):
        return {
            "user": 11,
            "model": 12,
            "assistant": 13,
        }.get(text, self.unk_token_id)


class FakeGemmaTokenizer:
    bos_token_id = 2
    unk_token_id = 0

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return {
            "<start_of_turn>user\n": [106, 1645, 108],
            "<start_of_turn>user": [106, 1645],
            "<start_of_turn>model\n": [106, 2516, 108],
            "<start_of_turn>model": [106, 2516],
        }.get(text, [self.unk_token_id])

    def convert_tokens_to_ids(self, text):
        del text
        return self.unk_token_id


class TokenRegionTests(unittest.TestCase):
    def test_gemma4_uses_gemma_chat_marker_strategy(self):
        special = detect_special_tokens(FakeGemmaTokenizer(), model_type="gemma4")

        self.assertEqual(special.user_marker, ((106, 1645, 108), (106, 1645)))
        self.assertEqual(special.assistant_marker, ((106, 2516, 108), (106, 2516)))

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

    def test_role_words_alone_are_not_detected_as_markers(self):
        tokenizer = FakeTokenizer()
        bare_role_strategy = {
            "user_marker": "user",
            "assistant_marker": "model",
        }

        token_module._GENERIC_STRATEGIES.append(bare_role_strategy)
        try:
            special = detect_special_tokens(tokenizer, model_type="unknown")
        finally:
            token_module._GENERIC_STRATEGIES.pop()

        self.assertIsNone(special.user_marker)
        self.assertIsNone(special.assistant_marker)


if __name__ == "__main__":
    unittest.main()
