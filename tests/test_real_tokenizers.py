import os
import unittest

from transformers import AutoTokenizer

from models.tokens import (
    SpecialTokenName,
    detect_special_tokens,
)


REAL_TOKENIZER_CASES = (
    ("qwen2", "Qwen/Qwen2.5-7B-Instruct"),
    ("gemma2", "google/gemma-2-2b-it"),
    ("gemma4_text", "google/gemma-4-E2B-it"),
)


def find_subsequence(tokens, pattern):
    return any(
        tuple(tokens[position:position + len(pattern)]) == pattern
        for position in range(len(tokens) - len(pattern) + 1)
    )


def input_ids(output) -> list[int]:
    if isinstance(output, dict) or hasattr(output, "input_ids"):
        output = output["input_ids"]
    if hasattr(output, "tolist"):
        output = output.tolist()
    if output and isinstance(output[0], list):
        output = output[0]
    return output


class RealTokenizerSmokeTests(unittest.TestCase):
    def load_tokenizer_or_skip(self, model_id: str):
        local_only = os.environ.get("HF_TOKENIZER_TEST_ONLINE") != "1"
        try:
            return AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                local_files_only=local_only,
            )
        except Exception as exc:
            mode = "local cache" if local_only else "Hugging Face"
            self.skipTest(f"{model_id} tokenizer unavailable from {mode}: {exc}")

    def test_detected_patterns_appear_in_real_tokenizer_outputs(self):
        for model_type, model_id in REAL_TOKENIZER_CASES:
            with self.subTest(model_id=model_id):
                tokenizer = self.load_tokenizer_or_skip(model_id)
                special = detect_special_tokens(tokenizer, model_type=model_type)

                user_chat = input_ids(
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": "TOKEN_MARKER_USER_SENTINEL_314159"}],
                        tokenize=True,
                        add_generation_prompt=False,
                    )
                )
                generation_chat = input_ids(
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": "TOKEN_MARKER_USER_SENTINEL_314159"}],
                        tokenize=True,
                        add_generation_prompt=True,
                    )
                )
                thinking_sources = []
                if special.think_start is not None or special.think_end is not None:
                    for text in (
                        "<think>reasoning</think>",
                        "<|channel>thought\nreasoning<channel|>",
                    ):
                        thinking_sources.append(
                            tokenizer.encode(text, add_special_tokens=False)
                        )

                sources = {
                    SpecialTokenName.BOS: [[tokenizer.bos_token_id]],
                    SpecialTokenName.USER_MARKER: [user_chat],
                    SpecialTokenName.ASSISTANT_MARKER: [generation_chat],
                    SpecialTokenName.THINK_START: thinking_sources,
                    SpecialTokenName.THINK_END: thinking_sources,
                }
                for name, patterns in special.items():
                    if patterns is None:
                        continue
                    self.assertTrue(
                        sources[name],
                        f"{model_id} has detected {name.value} but no source tokens",
                    )
                    for pattern in patterns:
                        self.assertTrue(
                            any(find_subsequence(source, pattern) for source in sources[name]),
                            f"{model_id} {name.value} pattern {pattern} not found",
                        )


if __name__ == "__main__":
    unittest.main()
