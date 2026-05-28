import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from analysis.attribution import token_probability_delta as tpd
from analysis.attribution.run_attribution import (
    DEEPSEEK_ASSISTANT_TOKEN,
    DEEPSEEK_BOS_TOKEN,
    DEEPSEEK_USER_TOKEN,
)


class TinyTokenizer:
    bos_token_id = 0

    def __init__(self):
        self.vocab = {
            "<bos>": 0,
            "<user>": 1,
            "<model>": 2,
            "A": 3,
            "B": 4,
            "!": 5,
        }
        self.inv_vocab = {value: key for key, value in self.vocab.items()}

    def encode(self, text, add_special_tokens=False):
        ids = [self.vocab[char] for char in text]
        if add_special_tokens:
            return [self.bos_token_id] + ids
        return ids

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(self.inv_vocab[int(token_id)] for token_id in token_ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        ids = [self.vocab["<bos>"]]
        for message in messages:
            if message["role"] != "user":
                raise ValueError(message["role"])
            ids.append(self.vocab["<user>"])
            ids.extend(self.encode(message["content"], add_special_tokens=False))
        if add_generation_prompt:
            ids.append(self.vocab["<model>"])
        return ids if tokenize else self.decode(ids)


class FakeMlp:
    disable_transcoder = False


class FakeOutput:
    def __init__(self, logits):
        self.logits = logits


class FakeModel:
    device = torch.device("cpu")

    def __init__(self):
        self.mlp = FakeMlp()

    def _transcoder_mlps(self):
        return [self.mlp]

    def __call__(self, input_ids):
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, 6)
        logits[:, :, 4] = 1.0
        if not self.mlp.disable_transcoder:
            logits[:, :, 4] = 3.0
        return FakeOutput(logits)


class TokenProbabilityDeltaTests(unittest.TestCase):
    def test_chat_prompt_marks_assistant_tokens(self):
        tokenizer = TinyTokenizer()
        prompt = (
            f"{DEEPSEEK_BOS_TOKEN}{DEEPSEEK_USER_TOKEN}AB"
            f"{DEEPSEEK_ASSISTANT_TOKEN}B!"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prompt.txt"
            path.write_text(prompt)

            token_ids, assistant_mask, decoded = tpd._parse_prompt_file(
                path,
                "chat",
                tokenizer,
            )

        self.assertEqual(decoded, "<bos><user>AB<model>B!")
        self.assertEqual(token_ids, [0, 1, 3, 4, 2, 4, 5])
        self.assertEqual(assistant_mask, [False, False, False, False, False, True, True])

    def test_compute_token_deltas_toggles_transcoder(self):
        tokenizer = TinyTokenizer()
        model = FakeModel()

        deltas = tpd.compute_token_deltas(
            model,
            tokenizer,
            token_ids=[0, 1, 4],
            assistant_mask=[False, False, True],
        )

        self.assertFalse(model.mlp.disable_transcoder)
        self.assertEqual(deltas[-1].text, "B")
        self.assertGreater(deltas[-1].delta_logprob, 0)
        self.assertGreater(deltas[-1].delta_prob, 0)
        self.assertTrue(deltas[-1].in_assistant)

    def test_dashboard_cache_dir_is_deterministic_and_content_sensitive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_dir = Path(tmpdir) / "prompts"
            prompts_dir.mkdir()
            prompt_path = prompts_dir / "example.txt"
            prompt_path.write_text("A")

            first = tpd.default_dashboard_cache_dir(
                model_path="org/model",
                prompts_dir=prompts_dir,
                prompt_path=prompt_path,
                prompt_format="chat",
            )
            second = tpd.default_dashboard_cache_dir(
                model_path="org/model",
                prompts_dir=prompts_dir,
                prompt_path=prompt_path,
                prompt_format="chat",
            )
            all_tokens = tpd.default_dashboard_cache_dir(
                model_path="org/model",
                prompts_dir=prompts_dir,
                prompt_path=prompt_path,
                prompt_format="chat",
                all_tokens=True,
            )

            prompt_path.write_text("B")
            changed = tpd.default_dashboard_cache_dir(
                model_path="org/model",
                prompts_dir=prompts_dir,
                prompt_path=prompt_path,
                prompt_format="chat",
            )

        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)
        self.assertNotEqual(first, all_tokens)

    def test_dashboard_cache_dir_ignores_terminal_newlines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_dir = Path(tmpdir) / "prompts"
            prompts_dir.mkdir()
            prompt_path = prompts_dir / "example.txt"
            prompt_path.write_text("A")

            without_newline = tpd.default_dashboard_cache_dir(
                model_path="org/model",
                prompts_dir=prompts_dir,
                prompt_path=prompt_path,
                prompt_format="chat",
            )
            prompt_path.write_text("A\n\n")
            with_newline = tpd.default_dashboard_cache_dir(
                model_path="org/model",
                prompts_dir=prompts_dir,
                prompt_path=prompt_path,
                prompt_format="chat",
            )

        self.assertEqual(without_newline, with_newline)

    def test_dashboard_cache_cleanup_removes_stale_same_prompt_hash_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prompts_dir = root / "prompts"
            prompts_dir.mkdir()
            prompt_path = prompts_dir / "example.txt"
            prompt_path.write_text("A")
            with patch.object(tpd, "PRODUCTS_DIR", root / "products"):
                current = tpd.default_dashboard_cache_dir(
                    model_path="org/model",
                    prompts_dir=prompts_dir,
                    prompt_path=prompt_path,
                    prompt_format="chat",
                )
                stale = current.parent / "example_000000000000_assistant_tokens"
                stale.mkdir(parents=True)
                current.mkdir(parents=True)

                removed = tpd._cleanup_stale_dashboard_caches(
                    model_path="org/model",
                    prompts_dir=prompts_dir,
                    prompt_path=prompt_path,
                    prompt_format="chat",
                    all_tokens=False,
                    current_output_dir=current,
                )

                self.assertEqual(removed, [stale])
                self.assertFalse(stale.exists())
                self.assertTrue(current.exists())


if __name__ == "__main__":
    unittest.main()
