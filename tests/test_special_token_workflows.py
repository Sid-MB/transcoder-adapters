import json
import tempfile
import unittest
from pathlib import Path

from analysis.attribution.run_attribution import (
    DEEPSEEK_ASSISTANT_TOKEN,
    DEEPSEEK_BOS_TOKEN,
    DEEPSEEK_USER_TOKEN,
    _build_auto_shard_worker_command,
    _parse_visible_cuda_devices,
    build_parser,
    load_prompt_file,
    load_prompts,
)
from analysis.evals.compute_token_metrics_onpolicy import tokenize_chat_prompt_response
from training.dataset.gemma2.lmsys_chat import LMSYSChatDataset
from training.dataset.openthoughts.open_thoughts import OpenThoughtsDataset


class GemmaLikeChatTokenizer:
    bos_token_id = 2
    eos_token_id = 1
    pad_token_id = 0

    user_marker = [106, 1645, 108]
    assistant_marker = [106, 2516, 108]
    end_of_turn = [107]

    def _char_id(self, char: str) -> int:
        return 1000 + ord(char)

    def _char(self, token_id: int) -> str:
        return chr(token_id - 1000)

    def encode(self, text, add_special_tokens=False):
        ids = [self._char_id(char) for char in text]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids
        return ids

    def decode(self, token_ids):
        chars = []
        for token_id in token_ids:
            if token_id >= 1000:
                chars.append(self._char(token_id))
        return "".join(chars)

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
    ):
        ids = [self.bos_token_id]

        for message in messages:
            if message["role"] == "user":
                ids.extend(self.user_marker)
                ids.extend(self.encode(message["content"], add_special_tokens=False))
                ids.extend(self.end_of_turn)
            elif message["role"] in ("assistant", "model"):
                ids.extend(self.assistant_marker)
                ids.extend(self.encode(message["content"], add_special_tokens=False))
                ids.extend(self.end_of_turn)
            else:
                raise ValueError(f"Unsupported role: {message['role']}")

        if add_generation_prompt:
            ids.extend(self.assistant_marker)

        if tokenize:
            return ids
        return self.decode(ids)


class SpecialTokenWorkflowTests(unittest.TestCase):
    def test_openthoughts_tokenizer_format_does_not_double_apply_bos(self):
        tokenizer = GemmaLikeChatTokenizer()
        example = {
            "conversations": [
                {"value": "Problem?"},
                {"value": "Answer!"},
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data.jsonl"
            data_path.write_text(json.dumps(example) + "\n")

            dataset = OpenThoughtsDataset(
                str(data_path),
                tokenizer,
                max_length=128,
                format="tokenizer",
                loss_on_prompt=False,
                filter_length=True,
            )
            item = dataset[0]

        expected_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Problem?"}],
            tokenize=True,
            add_generation_prompt=True,
        )
        expected_full = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": "Problem?"},
                {"role": "assistant", "content": "Answer!"},
            ],
            tokenize=True,
            add_generation_prompt=False,
        )

        self.assertEqual(item["input_ids"], expected_full)
        self.assertEqual(item["input_ids"].count(tokenizer.bos_token_id), 1)
        self.assertEqual(item["input_ids"][:len(expected_prompt)], expected_prompt)
        self.assertEqual(item["labels"][:len(expected_prompt)], [-100] * len(expected_prompt))
        self.assertEqual(item["labels"][len(expected_prompt):], expected_full[len(expected_prompt):])

    def test_gemma_attribution_auto_retemplates_marked_prompt_files(self):
        tokenizer = GemmaLikeChatTokenizer()
        raw_prompt = (
            f"{DEEPSEEK_BOS_TOKEN}{DEEPSEEK_USER_TOKEN}Problem?\n\nAnswer:"
            f"{DEEPSEEK_ASSISTANT_TOKEN}<think>\nabc!"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            prompt_path = Path(tmpdir) / "prompt.txt"
            prompt_path.write_text(raw_prompt)
            prompt_tokens, target, _ = load_prompt_file(
                prompt_path,
                tokenizer,
                prompt_format="auto",
                model_type="gemma2",
            )

        expected_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Problem?\n\nAnswer:"}],
            tokenize=True,
            add_generation_prompt=True,
        ) + tokenizer.encode("<think>\nabc", add_special_tokens=False)

        self.assertEqual(prompt_tokens, expected_prompt)
        self.assertEqual(target, tokenizer.encode("!", add_special_tokens=False)[0])
        self.assertNotIn(tokenizer._char_id("｜"), prompt_tokens)

    def test_attribution_raw_prompt_format_preserves_file_text(self):
        tokenizer = GemmaLikeChatTokenizer()
        raw_prompt = (
            f"{DEEPSEEK_BOS_TOKEN}{DEEPSEEK_USER_TOKEN}Problem?"
            f"{DEEPSEEK_ASSISTANT_TOKEN}A!"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            prompt_path = Path(tmpdir) / "prompt.txt"
            prompt_path.write_text(raw_prompt)
            prompt_tokens, target, _ = load_prompt_file(
                prompt_path,
                tokenizer,
                prompt_format="raw",
                model_type="gemma2",
            )

        raw_tokens = tokenizer.encode(raw_prompt, add_special_tokens=False)
        self.assertEqual(prompt_tokens, raw_tokens[:-1])
        self.assertEqual(target, raw_tokens[-1])

    def test_attribution_auto_warns_when_falling_back_to_raw_prompt_format(self):
        tokenizer = GemmaLikeChatTokenizer()

        with tempfile.TemporaryDirectory() as tmpdir:
            prompt_path = Path(tmpdir) / "prompt.txt"
            prompt_path.write_text("Problem? Answer!")
            with self.assertLogs("training", level="WARNING") as logs:
                prompt_tokens, target, _ = load_prompt_file(
                    prompt_path,
                    tokenizer,
                    prompt_format="auto",
                    model_type="gemma2",
                )

        raw_tokens = tokenizer.encode("Problem? Answer!", add_special_tokens=False)
        self.assertEqual(prompt_tokens, raw_tokens[:-1])
        self.assertEqual(target, raw_tokens[-1])
        self.assertTrue(
            any("falling back to raw tokenization" in message for message in logs.output)
        )

    def test_attribution_load_prompts_supports_round_robin_shards(self):
        tokenizer = GemmaLikeChatTokenizer()

        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_dir = Path(tmpdir)
            for name in ["a", "b", "c", "d", "e"]:
                (prompts_dir / f"{name}.txt").write_text(f"{name}!")

            prompts = load_prompts(
                str(prompts_dir),
                tokenizer,
                prompt_format="raw",
                num_shards=3,
                shard_index=1,
            )

        self.assertEqual(list(prompts), ["b", "e"])

    def test_auto_shard_parses_visible_cuda_devices(self):
        self.assertEqual(
            _parse_visible_cuda_devices("2,4", device_count=2),
            ["2", "4"],
        )
        self.assertEqual(
            _parse_visible_cuda_devices(None, device_count=3),
            ["0", "1", "2"],
        )
        self.assertEqual(
            _parse_visible_cuda_devices("-1", device_count=0),
            [],
        )
        self.assertEqual(
            _parse_visible_cuda_devices("2,4", device_count=0),
            [],
        )

    def test_auto_shard_worker_command_sets_explicit_shard_without_recursing(self):
        parser = build_parser()
        args = parser.parse_args([
            "--checkpoint", "ckpt",
            "--run_name", "run",
            "--scan", "scan",
            "--prompts", "prompts",
            "--output_dir", "out",
            "--auto_shard_gpus",
            "--prompt_format", "raw",
            "--max_n_logits", "3",
            "--batch_size", "5",
            "--max_feature_nodes", "7",
            "--node_threshold", "0.5",
            "--edge_threshold", "0.9",
        ])

        command = _build_auto_shard_worker_command(args, num_shards=4, shard_index=2)

        self.assertNotIn("--auto_shard_gpus", command)
        self.assertIn("--num_shards", command)
        self.assertIn("--shard_index", command)
        self.assertEqual(command[command.index("--num_shards") + 1], "4")
        self.assertEqual(command[command.index("--shard_index") + 1], "2")
        self.assertEqual(command[command.index("--device") + 1], "cuda")

    def test_lmsys_chat_accepts_batch_encoding_template_output(self):
        class BatchEncodingChatTokenizer(GemmaLikeChatTokenizer):
            def apply_chat_template(self, *args, **kwargs):
                ids = super().apply_chat_template(*args, **kwargs)
                if kwargs.get("tokenize", False):
                    return {"input_ids": ids, "attention_mask": [1] * len(ids)}
                return ids

        dataset = LMSYSChatDataset.__new__(LMSYSChatDataset)
        dataset.tokenizer = BatchEncodingChatTokenizer()

        messages = [
            {"role": "user", "content": "Problem?"},
            {"role": "assistant", "content": "Answer!"},
        ]
        ids = dataset._chat_template_ids(messages, add_generation_prompt=False)

        self.assertIsInstance(ids, list)
        self.assertTrue(all(isinstance(token_id, int) for token_id in ids))
        self.assertEqual(ids[0], dataset.tokenizer.bos_token_id)

    def test_eval_tokenize_chat_accepts_batch_encoding_template_output(self):
        class BatchEncodingChatTokenizer(GemmaLikeChatTokenizer):
            def apply_chat_template(self, *args, **kwargs):
                ids = super().apply_chat_template(*args, **kwargs)
                if kwargs.get("tokenize", False):
                    return {"input_ids": ids, "attention_mask": [1] * len(ids)}
                return ids

        tokenizer = BatchEncodingChatTokenizer()
        prompt_ids, response_ids = tokenize_chat_prompt_response(
            tokenizer,
            [{"role": "user", "content": "Problem?"}],
            "Answer!",
        )

        self.assertIsInstance(prompt_ids, list)
        self.assertIsInstance(response_ids, list)
        self.assertTrue(all(isinstance(token_id, int) for token_id in prompt_ids))
        self.assertTrue(all(isinstance(token_id, int) for token_id in response_ids))
        self.assertEqual(prompt_ids[0], tokenizer.bos_token_id)


if __name__ == "__main__":
    unittest.main()
