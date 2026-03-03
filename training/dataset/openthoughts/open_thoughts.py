from torch.utils.data import Dataset
import json

from training.dataset.types import DatasetItem
from training.dataset.openthoughts.types import DataFormat


# DeepSeek R1 Distill format tokens
DEEPSEEK_USER_TOKEN = "<｜User｜>"
DEEPSEEK_ASSISTANT_TOKEN = "<｜Assistant｜>"

# Qwen/QwQ chat format tokens
QWEN_IM_START = "<|im_start|>"
QWEN_IM_END = "<|im_end|>"


class OpenThoughtsDataset(Dataset):
    """Dataset for loading OpenThoughts reasoning traces.

    Supports multiple formats for both SFT and bridging training.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 8192,
        format: DataFormat = "tokenizer",
        truncate: bool = False,
        loss_on_prompt: bool = False,
        filter_length: bool = False,
    ):
        """Initialize the dataset.

        Args:
            data_path: Path to local JSONL file
            tokenizer: Tokenizer for processing text
            max_length: Maximum sequence length
            format: "tokenizer" uses apply_chat_template (default),
                    "deepseek"/"qwen" use explicit format tokens
            truncate: If True, truncate to max_length. If False, raise on overflow.
            loss_on_prompt: If True, include prompt tokens in loss.
            filter_length: Filter out examples exceeding max_length at load time.
        """
        if truncate and filter_length:
            print('truncating enabled, disabling filter_length')
            filter_length = False

        self._format = format
        self._tokenizer = tokenizer

        raw_examples = self._load_raw(data_path)
        self._items = self._tokenize_all(
            raw_examples, tokenizer, max_length, format, truncate,
            loss_on_prompt, filter_length,
        )

    @staticmethod
    def _load_raw(data_path: str) -> list[dict]:
        """Load raw JSONL data from local path or HuggingFace (hf://repo/path)."""
        print(f"Loading data from {data_path}")

        if data_path.startswith("hf://"):
            from huggingface_hub import hf_hub_download
            hf_path = data_path[len("hf://"):]
            parts = hf_path.split("/", 2)  # org, repo, filepath
            repo_id = f"{parts[0]}/{parts[1]}"
            filepath = parts[2]
            local_path = hf_hub_download(repo_id=repo_id, filename=filepath, repo_type="dataset")
        else:
            local_path = data_path

        examples = []
        with open(local_path, 'r') as f:
            for line in f:
                examples.append(json.loads(line.strip()))
        return examples

    @staticmethod
    def _format_and_tokenize(
        prompt: str, response: str, tokenizer, fmt: DataFormat,
    ) -> tuple[list[int], int]:
        """Format and tokenize a prompt-response pair.

        Returns:
            (input_ids, prompt_len) where prompt_len is the number of tokens
            before the assistant response (used for loss masking).
        """
        if fmt in ("deepseek", "qwen"):
            # Normalize "<think> " to "<think>\n" to match expected format
            if response.startswith("<think> "):
                response = "<think>\n" + response[len("<think> "):]

        if fmt == "deepseek":
            prompt_text = f"{DEEPSEEK_USER_TOKEN}{prompt}{DEEPSEEK_ASSISTANT_TOKEN}"
            full_text = f"{prompt_text}{response}"
        elif fmt == "qwen":
            prompt_text = f"{QWEN_IM_START}user\n{prompt}{QWEN_IM_END}\n{QWEN_IM_START}assistant\n"
            full_text = f"{prompt_text}{response}{QWEN_IM_END}\n"
        else:
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            input_ids = tokenizer.encode(full_text, add_special_tokens=True)
            prompt_len = len(tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            ))
            return input_ids, prompt_len

        input_ids = tokenizer.encode(full_text, add_special_tokens=True)
        prompt_len = len(tokenizer.encode(prompt_text, add_special_tokens=True))
        return input_ids, prompt_len

    @staticmethod
    def _tokenize_all(
        raw_examples: list[dict],
        tokenizer,
        max_length: int,
        fmt: DataFormat,
        truncate: bool,
        loss_on_prompt: bool,
        filter_length: bool,
    ) -> list[DatasetItem]:
        """Pre-tokenize all examples into DatasetItems."""
        total = len(raw_examples)
        print(f"Pre-tokenizing {total} examples...")
        items: list[DatasetItem] = []
        skipped = 0

        for i, example in enumerate(raw_examples):
            prompt = example['conversations'][0]['value']
            response = example['conversations'][1]['value']
            input_ids, prompt_len = OpenThoughtsDataset._format_and_tokenize(
                prompt, response, tokenizer, fmt,
            )

            original_length = len(input_ids)
            truncated = False
            if len(input_ids) > max_length:
                if filter_length:
                    skipped += 1
                    continue
                elif truncate:
                    input_ids = input_ids[:max_length]
                    truncated = True
                else:
                    raise ValueError(f"Sequence {len(input_ids)} > max_length {max_length}")

            if loss_on_prompt:
                labels = input_ids.copy()
            else:
                labels = [-100] * min(prompt_len, len(input_ids)) + input_ids[prompt_len:]

            items.append({
                "input_ids": input_ids,
                "labels": labels,
                "truncated": truncated,
                "original_length": original_length,
            })

            if (i + 1) % 10000 == 0:
                print(f"  Pre-tokenized {i + 1}/{total}")

        if filter_length:
            print(f"Kept {len(items)}/{total} after filtering ({skipped} skipped)")
        else:
            print(f"Pre-tokenization complete ({len(items)} examples)")
        return items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> DatasetItem:
        return self._items[idx]
