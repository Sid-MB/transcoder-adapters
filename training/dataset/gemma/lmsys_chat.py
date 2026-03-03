from torch.utils.data import Dataset

from datasets import Dataset as HFDataset, load_dataset
from training.dataset.types import DatasetItem


class LMSYSChatDataset(Dataset):
    """Dataset for LMSYS-Chat multi-turn conversations.

    Loads conversations via the HuggingFace datasets library and optionally
    masks non-assistant tokens from the loss.

    Schema (lmsys/lmsys-chat-1m):
        conversation_id, model, conversation, turn, language, openai_moderation, redacted
    The ``conversation`` column is a list of {"role": str, "content": str} dicts.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 8192,
        truncate: bool = False,
        loss_on_prompt: bool = False,
        split: str = "train",
        conversation_field: str = "conversation",
    ):
        """Initialize the dataset.

        Args:
            data_path: HuggingFace dataset identifier (e.g. "lmsys/lmsys-chat-1m").
            tokenizer: HuggingFace tokenizer.
            max_length: Maximum sequence length in tokens.
            truncate: If True, truncate long sequences. If False, raise on overflow.
            loss_on_prompt: If True, compute loss on all tokens.
                            If False, mask user/system turns (loss on assistant only).
            split: Dataset split to load.
            conversation_field: Column containing the conversation list.
        """
        print(f"Loading LMSYS-Chat data: {data_path} (split={split})")
        ds: HFDataset = load_dataset(data_path, split=split) # pyright: ignore[reportAssignmentType]
        print(f"Loaded {len(ds)} examples, pre-tokenizing...")

        self._items: list[DatasetItem] = []
        for i in range(len(ds)):
            conversation = ds[i][conversation_field]

            result = tokenizer.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_assistant_tokens_mask=True,
            )
            input_ids = result["input_ids"]
            assistant_mask = result["assistant_masks"]

            original_length = len(input_ids)
            truncated = False
            if len(input_ids) > max_length:
                if truncate:
                    input_ids = input_ids[:max_length]
                    assistant_mask = assistant_mask[:max_length]
                    truncated = True
                else:
                    raise ValueError(
                        f"Sequence length {len(input_ids)} > max_length {max_length}"
                    )

            if loss_on_prompt:
                labels = list(input_ids)
            else:
                labels = [tok if mask else -100 for tok, mask in zip(input_ids, assistant_mask)]

            self._items.append({
                "input_ids": list(input_ids),
                "labels": labels,
                "truncated": truncated,
                "original_length": original_length,
            })

            if (i + 1) % 50000 == 0:
                print(f"  Pre-tokenized {i + 1}/{len(ds)}")

        del ds
        print(f"Pre-tokenization complete ({len(self._items)} examples)")

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> DatasetItem:
        return self._items[idx]
