from torch.utils.data import Dataset

from datasets import load_dataset, Dataset as HFDataset
from training.dataset.types import DatasetItem


class FineWebDataset(Dataset):
    """Dataset for FineWeb pretraining data.

    Loads plain-text documents via the HuggingFace datasets library and applies
    language modeling loss on all tokens.

    Schema (science-of-finetuning/fineweb-1m-sample):
        text, id, dump, url, date, file_path, language, language_score, token_count
    """

    def __init__(
        self,
        data_path: str,
        *,
        tokenizer,
        max_length: int = 8192,
        truncate: bool = False,
        split: str = "train",
        text_field: str = "text",
    ):
        """Initialize the dataset.

        Args:
            data_path: HuggingFace dataset identifier
                       (e.g. "science-of-finetuning/fineweb-1m-sample").
            tokenizer: HuggingFace tokenizer.
            max_length: Maximum sequence length in tokens.
            truncate: If True, truncate long sequences. If False, raise on overflow.
            split: Dataset split to load (e.g. "train", "validation").
            text_field: Column containing the document text.
        """
        self.max_length = max_length
        self.truncate = truncate

        print(f"Loading FineWeb data: {data_path} (split={split})")
        ds: HFDataset = load_dataset(data_path, split=split) # pyright: ignore[reportAssignmentType]
        print(f"Loaded {len(ds)} examples, pre-tokenizing...")

        self._items: list[DatasetItem] = []
        for i in range(len(ds)):
            text = ds[i][text_field]
            input_ids = tokenizer.encode(text, add_special_tokens=True)

            original_length = len(input_ids)
            truncated = False
            if len(input_ids) > max_length:
                if truncate:
                    input_ids = input_ids[:max_length]
                    truncated = True
                else:
                    raise ValueError(
                        f"Sequence length {len(input_ids)} > max_length {max_length}"
                    )

            self._items.append({
                "input_ids": input_ids,
                "labels": input_ids.copy(),
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
