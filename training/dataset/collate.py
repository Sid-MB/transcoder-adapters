from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

def collate_fn(examples, tokenizer: "PreTrainedTokenizerFast"):
    """Simple collate function for batching examples."""
    # Extract sequences
    input_ids = [ex["input_ids"] for ex in examples]
    labels = [ex["labels"] for ex in examples]
    
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True # remove the warning. we aren't padding when we tokenize because we run an analysis of the # of tokens in each sample beforehand (and maybe other reasons too?). https://github.com/huggingface/transformers/issues/22638#issuecomment-1560406455

    # Pad input_ids and attention_mask
    batch = tokenizer.pad(
        {"input_ids": input_ids},
        padding=True,
        return_tensors="pt"
    )

    # Manually pad labels with -100
    max_length = batch["input_ids"].shape[1]
    padded_labels = []

    for label_seq in labels:
        padded = label_seq + [-100] * (max_length - len(label_seq))
        padded_labels.append(padded)

    batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

    # Pass through truncation stats
    batch["truncated"] = [ex["truncated"] for ex in examples]
    batch["original_length"] = [ex["original_length"] for ex in examples]

    return batch
