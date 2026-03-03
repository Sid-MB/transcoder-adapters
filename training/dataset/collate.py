import torch

def collate_fn(examples, pad_token_id: int):
    """Collate function for batching examples.

    Pads input_ids/labels and generates attention_mask. Takes only a
    pad_token_id (int) instead of the full tokenizer, avoiding the cost
    of pickling the tokenizer to DataLoader worker processes.
    """
    input_ids = [ex["input_ids"] for ex in examples]
    labels = [ex["labels"] for ex in examples]

    max_length = max(len(ids) for ids in input_ids)
    batch_size = len(examples)

    padded_input_ids = torch.full((batch_size, max_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_length, dtype=torch.long)
    padded_labels = torch.full((batch_size, max_length), -100, dtype=torch.long)

    for i, (ids, labs) in enumerate(zip(input_ids, labels)):
        seq_len = len(ids)
        padded_input_ids[i, :seq_len] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, :seq_len] = 1
        padded_labels[i, :len(labs)] = torch.tensor(labs, dtype=torch.long)

    return {
        "input_ids": padded_input_ids,
        "attention_mask": attention_mask,
        "labels": padded_labels,
        "truncated": [ex["truncated"] for ex in examples],
        "original_length": [ex["original_length"] for ex in examples],
    }
