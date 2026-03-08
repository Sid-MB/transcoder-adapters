
from typing import TypeVar

from torch.utils.data import Dataset
from training.dataset.types import SizedDataset

DatasetRow = TypeVar("DatasetRow")

class CachedDataset(SizedDataset[DatasetRow]):
    """Wraps a SizedDataset to cache its items in memory after the first access.

    This is useful for datasets that are expensive to compute but fit in memory.
    """

    def __init__(self, dataset: SizedDataset[DatasetRow]):
        if isinstance(dataset, CachedDataset):
            # Unwrap to avoid double-wrapping and redundant caching.
            self.dataset = dataset.dataset
            self.cache = dataset.cache
        else:
            self.dataset = dataset
            self.cache: dict[int, DatasetRow] = {}

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> DatasetRow:
        try:
            return self.cache[idx]
        except KeyError:
            self.cache[idx] = self.dataset[idx]
            return self.cache[idx]

    def clear(self):
        self.cache.clear()
