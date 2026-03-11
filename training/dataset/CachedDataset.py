
from typing import TypeVar

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
        self._index_map: list[int] | None = None

    def __len__(self) -> int:
        if self._index_map is not None:
            return len(self._index_map)
        return len(self.dataset)

    def __getitem__(self, idx: int) -> DatasetRow:
        real_idx = self._index_map[idx] if self._index_map is not None else idx
        try:
            return self.cache[real_idx]
        except KeyError:
            self.cache[real_idx] = self.dataset[real_idx]
            return self.cache[real_idx]

    def subsample(self, indices: list[int]) -> None:
        """Restrict this dataset to only the given indices, in-place.

        Remaps cached entries to new contiguous indices. The backing
        dataset is kept (since uncached rows may still need to be fetched),
        but only the subsampled indices are accessible.
        """
        if self._index_map is not None:
            # Compose with existing mapping
            indices = [self._index_map[i] for i in indices]

        # Prune cache to only keep entries in the new index set
        keep = set(indices)
        self.cache = {k: v for k, v in self.cache.items() if k in keep}

        self._index_map = indices

    def clear(self):
        self.cache.clear()
