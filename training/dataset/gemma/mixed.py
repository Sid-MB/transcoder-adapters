import random

import numpy as np
from torch.utils.data import Dataset
from training.dataset.types import DatasetItem, SizedDataset

class MixedDataset(Dataset):
    """Randomly interleaves multiple datasets according to specified weights.

    On each access, picks a source dataset with probability proportional to the
    given weights, then indexes into that dataset.  The total length equals the
    sum of all constituent dataset lengths.
    """

    def __init__(
        self,
        datasets: tuple[SizedDataset, ...],
        weights: tuple[float, ...],
        *,
        seed: int = 80,
    ):
        """Initialize the mixed dataset.

        Args:
            datasets: List of Dataset instances to mix.
            weights: Sampling weight for each dataset (need not sum to 1).
            seed: Random seed for reproducible mixing.
        """
        if len(datasets) != len(weights):
            raise ValueError(
                f"Got {len(datasets)} datasets but {len(weights)} weights"
            )

        self.datasets = datasets
        total_weight = sum(weights)
        self.weights = [w / total_weight for w in weights]

        # Pre-compute assignments for deterministic, shuffle-safe indexing.
        rng = random.Random(seed)
        total = sum(len(d) for d in datasets)
        ds_indices = rng.choices(range(len(datasets)), weights=self.weights, k=total)

        # Track per-dataset cursors so each dataset is sampled uniformly.
        local_pools: list[list[int]] = []
        for ds in datasets:
            pool = list(range(len(ds)))
            rng.shuffle(pool)
            local_pools.append(pool)
        cursors = [0] * len(datasets)

        # Store as numpy arrays instead of list[tuple] for ~20x less memory
        local_indices = np.empty(total, dtype=np.int32)
        for i, ds_idx in enumerate(ds_indices):
            pool = local_pools[ds_idx]
            local_indices[i] = pool[cursors[ds_idx] % len(pool)]
            cursors[ds_idx] += 1

        self._ds_indices = np.array(ds_indices, dtype=np.int8)
        self._local_indices = local_indices

    def __len__(self) -> int:
        return len(self._ds_indices)

    def __getitem__(self, idx: int) -> DatasetItem:
        ds_idx = int(self._ds_indices[idx])
        local_idx = int(self._local_indices[idx])
        return self.datasets[ds_idx][local_idx]
