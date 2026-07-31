from functools import partial
from collections.abc import Sequence
from typing import Literal
import torch
from torch.utils.data import DataLoader
from torch import Generator as TorchGenerator

from .collate import collate_fn
from .types import DatasetItem, SizedDataset
from helpers.log import logger

from training.config import DatasetEntryConfig, LengthExcessionBehavior

from .CachedDataset import CachedDataset


class PredefinedDataset:
    """
    Builds training (and optional validation) datasets from a list of DatasetEntryConfig entries,
    wrapping multiple datasets in a MixedDataset when needed.
    """

    # Type for loaded datasets
    DatasetSplits = Literal["train", "val"]
    LoadedDatasets = dict[DatasetSplits, SizedDataset[DatasetItem]]
    Dataloaders = dict[DatasetSplits, DataLoader[DatasetItem]]

    def __init__(
        self,
        dataset_entries: list[DatasetEntryConfig],
        *,
        tokenizer,
        loss_on_prompt: bool = False,
        batch_size: int = 1,
        dataloader_seed: int = 81,
        total_rows: int | None = None,
        weight_by: str = "rows",
    ):
        self.dataset_entries = dataset_entries
        self.tokenizer = tokenizer
        self.loss_on_prompt = loss_on_prompt
        self.total_rows = total_rows
        self.weight_by = weight_by

        self._loaded_datasets: PredefinedDataset.LoadedDatasets | None = None
        """Cache for loaded datasets. Initialized to None, and populated on first call to _load_dataset()."""
        self._caches: list[CachedDataset] = []
        """References to CachedDataset wrappers, for clearing between epochs."""

        self.batch_size = batch_size
        self.dataloader_seed = dataloader_seed

    def _load_dataset(self):
        """
        Loads the dataset according to the entries specified in the constructor.
        Caches the loaded dataset for future calls.
        """
        if self._loaded_datasets is None:
            self._loaded_datasets = self._make_dataset()
        return self._loaded_datasets

    @staticmethod
    def _filter_by_length(dataset: CachedDataset[DatasetItem], name: str = "") -> None:
        """Filter a dataset in-place to only include non-truncated examples.

        Assumes the dataset was constructed with truncate=True, so each item
        has a 'truncated' flag. Subsamples the CachedDataset to only items
        where truncated=False (i.e. the full sequence fit within max_length).
        The cache is preserved for kept items.
        """
        logger.info(f"Dropping truncated rows from dataset split of {f"\"{name}\"" if name else '[no dataset name]'}...")
        total = len(dataset)
        valid_indices = []
        for i in range(total):
            item = dataset[i]
            if not item["truncated"]:
                valid_indices.append(i)
        label = f" ({name})" if name else ""
        logger.info(f"Filtered{label}: kept {len(valid_indices)}/{total} examples that fit within max_length")
        dataset.subsample(valid_indices)

    def _build_single_dataset(
        self, entry: DatasetEntryConfig, *, truncate: bool, is_val: bool = False
    ) -> SizedDataset[DatasetItem]:
        """Instantiate a single dataset from its entry config."""
        datapath = entry.val_datapath if is_val else entry.datapath
        assert datapath is not None

        match entry.type:
            case "fineweb":
                from training.dataset.gemma2.fineweb import FineWebDataset
                return FineWebDataset(
                    data_path=datapath,
                    tokenizer=self.tokenizer,
                    max_length=entry.max_seq_length,
                    truncate=truncate,
                )
            case "lmsys_chat":
                from training.dataset.gemma2.lmsys_chat import LMSYSChatDataset
                return LMSYSChatDataset(
                    data_path=datapath,
                    tokenizer=self.tokenizer,
                    max_length=entry.max_seq_length,
                    truncate=truncate,
                    loss_on_prompt=self.loss_on_prompt,
                )
            case "open_thoughts":
                from training.dataset.openthoughts.open_thoughts import OpenThoughtsDataset
                return OpenThoughtsDataset(
                    data_path=datapath,
                    tokenizer=self.tokenizer,
                    max_length=entry.max_seq_length,
                    format=entry.data_format or "tokenizer",
                    truncate=truncate,
                    loss_on_prompt=self.loss_on_prompt,
                )
            case _:
                raise ValueError(f"Unknown dataset type: {entry.type}")

    def _process_entry(
        self, entry: DatasetEntryConfig
    ) -> tuple[CachedDataset[DatasetItem], CachedDataset[DatasetItem] | None]:
        """Build, filter, and subsample a single entry. Returns (train_ds, val_ds_or_None)."""
        should_filter = entry.length_excession_behavior == LengthExcessionBehavior.FILTER
        truncate = should_filter or entry.length_excession_behavior == LengthExcessionBehavior.TRUNCATE

        # Build train dataset and wrap in CachedDataset early so filtering populates the cache
        train_ds = CachedDataset(self._build_single_dataset(entry, truncate=truncate))
        self._caches.append(train_ds)

        if should_filter:
            self._filter_by_length(train_ds, entry.datapath)

        if entry.num_rows is not None and len(train_ds) > entry.num_rows:
            g = TorchGenerator().manual_seed(self.dataloader_seed)
            indices = torch.randperm(len(train_ds), generator=g)[:entry.num_rows].tolist()
            logger.info(f"Subsampled {entry.num_rows}/{len(train_ds)} rows from '{entry.datapath}'")
            train_ds.subsample(indices)

        # Build val dataset if path provided
        val_ds: CachedDataset[DatasetItem] | None = None
        if entry.val_datapath:
            val_ds = CachedDataset(self._build_single_dataset(entry, truncate=truncate, is_val=True))
            if should_filter:
                self._filter_by_length(val_ds, entry.val_datapath)

        return train_ds, val_ds

    @staticmethod
    def _estimate_avg_tokens(
        datasets: Sequence[SizedDataset[DatasetItem]], seed: int, sample_size: int = 1000
    ) -> list[float]:
        """Estimate average token count per row for each dataset by random sampling.

        Uses seed+i per dataset, matching the allocation permutation so that
        sampled rows are a subset of the final allocated rows (and get cached).
        """
        avg_tokens: list[float] = []
        for i, ds in enumerate(datasets):
            n = min(sample_size, len(ds))
            indices = torch.randperm(len(ds), generator=TorchGenerator().manual_seed(seed + i))[:n].tolist()
            total_toks = sum(len(ds[idx]["input_ids"]) for idx in indices)
            avg = total_toks / n
            avg_tokens.append(avg)
        logger.info(f"Estimated avg tokens per dataset: {[f'{t:.0f}' for t in avg_tokens]}")
        return avg_tokens

    def _make_dataset(self) -> LoadedDatasets:
        logger.info(f"Loading {len(self.dataset_entries)} dataset(s)")

        train_datasets: list[CachedDataset[DatasetItem]] = []
        val_datasets: list[CachedDataset[DatasetItem]] = []
        weights: list[float] = []
        active_entries: list[DatasetEntryConfig] = []

        for entry in self.dataset_entries:
            if entry.weight <= 0:
                logger.info(f"Skipping dataset '{entry.datapath}' (weight={entry.weight})")
                continue
            train_ds, val_ds = self._process_entry(entry)
            train_datasets.append(train_ds)
            weights.append(entry.weight)
            active_entries.append(entry)
            if val_ds is not None:
                val_datasets.append(val_ds)

        # Compute effective weights (adjust for token length if needed)
        effective_weights = list(weights)
        if self.weight_by == "tokens" and len(train_datasets) > 1:
            avg_tokens = self._estimate_avg_tokens(train_datasets, seed=self.dataloader_seed)
            effective_weights = [w / t for w, t in zip(weights, avg_tokens)]

        # Allocate total_rows across datasets if set
        if self.total_rows is not None:
            total_ew = sum(effective_weights)
            row_counts = [round(self.total_rows * ew / total_ew) for ew in effective_weights]
            for i, target in enumerate(row_counts):
                if len(train_datasets[i]) > target:
                    g = TorchGenerator().manual_seed(self.dataloader_seed + i)
                    indices = torch.randperm(len(train_datasets[i]), generator=g)[:target].tolist()
                    logger.info(
                        f"Allocated {target}/{len(train_datasets[i])} rows "
                        f"from '{active_entries[i].datapath}'"
                    )
                    train_datasets[i].subsample(indices)
                else:
                    logger.warning(
                        f"Dataset '{active_entries[i].datapath}' has only "
                        f"{len(train_datasets[i])} rows but {target} were requested"
                    )
            # After subsampling to target counts, use equal weights in MixedDataset
            mix_weights: tuple[float, ...] = tuple(1.0 for _ in train_datasets)
        else:
            mix_weights = tuple(effective_weights)

        # Compute exact per-dataset stats over all rows that will be used in training.
        # This populates the per-dataset caches so training doesn't re-tokenize.
        self.dataset_stats: list[dict[str, str | int]] = []
        for i, ds in enumerate(train_datasets):
            entry = active_entries[i]
            n_rows = len(ds)
            total_tokens = sum(len(ds[j]["input_ids"]) for j in range(n_rows)) # We're fine because this happens before padding
            self.dataset_stats.append({
                "type": entry.type,
                "datapath": entry.datapath,
                "rows": n_rows,
                "total_tokens": total_tokens,
            })
            logger.info(f"Dataset '{entry.datapath}': {n_rows:,} rows, {total_tokens:,} tokens")

        # Wrap in MixedDataset if multiple, otherwise use directly
        if len(train_datasets) == 1:
            train_final: SizedDataset[DatasetItem] = train_datasets[0]
        else:
            from training.dataset.MixedDataset import MixedDataset
            train_final = MixedDataset(datasets=tuple(train_datasets), weights=mix_weights)
            logger.info(f"Created mixed dataset, rows={len(train_final)}")

        result: PredefinedDataset.LoadedDatasets = {"train": train_final}

        if val_datasets:
            if len(val_datasets) == 1:
                result["val"] = val_datasets[0]
            else:
                from training.dataset.MixedDataset import MixedDataset
                result["val"] = MixedDataset(datasets=tuple(val_datasets))

        return result

    def _make_dataloader(self, datasets: LoadedDatasets) -> Dataloaders:
        """
        Creates new dataloaders for each split in the given datasets.
        """
        collate_with_tokenizer = partial(collate_fn, tokenizer=self.tokenizer)

        generator = TorchGenerator()
        generator.manual_seed(self.dataloader_seed)

        assert "train" in datasets, (
            "Training split ('train') is required in loaded datasets"
        )
        logger.info(f"Creating dataloaders, shuffling (seed={self.dataloader_seed})")
        dataloaders: PredefinedDataset.Dataloaders = {
            "train": DataLoader(
                datasets["train"], # pyright: ignore[reportArgumentType]
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=collate_with_tokenizer,
                num_workers=4,
                pin_memory=False,
                persistent_workers=True,
                generator=generator,
            )
        }

        if "val" in datasets:
            dataloaders["val"] = DataLoader(
                datasets["val"],  # pyright: ignore[reportArgumentType]
                batch_size=1,
                shuffle=False,
                collate_fn=collate_with_tokenizer,
                num_workers=4,
                pin_memory=False,  # note: setting this True caused dataloader errors in our runs
                persistent_workers=True,
            )

        assert dataloaders.keys() == datasets.keys(), (
            "Note: Dataloader did not generate loaders for all dataset splits"
        )
        return dataloaders

    def clear_caches(self):
        """Clear all CachedDataset caches to free memory between epochs."""
        for cache in self._caches:
            cache.clear()

    def get_epoch_dataloaders(self, epoch: int) -> Dataloaders:
        """
        Loads the dataset (if not already loaded) and creates dataloaders.
        """
        self._load_dataset()
        self.clear_caches()
        assert self._loaded_datasets is not None
        return self._make_dataloader(self._loaded_datasets)

    def load_datasets_and_dataloaders(self) -> tuple[LoadedDatasets, Dataloaders]:
        """
        Loads the dataset (if not already loaded) and creates dataloaders for each split.
        Returns a tuple of (loaded_datasets, dataloaders).
        """
        datasets = self._load_dataset()
        dataloaders = self._make_dataloader(datasets)
        return datasets, dataloaders
