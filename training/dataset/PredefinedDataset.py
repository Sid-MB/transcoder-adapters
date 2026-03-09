from enum import Enum
from functools import partial
from typing import Literal
import torch
from torch.utils.data import DataLoader, Subset
from torch import Generator as TorchGenerator

from .collate import collate_fn
from .types import DatasetItem, SizedDataset
from helpers.log import logger

from .gemma.config import FineWebLMSysMixedConfig

from .CachedDataset import CachedDataset
from .datasetspecific_config import DatasetSpecificConfig, DatasetType
from .openthoughts.config import OpenThoughtsConfig


class LengthExcessionBehavior(Enum):
    TRUNCATE = "truncate"
    ERROR = "error"
    """Throw if any sequences are over the max length."""
    FILTER = "filter"
    """Filter out any sequences that exceed the maximum length."""


class PredefinedDataset:
    """
    Specifies how a dataset should be loaded and processed for training
    """

    # Type for loaded datasets
    DatasetSplits = Literal["train", "val"]
    LoadedDatasets = dict[DatasetSplits, SizedDataset[DatasetItem]]
    Dataloaders = dict[DatasetSplits, DataLoader[DatasetItem]]

    def __init__(
        self,
        dataset_type: DatasetType,
        *,
        tokenizer,
        length_excession_behavior: LengthExcessionBehavior = LengthExcessionBehavior.ERROR,
        loss_on_prompt: bool = False,
        dataset_specific_config: DatasetSpecificConfig | None = None,
        batch_size: int = 1,
        dataloader_seed: int = 81,
        dataset_rows: int | None = None,
    ):
        self.dataset_type = dataset_type
        self.tokenizer = tokenizer
        self.length_excession_behavior = length_excession_behavior
        self.loss_on_prompt = loss_on_prompt
        self.dataset_specific_config = dataset_specific_config
        self.dataset_rows = dataset_rows
        if dataset_specific_config is not None:
            assert dataset_specific_config.dataset_type == dataset_type, (
                f"Config type mismatch: config is for {dataset_specific_config.dataset_type}, "
                f"but dataset_type is {dataset_type}"
            )

        self._loaded_datasets: PredefinedDataset.LoadedDatasets | None = None
        """
        Cache for loaded datasets (full, before dataset_rows subsampling). Initialized to None, and populated on first call to _load_dataset().
        """
        self._caches: list[CachedDataset] = []
        """References to CachedDataset wrappers, for clearing between epochs."""

        self.batch_size = batch_size
        self.dataloader_seed = dataloader_seed

    def _load_dataset(self):
        """
        Loads the dataset according to the type and configuration parameters specified in the constructor. Caches the loaded dataset for future calls; there's no need to call this method more than once per instance.
        """
        if self._loaded_datasets is None:
            self._loaded_datasets = self._make_dataset()
        return self._loaded_datasets

    @staticmethod
    def _filter_by_length(dataset: SizedDataset[DatasetItem], name: str = "") -> Subset:
        """Filter a dataset to only include non-truncated examples.

        Assumes the dataset was constructed with truncate=True, so each item
        has a 'truncated' flag. Returns a Subset containing only items where
        truncated=False (i.e. the full sequence fit within max_length).
        """
        logger.info(f"Dropping truncated rows from dataset split {f'({name})' if name else '[no name]'}...")
        valid_indices = []
        for i in range(len(dataset)):
            item = dataset[i]
            if not item["truncated"]:
                valid_indices.append(i)
        label = f" ({name})" if name else ""
        logger.info(f"Filtered{label}: kept {len(valid_indices)}/{len(dataset)} examples that fit within max_length")
        return Subset(dataset, valid_indices)  # pyright: ignore[reportArgumentType]

    def _make_dataset(self) -> LoadedDatasets:
        logger.info(f"Loading training dataset of type {self.dataset_type} with config: {self.dataset_specific_config}")
        _should_filter = self.length_excession_behavior == LengthExcessionBehavior.FILTER
        # When filtering, we construct with truncate=True so __getitem__ doesn't
        # error, then drop truncated examples after construction.
        _truncate = _should_filter or self.length_excession_behavior == LengthExcessionBehavior.TRUNCATE

        datasets = self._make_dataset_splits(_truncate)

        if _should_filter:
            logger.info("Filtering datasets...")
            for split in datasets:
                datasets[split] = self._filter_by_length(datasets[split], split)  # pyright: ignore[reportArgumentType]

        return datasets

    def _make_dataset_splits(self, truncate: bool) -> LoadedDatasets:
        match self.dataset_type:
            case DatasetType.OPEN_THOUGHTS:
                from training.dataset.openthoughts.open_thoughts import (
                    OpenThoughtsDataset,
                )

                assert isinstance(self.dataset_specific_config, OpenThoughtsConfig)
                datasets: PredefinedDataset.LoadedDatasets = {
                    "train": OpenThoughtsDataset(
                        data_path=self.dataset_specific_config.data_path,
                        tokenizer=self.tokenizer,
                        max_length=self.dataset_specific_config.max_seq_length,
                        format=self.dataset_specific_config.data_format,
                        truncate=truncate,
                        loss_on_prompt=self.loss_on_prompt,
                    )
                }
                if self.dataset_specific_config.val_data_path is not None:
                    datasets["val"] = OpenThoughtsDataset(
                        data_path=self.dataset_specific_config.val_data_path,
                        tokenizer=self.tokenizer,
                        max_length=self.dataset_specific_config.max_seq_length,
                        format=self.dataset_specific_config.data_format,
                        truncate=truncate,
                        loss_on_prompt=self.loss_on_prompt,
                    )
                return datasets

            case DatasetType.FINEWEB_LMYSYSCHAT_MIXED:
                from training.dataset.MixedDataset import MixedDataset
                from training.dataset.gemma.fineweb import FineWebDataset
                from training.dataset.gemma.lmsys_chat import LMSYSChatDataset

                assert isinstance(self.dataset_specific_config, FineWebLMSysMixedConfig)
                pretraining_dataset = FineWebDataset(
                    data_path=self.dataset_specific_config.pretraining_datapath,
                    tokenizer=self.tokenizer,
                    max_length=self.dataset_specific_config.pretraining_max_seq_length,
                    truncate=truncate,
                )
                chat_dataset = LMSYSChatDataset(
                    data_path=self.dataset_specific_config.chat_conversations_datapath,
                    tokenizer=self.tokenizer,
                    max_length=self.dataset_specific_config.chat_max_seq_length if self.dataset_specific_config.chat_max_seq_length != "pretraining_max_seq_length" else self.dataset_specific_config.pretraining_max_seq_length,
                    truncate=truncate,
                )
                mixed = MixedDataset(
                    datasets=(pretraining_dataset, chat_dataset), weights=(0.5, 0.5)
                )
                logger.info(f"Created mixed dataset, rows={len(mixed)}")
                cached = CachedDataset(mixed)
                self._caches.append(cached)
                return {
                    "train": cached,
                }
            case _:
                raise ValueError(f"Unsupported dataset type: {self.dataset_type}")

    def _subsample_for_epoch(self, epoch: int) -> LoadedDatasets:
        """Subsample dataset_rows from each split using an epoch-specific seed."""
        assert self._loaded_datasets is not None
        if self.dataset_rows is None:
            return self._loaded_datasets

        subsampled: PredefinedDataset.LoadedDatasets = {}
        for split, ds in self._loaded_datasets.items():
            if len(ds) > self.dataset_rows:
                epoch_seed = self.dataloader_seed + epoch
                g = TorchGenerator().manual_seed(epoch_seed)
                indices = torch.randperm(len(ds), generator=g)[:self.dataset_rows].tolist()
                logger.info(
                    f"Epoch {epoch}: subsampling {self.dataset_rows}/{len(ds)} rows from '{split}' (seed={epoch_seed})"
                )
                subsampled[split] = Subset(ds, indices)  # pyright: ignore[reportArgumentType]
            else:
                subsampled[split] = ds
        return subsampled

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
        logger.info(f"Loading dataset for {self.dataset_type}, shuffling (seed={self.dataloader_seed})")
        dataloaders: PredefinedDataset.Dataloaders = {
            "train": DataLoader(
                datasets["train"], # pyright: ignore[reportArgumentType]
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=collate_with_tokenizer,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True,
                generator=generator,
            )
        }

        if "val" in datasets:
            dataloaders["val"] = DataLoader(
                datasets["val"], # pyright: ignore[reportArgumentType]
                batch_size=1,
                shuffle=False,
                collate_fn=collate_with_tokenizer,
                num_workers=4,
                pin_memory=True,
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
        Loads the dataset (if not already loaded), subsamples for the given epoch, and creates dataloaders.
        """
        self._load_dataset()
        self.clear_caches()
        epoch_datasets = self._subsample_for_epoch(epoch)
        return self._make_dataloader(epoch_datasets)

    def load_datasets_and_dataloaders(self) -> tuple[LoadedDatasets, Dataloaders]:
        """
        Loads the dataset (if not already loaded) and creates dataloaders for each split (without epoch-based subsampling). Returns a tuple of (loaded_datasets, dataloaders).
        """
        datasets = self._load_dataset()
        dataloaders = self._make_dataloader(datasets)
        return datasets, dataloaders
