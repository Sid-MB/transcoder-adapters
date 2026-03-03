from enum import Enum
from functools import partial
from typing import Literal
from torch.utils.data import DataLoader
from torch import Generator as TorchGenerator

from .collate import collate_fn
from .types import DatasetItem, SizedDataset
from .gemma.config import FineWebLMSysMixedConfig
from .datasetspecific_config import DatasetSpecificConfig, DatasetType
from .openthoughts.config import OpenThoughtsConfig


class LengthExcessionBehavior(Enum):
    TRUNCATE = "truncate"
    ERROR = "error"


DatasetSplits = Literal["train", "val"]
LoadedDatasets = dict[DatasetSplits, SizedDataset[DatasetItem]]
Dataloaders = dict[DatasetSplits, DataLoader[DatasetItem]]


class PredefinedDataset:
    """Loads a dataset and creates DataLoaders for training."""

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
    ):
        self.dataset_type = dataset_type
        self.tokenizer = tokenizer
        self.length_excession_behavior = length_excession_behavior
        self.loss_on_prompt = loss_on_prompt
        self.dataset_specific_config = dataset_specific_config
        self.batch_size = batch_size
        self.dataloader_seed = dataloader_seed

        if dataset_specific_config is not None:
            assert dataset_specific_config.dataset_type == dataset_type, (
                f"Config type mismatch: config is for {dataset_specific_config.dataset_type}, "
                f"but dataset_type is {dataset_type}"
            )

        self._loaded_datasets: LoadedDatasets | None = None

    def load_datasets_and_dataloaders(self) -> tuple[LoadedDatasets, Dataloaders]:
        """Load the dataset (if not already loaded) and create dataloaders."""
        if self._loaded_datasets is None:
            self._loaded_datasets = self._make_dataset()
        return self._loaded_datasets, self._make_dataloaders()

    def _make_dataset(self) -> LoadedDatasets:
        print(f"Loading dataset: {self.dataset_type} | config: {self.dataset_specific_config}")
        truncate = self.length_excession_behavior == LengthExcessionBehavior.TRUNCATE

        match self.dataset_type:
            case DatasetType.OPEN_THOUGHTS:
                from training.dataset.openthoughts.open_thoughts import OpenThoughtsDataset

                assert isinstance(self.dataset_specific_config, OpenThoughtsConfig)
                cfg = self.dataset_specific_config
                datasets: LoadedDatasets = {
                    "train": OpenThoughtsDataset(
                        data_path=cfg.data_path,
                        tokenizer=self.tokenizer,
                        max_length=cfg.max_seq_length,
                        format=cfg.data_format,
                        truncate=truncate,
                        loss_on_prompt=self.loss_on_prompt,
                    )
                }
                if cfg.val_data_path is not None:
                    datasets["val"] = OpenThoughtsDataset(
                        data_path=cfg.val_data_path,
                        tokenizer=self.tokenizer,
                        max_length=cfg.max_seq_length,
                        format=cfg.data_format,
                        truncate=truncate,
                        loss_on_prompt=self.loss_on_prompt,
                    )
                return datasets

            case DatasetType.FINEWEB_LMYSYSCHAT_MIXED:
                from training.dataset.gemma.mixed import MixedDataset
                from training.dataset.gemma.fineweb import FineWebDataset
                from training.dataset.gemma.lmsys_chat import LMSYSChatDataset

                assert isinstance(self.dataset_specific_config, FineWebLMSysMixedConfig)
                cfg = self.dataset_specific_config
                chat_max_len = (
                    cfg.pretraining_max_seq_length
                    if cfg.chat_max_seq_length == "pretraining_max_seq_length"
                    else cfg.chat_max_seq_length
                )
                return {
                    "train": MixedDataset(
                        datasets=(
                            FineWebDataset(
                                data_path=cfg.pretraining_datapath,
                                tokenizer=self.tokenizer,
                                max_length=cfg.pretraining_max_seq_length,
                                truncate=truncate,
                            ),
                            LMSYSChatDataset(
                                data_path=cfg.chat_conversations_datapath,
                                tokenizer=self.tokenizer,
                                max_length=chat_max_len,
                                truncate=truncate,
                            ),
                        ),
                        weights=(0.5, 0.5),
                    ),
                }

            case _:
                raise ValueError(f"Unsupported dataset type: {self.dataset_type}")

    def _make_dataloaders(self) -> Dataloaders:
        assert self._loaded_datasets is not None and "train" in self._loaded_datasets

        collate = partial(collate_fn, pad_token_id=self.tokenizer.pad_token_id)
        generator = TorchGenerator()
        generator.manual_seed(self.dataloader_seed)

        dataloaders: Dataloaders = {
            "train": DataLoader(
                self._loaded_datasets["train"], # pyright: ignore[reportArgumentType]
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=collate,
                num_workers=2,
                pin_memory=True,
                persistent_workers=True,
                generator=generator,
            )
        }

        if "val" in self._loaded_datasets:
            dataloaders["val"] = DataLoader(
                self._loaded_datasets["val"], # pyright: ignore[reportArgumentType]
                batch_size=1,
                shuffle=False,
                collate_fn=collate,
                num_workers=2,
                pin_memory=True,
                persistent_workers=True,
            )

        return dataloaders
