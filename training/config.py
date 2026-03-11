"""Configuration management for sparse adaptation experiments."""

import yaml
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from helpers.log import logger

from pathlib import Path

from .dataset.openthoughts.types import DataFormat


class LengthExcessionBehavior(Enum):
    TRUNCATE = "truncate"
    ERROR = "error"
    """Throw if any sequences are over the max length."""
    FILTER = "filter"
    """Filter out any sequences that exceed the maximum length."""


@dataclass
class DatasetEntryConfig:
    """Configuration for a single dataset in the training mix."""
    type: str  # "fineweb", "lmsys_chat", "open_thoughts"
    datapath: str
    max_seq_length: int = 8192
    num_rows: int | None = None
    length_excession_behavior: LengthExcessionBehavior = LengthExcessionBehavior.TRUNCATE
    weight: float = 1.0
    # open_thoughts-specific
    data_format: DataFormat | None = None  # "tokenizer", "deepseek", "qwen"
    val_datapath: str | None = None


@dataclass
class TranscoderConfig:
    """Transcoder adapter configuration."""
    n_features: int = 8192
    dec_bias: bool = True  # Whether to include bias in decoder
    l1_weight: float | None = 0.001  # Weight for L1 regularization on features
    normalize_by_layer: bool = False  # Whether to normalize L1 weights by layer output norm
    schedule_l1_weight: bool = False  # Whether to linearly ramp L1 weight from 0 to target weight
    pre_activation_loss_weight: float = 0.0  # Weight for pre-activation loss (prevents dead features)


@dataclass
class BridgingConfig:
    """Bridging loss configuration for distillation with layer-wise compatibility."""
    reference_model_path: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    loss_type: str = "kl"  # "kl" (distillation to ref) or "lm" (language modeling)
    n_cutoffs: int = 1  # Number of layer cutoffs to sample per batch
    sampling: str = "uniform"  # "uniform" or list of fixed layer indices
    always_include_adapt_only: bool = True  # Always include clean adapter forward pass
    lambda_adapt: float = 1.0  # Weight on adapt-only loss (end-to-end KL/LM)
    lambda_bridge: float = 2.0  # Weight on bridging losses (mixed forward passes)
    lambda_nmse: float = 1.0  # Weight on activation matching loss (optional)
    backbone: str = "target"  # "base" or "target" (which model provides attn/embed/layernorm)


@dataclass
class DirectConfig:
    """Direct fine-tuning configuration (no bridging, LM loss only on response tokens)."""
    reference_model_path: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # Source for copied token embeddings
    copied_tokens: list[str] = field(default_factory=lambda: ["<think>", "</think>"])  # Tokens to add and copy from reference


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    # Model settings
    model_name: str = "Qwen/Qwen2.5-Math-7B"
    model_arch: str | None = None  # "qwen2", "gemma2", etc. Auto-detected from model_name if None

    # Transcoder configuration
    transcoder: TranscoderConfig | None = None

    # Training mode (exactly one of bridging or direct must be set)
    bridging: BridgingConfig | None = None
    direct: DirectConfig | None = None

    # Training hyperparameters
    learning_rate: float = 8e-4
    batch_size: int = 1
    micro_batch_size: int | None = None # not really doing gradient accumulation anymore, see note in PredefinedDataset's _make_dataloader function. If None, this will be set to batch_size.
    num_epochs: int = 1
    warmup_ratio: float = 0.05
    gradient_clip_norm: float = 1.0
    seed: int = 42

    # Data settings
    datasets: list[DatasetEntryConfig] = field(default_factory=lambda: [
        DatasetEntryConfig(
            type="open_thoughts",
            datapath="/nlp/scr/nathu/sparse-adaptation/data/openthoughts/stratified_n55000_t10000_s42_train.jsonl",
            data_format="deepseek",
            max_seq_length=10000,
            val_datapath="/nlp/scr/nathu/sparse-adaptation/data/openthoughts/stratified_n55000_t10000_s42_val.jsonl",
        )
    ])
    total_rows: int | None = None  # Total rows in the mixed dataset. If set, rows are allocated across datasets proportionally to weights.
    weight_by: str = "rows"  # "rows" or "tokens". If "tokens", weights represent desired token proportions (adjusts for avg sequence length).
    loss_on_prompt: bool = True

    val_frequency: int = 1000  # Run validation every N steps
    layerwise_val_frequency: int = 2000  # Run layerwise validation every N steps

    # Output settings (auto-computed)
    output_dir: str | None = None  # Will be computed from hyperparameters
    wandb_run_name: str | None = None  # Will be computed from hyperparams
    run_name_prefix: str | None = None  # Optional prefix for run name (e.g., "r1_distil_yolo") for use in wandb and the output dir.

    # WandB settings
    use_wandb: bool = True
    wandb_project: str = "sparse-adaptation"

    # Checkpoint settings
    save_checkpoints: bool = False  # If True, save periodic checkpoints (overwrites single 'latest' dir)
    checkpoint_frequency: int = 8192  # Save checkpoint every N steps

    # Hub settings
    push_to_hub: bool = False  # If True, push final model to Hugging Face Hub after training
    hub_org: str | None = None  # HF org/user to push to (defaults to authenticated user)

    # Debug settings
    debug_mode: bool = False  # If True, break after 50 steps for quick testing

    all_configs: list[str] = field(default_factory=list)
    """Don't set directly, used to store which config paths were loaded. Later configs override previous ones."""


def load_config(config_path: str | list[str], overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    """Load configuration from one or more YAML files.

    Args:
        config_path: Path (or list of paths) to YAML config file(s).
            When multiple paths are given, later files shallow-override earlier ones.
        overrides: Optional dict of overrides. Keys must be valid ExperimentConfig fields.
            If debug_mode is set to True, wandb is also disabled and _debug is appended to run_name_prefix.
            If any overrides are applied, wandb_run_name and output_dir are regenerated.
    """
    paths = config_path if isinstance(config_path, list) else [config_path]
    """The paths of all configs to load"""

    config_dict: dict = {}
    for n, p in enumerate(paths):
        with open(p, 'r') as f:
            layer = yaml.safe_load(f)
        if layer:
            config_dict = {**config_dict, **layer}
            if n != 0:
                logger.info(f'Overriding base config "{paths[0]}" with values from {p}: {layer}')
        else:
            raise ValueError(f"Config file {p} is empty or invalid.")

    # Handle nested configs
    adapter_configs = {}
    if 'transcoder' in config_dict:
        adapter_configs['transcoder'] = TranscoderConfig(**config_dict.pop('transcoder'))
    if 'bridging' in config_dict:
        adapter_configs['bridging'] = BridgingConfig(**config_dict.pop('bridging'))
    if 'direct' in config_dict:
        adapter_configs['direct'] = DirectConfig(**config_dict.pop('direct'))

    # Parse datasets list before creating ExperimentConfig
    if 'datasets' in config_dict:
        raw_datasets = config_dict.pop('datasets')
        parsed_datasets = []
        for entry in raw_datasets:
            if isinstance(entry, dict):
                if 'length_excession_behavior' in entry and isinstance(entry['length_excession_behavior'], str):
                    entry['length_excession_behavior'] = LengthExcessionBehavior(entry['length_excession_behavior'])
                if 'num_rows' in entry and entry['num_rows'] is not None:
                    entry['num_rows'] = int(entry['num_rows'])
                parsed_datasets.append(DatasetEntryConfig(**entry))
            else:
                parsed_datasets.append(entry)
        adapter_configs['datasets'] = parsed_datasets

    # Create main config with adapter configs
    config = ExperimentConfig(**config_dict, **adapter_configs)
    config.all_configs = paths

    # Apply overrides
    if overrides:
        valid_keys = set(ExperimentConfig.__dataclass_fields__.keys())
        # l1_weight is a known nested override (transcoder.l1_weight)
        NESTED_OVERRIDES = {
            "l1_weight": ("transcoder", "l1_weight"),
            "n_features": ("transcoder", "n_features"),
        }
        invalid_keys = set(overrides.keys()) - valid_keys - set(NESTED_OVERRIDES.keys())
        if invalid_keys:
            raise ValueError(
                f"Invalid override keys (not in ExperimentConfig): {invalid_keys}"
            )

        for key, value in overrides.items():
            if key in NESTED_OVERRIDES:
                parent_attr, child_attr = NESTED_OVERRIDES[key]
                parent = getattr(config, parent_attr, None)
                if parent is not None:
                    setattr(parent, child_attr, value)
                    logger.info(f"Override {parent_attr}.{child_attr}: {value}")
            else:
                setattr(config, key, value)
                logger.info(f"Override {key}: {value}")

        # debug_mode=True has side effects
        if overrides.get("debug_mode") is True:
            config.use_wandb = False
            logger.info("Debug mode enabled through override: wandb disabled")
            if config.run_name_prefix and not config.run_name_prefix.endswith("_debug"):
                config.run_name_prefix += "_debug"
                logger.info(f"Added _debug to run_name_prefix: '{config.run_name_prefix}'")

        # Force regeneration of computed fields
        config.wandb_run_name = None
        config.output_dir = None

    # Ensure numeric types are correct (YAML can load as strings)
    config.learning_rate = float(config.learning_rate)
    config.batch_size = int(config.batch_size)
    if config.micro_batch_size is not None:
        config.micro_batch_size = int(config.micro_batch_size)
        assert config.micro_batch_size <= config.batch_size, "micro_batch_size cannot be greater than batch_size"
        assert config.batch_size % config.micro_batch_size == 0, "batch_size must be divisible by micro_batch_size"

    if config.transcoder:
        # Convert transcoder weights to float if they exist
        if config.transcoder.l1_weight is not None:
            config.transcoder.l1_weight = float(config.transcoder.l1_weight)
        if config.transcoder.pre_activation_loss_weight is not None:
            config.transcoder.pre_activation_loss_weight = float(config.transcoder.pre_activation_loss_weight)

    # Resolve model_arch from model_name if not set
    if config.model_arch is None:
        from models import detect_architecture
        config.model_arch = detect_architecture(config.model_name)

    # Print a warning if there were any extra keys in the YAML that were not used in the config dataclass
    extra_keys = set(config_dict.keys()) - set(ExperimentConfig.__dataclass_fields__.keys())
    if extra_keys:
        logger.warning(f"The following keys in the config file were not recognized and will be ignored: {extra_keys}")

    # Auto-compute run name and output dir if not specified
    config = _finalize_config(config)
    return config


def _finalize_config(config: ExperimentConfig) -> ExperimentConfig:
    """Finalize config by computing run names and output directories."""
    import os
    from helpers.paths import SLURM_JOB_ID
    slurm_job_id = SLURM_JOB_ID

    # Build run name from hyperparameters
    if config.wandb_run_name is None:
        run_parts: list[str] = []

        # Use prefix if provided, otherwise generate default prefix
        if config.run_name_prefix:
            run_parts.append(config.run_name_prefix)
        else:
            model_size = _extract_model_size(config.model_name)
            run_parts.extend(["transcoder", model_size])

        # Add overlay config names (skip the first/base config)
        if len(config.all_configs) > 1:
            for cfg_path in config.all_configs[1:]:
                cfg_name = Path(cfg_path).stem
                run_parts.append(cfg_name)

        # Add transcoder params
        if config.transcoder:
            run_parts.append(f"tc{config.transcoder.n_features}")
            if getattr(config.transcoder, 'dec_bias', False):
                run_parts.append("decb")
            if config.transcoder.l1_weight:
                run_parts.append(f"l1w{config.transcoder.l1_weight}")
            if getattr(config.transcoder, 'normalize_by_layer', False):
                run_parts.append("norm")
            if getattr(config.transcoder, 'schedule_l1_weight', False):
                run_parts.append("sch")
            if getattr(config.transcoder, 'pre_activation_loss_weight', 0.0) > 0:
                run_parts.append(f"pre{config.transcoder.pre_activation_loss_weight}")

        # Add mode-specific params
        if config.bridging:
            backbone = getattr(config.bridging, 'backbone', 'base')
            run_parts.append(f"{backbone[:3]}bb")  # "basbb" or "tgtbb"
            run_parts.append(f"lb{config.bridging.lambda_bridge}")
            run_parts.append(f"ln{config.bridging.lambda_nmse}")
        elif config.direct:
            run_parts.append("direct")

        # Add data info
        if config.total_rows is not None:
            run_parts.append(f"dr{config.total_rows}")
        else:
            entries_with_rows = [e for e in config.datasets if e.num_rows is not None]
            if entries_with_rows:
                total_rows = sum(e.num_rows for e in entries_with_rows)  # type: ignore
                run_parts.append(f"dr{total_rows}")
            else:
                run_parts.append("drall")

        # Add training params
        run_parts.append(f"lr{config.learning_rate:.0e}")
        run_parts.append(f"bs{config.batch_size}")

        run_parts.append(f"sl{slurm_job_id}") # so we can cross-reference if needed

        config.wandb_run_name = "_".join(run_parts)

    # Build output directory
    if config.output_dir is None:
        from datetime import datetime
        user = os.environ.get("USER")
        if not user:
            raise RuntimeError("$USER environment variable is not set. Provide an output_dir in your config or set the USER environment variable so we know where to save checkpoints.")
        date_str = datetime.now().strftime("%Y-%m-%d_%H%M")
        config.output_dir = f"/nlp/scr/{user}/sparse-adaptation/checkpoints/{config.wandb_run_name}_{date_str}_{slurm_job_id}"
        logger.info(f"Checkpoints save directory: {config.output_dir}")

    return config


def _extract_model_size(model_name: str) -> str:
    """Extract model size from model name (e.g., '7B' from 'Qwen/Qwen2.5-7B-Instruct')."""
    import re
    # Look for patterns like 1.5B, 3B, 7B, 14B, etc.
    match = re.search(r'(\d+(?:\.\d+)?[BMK])', model_name, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return "unknown"


def apply_overrides(config: ExperimentConfig, overrides: dict[str, Any]) -> ExperimentConfig:
    """Apply command line overrides to config.

    Supports:
    - Direct params: learning_rate=1e-3
    - Nested params: transcoder.n_features=2048, bridging.lambda_bridge=2.0
    """
    for key, value in overrides.items():
        if '.' in key:
            section, param = key.split('.', 1)
            if section == 'transcoder' and config.transcoder:
                setattr(config.transcoder, param, value)
            elif section == 'bridging' and config.bridging:
                setattr(config.bridging, param, value)
            elif section == 'direct' and config.direct:
                setattr(config.direct, param, value)
            else:
                raise ValueError(f"Invalid override section: {section}")
        else:
            # Handle top-level params
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                raise ValueError(f"Invalid config parameter: {key}")

    return config


def save_config(config: ExperimentConfig, output_path: str):
    """Save configuration to YAML file."""
    config_dict = {}
    for field_name in config.__dataclass_fields__:
        val = getattr(config, field_name)
        if hasattr(val, '__dataclass_fields__'):
            config_dict[field_name] = val.__dict__
        elif isinstance(val, list) and val and hasattr(val[0], '__dataclass_fields__'):
            config_dict[field_name] = [item.__dict__ for item in val]
        else:
            config_dict[field_name] = val

    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, indent=2, default_flow_style=False)
