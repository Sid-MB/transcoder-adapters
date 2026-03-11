"""Push trained model to Hugging Face Hub with metadata."""

import re
import tempfile

from huggingface_hub import HfApi, ModelCard, ModelCardData
from helpers.log import logger


HUB_NAME_PREFIX = "2026.TA"
MAX_REPO_NAME_LEN = 96


def push_to_hub(
    model,
    config,
    repo_id: str,
    wandb_url: str | None = None,
):
    """Push trained model to Hugging Face Hub with metadata.

    The tokenizer is not pushed — it comes from the base or reference model
    and is referenced in the model card.

    Args:
        model: The trained model.
        config: ExperimentConfig used for training.
        repo_id: Full repo ID (e.g., "nathu0/2026.TA.gemma2_2b_...").
        wandb_url: Optional W&B run URL to include in the model card.
    """
    api = HfApi()

    # Create the repo (no-op if it already exists)
    api.create_repo(repo_id, exist_ok=True)

    # Push model weights and config
    model.push_to_hub(repo_id)

    # Upload training config YAML
    _upload_training_config(api, config, repo_id)

    # Build and push model card with metadata
    full_name = f"{HUB_NAME_PREFIX}.{config.wandb_run_name}" if config.wandb_run_name else None
    card = _build_model_card(config, repo_id, full_name=full_name, wandb_url=wandb_url)
    card.push_to_hub(repo_id)

    logger.info(f"Model pushed to https://huggingface.co/{repo_id}")


def _upload_training_config(api: HfApi, config, repo_id: str):
    """Save and upload the training config YAML to the HF repo."""
    from training.config import save_config

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        save_config(config, f.name)
        tmp_path = f.name

    api.upload_file(
        path_or_fileobj=tmp_path,
        path_in_repo="training-config.yaml",
        repo_id=repo_id,
        commit_message="Add training config",
    )


def verify_hub_access(repo_id: str):
    """Verify the user has write access to the target Hub namespace.

    Call this before training starts so we fail fast rather than after
    hours of GPU time.
    """
    api = HfApi()
    try:
        user_info = api.whoami()
    except Exception as e:
        raise RuntimeError(
            "No valid Hugging Face token found. "
            "Run `huggingface-cli login` or set the HF_TOKEN environment variable."
        ) from e

    # Check that the token has write permission
    auth = user_info.get("auth", {})
    access_token = auth.get("accessToken", {})
    role = access_token.get("role", None)
    if role == "read":
        raise RuntimeError(
            "Your Hugging Face token has read-only access. "
            "Use a token with write permissions."
        )

    # Check namespace access: either user's own namespace or an org they belong to
    target_namespace = repo_id.split("/")[0]
    username = user_info["name"]
    orgs = [org["name"] for org in user_info.get("orgs", [])]

    if target_namespace != username and target_namespace not in orgs:
        raise RuntimeError(
            f"Cannot push to '{target_namespace}/' — you are logged in as '{username}' "
            f"and belong to orgs: {orgs}. "
            f"Set hub_org to your username or one of your orgs."
        )

    logger.info(f"Hub access verified: pushing to {repo_id}")


def truncate_repo_name(name: str, max_len: int = MAX_REPO_NAME_LEN) -> str:
    """Truncate a repo name to fit HF's limit while preserving key segments.

    Keeps the model identity prefix and slurm ID suffix (sl{id}), drops
    middle segments (training hyperparams) until it fits.

    Returns (truncated_name, was_truncated).
    """
    if len(name) <= max_len:
        return name

    parts = name.split("_")

    # Find slurm suffix index (last part matching sl{digits})
    slurm_idx = None
    for i in range(len(parts) - 1, -1, -1):
        if re.match(r"^sl\d+$", parts[i]):
            slurm_idx = i
            break

    if slurm_idx is None:
        # No slurm ID found — just truncate from the end
        return name[:max_len].rstrip("-._")

    prefix_parts = parts[:slurm_idx]
    suffix_parts = parts[slurm_idx:]  # includes sl{id} and anything after
    suffix = "_".join(suffix_parts)

    # Drop middle segments (from the end of prefix, working backwards) until it fits
    while prefix_parts and len("_".join(prefix_parts) + "_" + suffix) > max_len:
        prefix_parts.pop()

    if not prefix_parts:
        # Extreme case: even prefix alone is too long
        return (prefix_parts[0][:max_len - len(suffix) - 1] + "_" + suffix) if parts else name[:max_len]

    return "_".join(prefix_parts) + "_" + suffix


def build_hub_repo_id(config) -> str:
    """Build the Hub repo ID from config.

    Format: {hub_org}/2026.TA.{wandb_run_name}

    Falls back to the authenticated user's namespace if hub_org is not set.
    The model name is truncated to 96 chars (HF limit) while preserving
    the model identity prefix and slurm ID suffix.
    """
    full_name = f"{HUB_NAME_PREFIX}.{config.wandb_run_name}"
    model_name = truncate_repo_name(full_name)

    if config.hub_org:
        org = config.hub_org
    else:
        api = HfApi()
        org = api.whoami()["name"]

    return f"{org}/{model_name}"


def _build_model_card(
    config,
    repo_id: str,
    full_name: str | None = None,
    wandb_url: str | None = None,
) -> ModelCard:
    """Build a ModelCard with training metadata.

    Args:
        config: ExperimentConfig used for training.
        repo_id: The HF repo ID (may be truncated).
        full_name: The full untruncated model name, if it was truncated.
        wandb_url: Optional W&B run URL.
    """
    github_repo = "https://github.com/Sid-MB/transcoder-adapters"

    # Determine base model, training mode, and tokenizer source
    mode_prefix = "direct" if config.direct else "bridging"
    if config.direct:
        ref_model = config.direct.reference_model_path
    elif config.bridging:
        ref_model = config.bridging.reference_model_path
    else:
        ref_model = None

    # Tokenizer comes from reference model for bridging, base model for direct
    if config.bridging:
        tokenizer_source = config.bridging.reference_model_path
    else:
        tokenizer_source = config.model_name

    # Collect dataset identifiers
    datasets = _collect_dataset_ids(config)

    # Build tags
    tags = [
        "transcoder-adapters",
        "sparse-adaptation",
        mode_prefix,
    ]

    card_data = ModelCardData(
        base_model=config.model_name,
        tags=tags,
        datasets=datasets,
        library_name="transformers",
    )

    # Build markdown content
    display_name = repo_id.split("/")[-1]
    lines = [
        f"# {display_name}",
        "",
        f"Sparse transcoder adapter trained with **{mode_prefix}** mode.",
        "",
    ]

    # Show full name if it was truncated
    if full_name and full_name != display_name:
        lines.append(f"**Full name**: `{full_name}`")
        lines.append("")

    lines.extend([
        "## Model Details",
        "",
        f"- **Base model**: [{config.model_name}](https://huggingface.co/{config.model_name})",
    ])
    if ref_model:
        lines.append(f"- **Reference model**: [{ref_model}](https://huggingface.co/{ref_model})")
    lines.extend([
        f"- **Architecture**: {config.model_arch}",
        f"- **Training mode**: {mode_prefix}",
        f"- **Tokenizer**: [{tokenizer_source}](https://huggingface.co/{tokenizer_source})",
        "- **Training config**: [training-config.yaml](training-config.yaml)",
        f"- **GitHub**: [{github_repo}]({github_repo})",
    ])
    if wandb_url:
        lines.append(f"- **W&B run**: [{wandb_url}]({wandb_url})")

    # Transcoder details
    if config.transcoder:
        lines.extend([
            "",
            "## Transcoder Configuration",
            "",
            f"- **n_features**: {config.transcoder.n_features}",
            f"- **dec_bias**: {config.transcoder.dec_bias}",
            f"- **l1_weight**: {config.transcoder.l1_weight}",
        ])

    # Training details
    lines.extend([
        "",
        "## Training",
        "",
        f"- **Learning rate**: {config.learning_rate}",
        f"- **Batch size**: {config.batch_size}",
        f"- **Epochs**: {config.num_epochs}",
        f"- **Warmup ratio**: {config.warmup_ratio}",
    ])

    if config.bridging:
        lines.extend([
            f"- **Loss type**: {config.bridging.loss_type}",
            f"- **lambda_adapt**: {config.bridging.lambda_adapt}",
            f"- **lambda_bridge**: {config.bridging.lambda_bridge}",
            f"- **lambda_nmse**: {config.bridging.lambda_nmse}",
            f"- **n_cutoffs**: {config.bridging.n_cutoffs}",
            f"- **backbone**: {config.bridging.backbone}",
        ])

    # Datasets
    if datasets:
        lines.extend([
            "",
            "## Training Data",
            "",
        ])
        for ds in datasets:
            lines.append(f"- [{ds}](https://huggingface.co/datasets/{ds})")

    content = "\n".join(lines) + "\n"

    return ModelCard(content=f"---\n{card_data.to_yaml()}\n---\n{content}")


def _collect_dataset_ids(config) -> list[str]:
    """Extract HuggingFace dataset identifiers from config."""
    ids: list[str] = []

    for entry in config.datasets:
        for path in [entry.datapath, entry.val_datapath]:
            if not path:
                continue
            if path.startswith("hf://"):
                # Extract "org/dataset-name" from "hf://org/dataset-name/..."
                parts = path[len("hf://"):].split("/", 2)
                if len(parts) >= 2:
                    dataset_id = f"{parts[0]}/{parts[1]}"
                    if dataset_id not in ids:
                        ids.append(dataset_id)
            elif not path.startswith("/"):
                # Looks like a HF dataset ID (e.g., "science-of-finetuning/fineweb-1m-sample")
                if path not in ids:
                    ids.append(path)

    return ids
