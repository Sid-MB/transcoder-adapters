"""Push trained model to Hugging Face Hub with metadata."""

from huggingface_hub import HfApi, ModelCard, ModelCardData


def push_to_hub(
    model,
    config,
    repo_id: str,
):
    """Push trained model to Hugging Face Hub with metadata.

    The tokenizer is not pushed — it comes from the base or reference model
    and is referenced in the model card.

    Args:
        model: The trained model.
        config: ExperimentConfig used for training.
        repo_id: Full repo ID (e.g., "nathu0/2026.sparse-adaptation.bridging_transcoder_7B_...").
    """
    api = HfApi()

    # Create the repo (no-op if it already exists)
    api.create_repo(repo_id, exist_ok=True)

    # Push model weights and config
    model.push_to_hub(repo_id)

    # Build and push model card with metadata
    card = _build_model_card(config, repo_id)
    card.push_to_hub(repo_id)

    print(f"Model pushed to https://huggingface.co/{repo_id}")


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

    print(f"Hub access verified: pushing to {repo_id}")


def build_hub_repo_id(config) -> str:
    """Build the Hub repo ID from config.

    Format: {hub_org}/2026.{wandb_project}.{mode_prefix}_{wandb_run_name}

    Falls back to the authenticated user's namespace if hub_org is not set.
    """
    mode_prefix = "direct" if config.direct else "bridging"
    model_name = f"2026.{config.wandb_project}.{mode_prefix}_{config.wandb_run_name}"

    if config.hub_org:
        return f"{config.hub_org}/{model_name}"

    # Fall back to authenticated user
    api = HfApi()
    user = api.whoami()["name"]
    return f"{user}/{model_name}"


def _build_model_card(config, repo_id: str) -> ModelCard:
    """Build a ModelCard with training metadata."""
    github_repo = "https://github.com/nathanhu0/transcoder-adapters"

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
    lines = [
        f"# {repo_id.split('/')[-1]}",
        "",
        f"Sparse transcoder adapter trained with **{mode_prefix}** mode.",
        "",
        "## Model Details",
        "",
        f"- **Base model**: [{config.model_name}](https://huggingface.co/{config.model_name})",
    ]
    if ref_model:
        lines.append(f"- **Reference model**: [{ref_model}](https://huggingface.co/{ref_model})")
    lines.extend([
        f"- **Architecture**: {config.model_arch}",
        f"- **Training mode**: {mode_prefix}",
        f"- **Tokenizer**: [{tokenizer_source}](https://huggingface.co/{tokenizer_source})",
        f"- **GitHub**: [{github_repo}]({github_repo})",
    ])

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
    from .dataset.openthoughts.config import OpenThoughtsConfig
    from .dataset.gemma.config import FineWebLMSysMixedConfig

    ids: list[str] = []
    ds = config.dataset

    if isinstance(ds, OpenThoughtsConfig):
        # data_path may be like "hf://nathu0/transcoder-adapters-openthoughts3-stratified-55k/data/train.jsonl"
        for path in [ds.data_path, getattr(ds, "val_data_path", None)]:
            if path and path.startswith("hf://"):
                # Extract "org/dataset-name" from "hf://org/dataset-name/..."
                parts = path[len("hf://"):].split("/", 2)
                if len(parts) >= 2:
                    dataset_id = f"{parts[0]}/{parts[1]}"
                    if dataset_id not in ids:
                        ids.append(dataset_id)
    elif isinstance(ds, FineWebLMSysMixedConfig):
        for path in [ds.pretraining_datapath, ds.chat_conversations_datapath]:
            if path and not path.startswith("/"):
                # Looks like a HF dataset ID (e.g., "science-of-finetuning/fineweb-1m-sample")
                if path not in ids:
                    ids.append(path)

    return ids
