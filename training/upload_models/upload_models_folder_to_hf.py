"""Upload saved checkpoints to Hugging Face Hub.

Uploads one or more checkpoint directories (saved by train.py's save_checkpoint)
to HuggingFace. The HF repo name is derived from the checkpoint folder name,
not from the config — since folder names contain run-specific info (lr, bs, slurm
job ID, timestamp) while the config may be generic across runs.

Usage:
    # Upload all checkpoints in a folder:
    python -m training.upload_models.upload_models_folder_to_hf \\
        /nlp/scr/siddharth/sparse-adaptation/checkpoints \\
        --config training/configs/gemma2_2b.yaml \\
        --hub_org siddharthmb

    # Upload a single checkpoint:
    python -m training.upload_models.upload_models_folder_to_hf \\
        /nlp/scr/siddharth/sparse-adaptation/checkpoints/gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr10000_lr8e-04_bs4_sl14701501_2026-03-03_0039_14701501 \\
        --config training/configs/gemma2_2b.yaml

    # Dry run (just print what would be uploaded):
    python -m training.upload_models.upload_models_folder_to_hf \\
        /nlp/scr/siddharth/sparse-adaptation/checkpoints \\
        --dry_run
"""

import argparse
import os
import sys
from concurrent.futures import Future

from huggingface_hub import HfApi

from helpers.log import logger
from ..config import CHECKPOINT_CONFIG_FILENAME, load_config
from .hub import verify_hub_access, _build_model_card, _upload_training_config, truncate_repo_name, HUB_NAME_PREFIX


def find_checkpoints(path: str) -> list[str]:
    """Find checkpoint directories at the given path.

    If path is itself a checkpoint (contains model files), returns [path].
    Otherwise, returns all immediate subdirectories that look like checkpoints.
    Skips latest_step_* subdirectories (intermediate checkpoints).
    """
    if _is_checkpoint_dir(path):
        return [path]

    checkpoints = []
    for entry in sorted(os.listdir(path)):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path) and not entry.startswith("latest_step_"):
            if _is_checkpoint_dir(full_path):
                checkpoints.append(full_path)

    return checkpoints


def _is_checkpoint_dir(path: str) -> bool:
    """Check if a directory looks like a HuggingFace model checkpoint."""
    if not os.path.isdir(path):
        return False
    # A valid checkpoint has either model.safetensors or model.safetensors.index.json
    return (
        os.path.exists(os.path.join(path, "model.safetensors"))
        or os.path.exists(os.path.join(path, "model.safetensors.index.json"))
        or os.path.exists(os.path.join(path, "pytorch_model.bin"))
        or os.path.exists(os.path.join(path, "pytorch_model.bin.index.json"))
    )


def repo_id_from_checkpoint(checkpoint_dir: str, hub_org: str | None = None) -> str:
    """Build HF repo ID from a checkpoint directory name.

    The checkpoint folder name is used as the model name, prefixed with the
    standard project prefix. The trailing date/time/slurm suffix
    (e.g. _2026-03-03_0039_14701501) is stripped since the slurm ID already
    appears earlier as sl{id}, and HF repo names have a 96-char limit.

    Example:
        gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr10000_lr1e-04_bs4_sl14701501_2026-03-03_0039_14701501
        -> nathu0/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr10000_lr1e-04_bs4_sl14701501
    """
    folder_name = os.path.basename(os.path.normpath(checkpoint_dir))

    # Strip trailing _YYYY-MM-DD_HHMM_JOBID suffix (added by _finalize_config)
    folder_name = _strip_checkpoint_suffix(folder_name)

    model_name = truncate_repo_name(f"{HUB_NAME_PREFIX}.{folder_name}")

    if hub_org:
        return f"{hub_org}/{model_name}"

    api = HfApi()
    user = api.whoami()["name"]
    return f"{user}/{model_name}"


def repo_exists_and_nonempty(repo_id: str) -> bool:
    from huggingface_hub.errors import RepositoryNotFoundError
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id)
        if len(files) <= 1:
            logger.info(f"Repo {repo_id} already exists, but it has no content (files: {files})")
            return False
        else:
            # Repo exists and has content
            return True
    except RepositoryNotFoundError:
        # Does not exist
        return False

def _strip_checkpoint_suffix(name: str) -> str:
    """Strip the trailing _YYYY-MM-DD_HHMM_JOBID suffix from a checkpoint folder name.

    The output_dir format from _finalize_config is:
        {wandb_run_name}_{date}_{slurm_job_id}
    where date is YYYY-MM-DD_HHMM. This strips everything from the date onward.
    """
    import re
    # Match _YYYY-MM-DD_HHMM_SOMETHING at the end
    return re.sub(r"_\d{4}-\d{2}-\d{2}_\d{4}_\w+$", "", name)


def upload_checkpoint(
    checkpoint_dir: str,
    repo_id: str,
    config=None,
) -> Future | None:
    """Upload a single checkpoint directory to HuggingFace Hub.

    Returns a Future for the upload_folder call (runs asynchronously).
    The model card is pushed synchronously since it's small.

    If no config is provided, will try to find a config file within the checkpoint dir.
    """
    api = HfApi()

    # Create repo
    api.create_repo(repo_id, exist_ok=True)

    try:
        config_path = os.path.join(checkpoint_dir, CHECKPOINT_CONFIG_FILENAME)
        loaded_config = load_config(config_path)
        if config is not None:
            logger.warning(f"A config file was found at {loaded_config} but a config was also provided with the `--config` flag. Using the provided config, but the config file in the checkpoint directory is probably more accurate. Consider removing the `--config` flag.")
        else:
            config = loaded_config
    except FileNotFoundError:
        if config is None:
            logger.warning(f"Config file not found in {checkpoint_dir} and no config was provided. Model card will have not have metadata.")
        # Otherwise, we already had a config set, so we're fine

    # Add model card and training config (small, synchronous)
    if config is not None:
        _upload_training_config(api, config, repo_id)
        card = _build_model_card(config, repo_id)
        card.push_to_hub(repo_id)

    # Upload checkpoint files asynchronously, excluding intermediate checkpoint subdirs
    future = api.upload_folder(
        folder_path=checkpoint_dir,
        repo_id=repo_id,
        commit_message=f"Upload checkpoint from {os.path.basename(checkpoint_dir)}",
        ignore_patterns=["latest_step_*"],
        run_as_future=True,
    )

    logger.info(f"  Started upload: {os.path.basename(checkpoint_dir)} -> {repo_id}")
    return future


def main():
    from helpers.log import setup_logging
    setup_logging()
    from helpers.paths import PRODUCTS_DIR
    default_path = str(PRODUCTS_DIR / "checkpoints")
    parser = argparse.ArgumentParser(
        description="Upload saved checkpoints to Hugging Face Hub",
    )
    parser.add_argument(
        "path",
        help=f"Path to a checkpoint directory or a folder containing checkpoint directories. Default: {default_path}",
        default=default_path,
    )
    parser.add_argument(
        "--config",
        help=f'Path to the training config YAML (for model card metadata). Optional. If not given, will try to find a "{CHECKPOINT_CONFIG_FILENAME}" file within each checkpoint directory.',
        default=None,
    )
    parser.add_argument(
        "--hub_org",
        help="HF org/user to push to (defaults to authenticated user)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be uploaded without actually uploading",
    )
    parser.add_argument(
        "--exists",
        choices=["skip", "overwrite"],
        default=None,
        help="What to do if a repo already exists on the Hub. "
             "skip: skip it, overwrite: push anyway. "
             "If not set, error out when a repo already exists.",
    )
    args = parser.parse_args()

    # Find checkpoints
    checkpoints = find_checkpoints(args.path)
    if not checkpoints:
        logger.error(f"No checkpoints found at {args.path}")
        sys.exit(1)

    logger.info(f"Found {len(checkpoints)} checkpoint(s):")
    for cp in checkpoints:
        logger.info(f"  {os.path.basename(cp)}")

    # Load config if provided (for model card metadata)
    config = None
    if args.config:
        from training.config import load_config
        config = load_config(args.config)
        logger.info(f"Loaded config from {args.config}")

    # Build repo IDs and verify access on the first one
    repo_ids = [repo_id_from_checkpoint(cp, args.hub_org) for cp in checkpoints]

    if args.dry_run:
        logger.info("Dry run — would upload:")
        for cp, repo_id in zip(checkpoints, repo_ids):
            logger.info(f"  {os.path.basename(cp)} -> {repo_id}")
        return

    verify_hub_access(repo_ids[0])

    # Check for existing repos
    existing = [rid for rid in repo_ids if repo_exists_and_nonempty(rid)]
    if existing:
        if args.exists == "skip":
            logger.info(f"Skipping {len(existing)} existing repo(s):")
            for rid in existing:
                logger.info(f"  {rid}")
            # Filter out existing ones
            pairs = [(cp, rid) for cp, rid in zip(checkpoints, repo_ids) if rid not in existing]
            checkpoints = [cp for cp, _ in pairs]
            repo_ids = [rid for _, rid in pairs]
            if not checkpoints:
                logger.info("Nothing to upload.")
                return
        elif args.exists == "overwrite":
            logger.info(f"{len(existing)} repo(s) already exist and will be overwritten.")
        else:
            logger.error(f"{len(existing)} repo(s) already exist on the Hub:")
            for rid in existing:
                logger.info(f"  https://huggingface.co/{rid}")
            logger.error("Use --exists=skip to skip them or --exists=overwrite to overwrite.")
            sys.exit(1)

    # Launch all uploads concurrently
    futures: list[tuple[str, Future]] = []
    for cp, repo_id in zip(checkpoints, repo_ids):
        future = upload_checkpoint(cp, repo_id, config)
        if future is not None:
            futures.append((repo_id, future))

    # Wait for all uploads to complete
    logger.info(f"Waiting for {len(futures)} upload(s) to complete...")
    failed = []
    for repo_id, future in futures:
        try:
            future.result()
            logger.info(f"  Done: https://huggingface.co/{repo_id}")
        except Exception as e:
            logger.error(f"  FAILED: {repo_id}: {e}")
            failed.append(repo_id)

    if failed:
        logger.error(f"{len(failed)} upload(s) failed: {failed}")
        sys.exit(1)
    else:
        logger.info(f"All {len(futures)} upload(s) completed successfully.")


if __name__ == "__main__":
    main()
