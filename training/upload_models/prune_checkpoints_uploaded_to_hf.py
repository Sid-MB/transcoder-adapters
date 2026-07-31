"""Delete local checkpoints that are already uploaded to Hugging Face.

By default this scans:
    $LARGE_ARTIFACTS_DIR/transcoder-adapters/checkpoints/

For each checkpoint directory, the script derives the Hugging Face repo ID using
the same code path as train.py when a saved training config is available. Older
checkpoints without a saved config fall back to the upload script's checkpoint
folder naming logic.
"""

# uv run python -m training.upload_models.prune_checkpoints_uploaded_to_hf

import argparse
import shutil
import sys
from dataclasses import replace
from pathlib import Path

from helpers.log import logger

# None => fall back to the logged-in Hugging Face user's namespace (via whoami).
DEFAULT_HUB_ORG = None


def _default_checkpoint_root() -> Path:
    from helpers.paths import PRODUCTS_DIR
    return PRODUCTS_DIR / "checkpoints"


def _repo_id_for_checkpoint(checkpoint_dir: str, hub_org: str) -> str:
    """Resolve the HF repo ID for a local checkpoint directory."""
    from training.config import CHECKPOINT_CONFIG_FILENAME, load_config
    from training.upload_models.hub import build_hub_repo_id
    from training.upload_models.upload_models_folder_to_hf import repo_id_from_checkpoint

    config_path = Path(checkpoint_dir) / CHECKPOINT_CONFIG_FILENAME
    if config_path.exists():
        try:
            config = load_config(str(config_path))
            config = replace(config, hub_org=hub_org)
            return build_hub_repo_id(config)
        except Exception as exc:
            logger.warning(
                f"Could not derive repo ID from {config_path}; falling back to checkpoint folder name: {exc}"
            )

    return repo_id_from_checkpoint(checkpoint_dir, hub_org)


def _format_table(rows: list[tuple[str, str]]) -> list[str]:
    if not rows:
        return []

    local_header = "Local directory"
    hf_header = "Hugging Face URL"
    local_width = max(len(local_header), *(len(local) for local, _ in rows))
    hf_width = max(len(hf_header), *(len(url) for _, url in rows))
    separator = f"{'-' * local_width}  {'-' * hf_width}"

    lines = [
        f"{local_header:<{local_width}}  {hf_header:<{hf_width}}",
        separator,
    ]
    for local, url in rows:
        lines.append(f"{local:<{local_width}}  {url:<{hf_width}}")
    return lines


def _log_table(title: str, rows: list[tuple[str, str]]) -> None:
    if not rows:
        return
    logger.info(title)
    for line in _format_table(rows):
        logger.info(line)


def _confirm_deletion(count: int) -> bool:
    response = input(
        f"Delete these {count} local checkpoint director{'y' if count == 1 else 'ies'}? "
        "Type 'delete' to confirm: "
    )
    return response.strip() == "delete"


def main() -> None:
    from helpers.log import setup_logging

    setup_logging()

    default_path = "$LARGE_ARTIFACTS_DIR/transcoder-adapters/checkpoints"
    parser = argparse.ArgumentParser(
        description="Delete local checkpoint directories that already exist on Hugging Face.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help=(
            "Path to a checkpoint directory or a folder containing checkpoint directories. "
            f"Default: {default_path}"
        ),
    )
    parser.add_argument(
        "--hub_org",
        default=DEFAULT_HUB_ORG,
        help="HF org/user namespace to check. Default: your logged-in Hugging Face user.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be deleted without deleting anything.",
    )
    args = parser.parse_args()
    if args.path is None:
        try:
            args.path = str(_default_checkpoint_root())
        except RuntimeError as exc:
            parser.error(str(exc))

    from training.upload_models.upload_models_folder_to_hf import (
        find_checkpoints,
        repo_exists_and_nonempty,
    )

    checkpoints = find_checkpoints(args.path)
    if not checkpoints:
        logger.error(f"No checkpoints found at {args.path}")
        sys.exit(1)

    logger.info(f"Found {len(checkpoints)} checkpoint(s) under {args.path}")

    uploaded: list[tuple[str, str]] = []
    missing: list[tuple[str, str]] = []
    for checkpoint_dir in checkpoints:
        repo_id = _repo_id_for_checkpoint(checkpoint_dir, args.hub_org)
        hf_url = f"https://huggingface.co/{repo_id}"
        if repo_exists_and_nonempty(repo_id):
            uploaded.append((checkpoint_dir, hf_url))
        else:
            missing.append((checkpoint_dir, hf_url))

    _log_table(
        f"{len(uploaded)} checkpoint(s) already uploaded and eligible for local deletion:",
        uploaded,
    )
    _log_table(
        f"{len(missing)} checkpoint(s) not found on Hugging Face; keeping locally:",
        missing,
    )

    if not uploaded:
        logger.info("No uploaded checkpoints to prune.")
        return

    if args.dry_run:
        logger.info("Dry run: no local directories were deleted.")
        return

    if not _confirm_deletion(len(uploaded)):
        logger.info("Deletion cancelled.")
        return

    failures: list[tuple[str, str]] = []
    for checkpoint_dir, hf_url in uploaded:
        try:
            shutil.rmtree(checkpoint_dir)
            logger.info(f"Deleted {checkpoint_dir} ({hf_url})")
        except Exception as exc:
            logger.error(f"Failed to delete {checkpoint_dir} ({hf_url}): {exc}")
            failures.append((checkpoint_dir, hf_url))

    if failures:
        logger.error(f"Failed to delete {len(failures)} checkpoint(s).")
        sys.exit(1)

    logger.info(f"Deleted {len(uploaded)} local checkpoint director{'y' if len(uploaded) == 1 else 'ies'}.")


if __name__ == "__main__":
    main()
