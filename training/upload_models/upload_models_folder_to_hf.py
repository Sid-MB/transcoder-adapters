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
        --hub_org nathu0

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

from huggingface_hub import HfApi

from .hub import verify_hub_access, _build_model_card


HUB_NAME_PREFIX = "2026.sparse-adaptation"


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

    The checkpoint folder name is used directly as the model name, prefixed
    with the standard project prefix.

    Example:
        gemma2_2b_tc8192_decb_..._sl14701501_2026-03-03_0039_14701501
        -> nathu0/2026.sparse-adaptation.gemma2_2b_tc8192_decb_..._sl14701501_2026-03-03_0039_14701501
    """
    folder_name = os.path.basename(os.path.normpath(checkpoint_dir))
    model_name = f"{HUB_NAME_PREFIX}.{folder_name}"

    if hub_org:
        return f"{hub_org}/{model_name}"

    api = HfApi()
    user = api.whoami()["name"]
    return f"{user}/{model_name}"


def upload_checkpoint(
    checkpoint_dir: str,
    repo_id: str,
    config=None,
):
    """Upload a single checkpoint directory to HuggingFace Hub."""
    api = HfApi()

    # Create repo
    api.create_repo(repo_id, exist_ok=True)

    # Upload all files in the checkpoint directory
    api.upload_folder(
        folder_path=checkpoint_dir,
        repo_id=repo_id,
        commit_message=f"Upload checkpoint from {os.path.basename(checkpoint_dir)}",
    )

    # Add model card if config is available
    if config is not None:
        card = _build_model_card(config, repo_id)
        card.push_to_hub(repo_id)

    print(f"  Uploaded to https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload saved checkpoints to Hugging Face Hub",
    )
    parser.add_argument(
        "path",
        help="Path to a checkpoint directory or a folder containing checkpoint directories",
    )
    parser.add_argument(
        "--config",
        help="Path to the training config YAML (for model card metadata). Optional.",
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
    args = parser.parse_args()

    # Find checkpoints
    checkpoints = find_checkpoints(args.path)
    if not checkpoints:
        print(f"No checkpoints found at {args.path}")
        sys.exit(1)

    print(f"Found {len(checkpoints)} checkpoint(s):")
    for cp in checkpoints:
        print(f"  {os.path.basename(cp)}")

    # Load config if provided (for model card metadata)
    config = None
    if args.config:
        from training.config import load_config
        config = load_config(args.config)
        print(f"Loaded config from {args.config}")

    # Build repo IDs and verify access on the first one
    repo_ids = [repo_id_from_checkpoint(cp, args.hub_org) for cp in checkpoints]

    if args.dry_run:
        print("\nDry run — would upload:")
        for cp, repo_id in zip(checkpoints, repo_ids):
            print(f"  {os.path.basename(cp)} -> {repo_id}")
        return

    verify_hub_access(repo_ids[0])

    # Upload
    print()
    for cp, repo_id in zip(checkpoints, repo_ids):
        print(f"Uploading {os.path.basename(cp)} -> {repo_id}")
        upload_checkpoint(cp, repo_id, config)

    print(f"\nDone! Uploaded {len(checkpoints)} checkpoint(s).")


if __name__ == "__main__":
    main()
