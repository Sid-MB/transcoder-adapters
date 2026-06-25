#!/usr/bin/env -S uv run --no-sync python
"""Verify the environment is ready to use this repo.

Run from the repo root, either:
    uv run python check-env.py
    ./check-env.py            # uses the uv shebang above

Checks (exits non-zero on the first problem found):
  - LARGE_ARTIFACTS_DIR is set, exists, and is writable
  - TMPDIR is set and exists
  - uv is installed
  - Hugging Face is logged in
  - Weights & Biases is logged in

See set-env.sh.example for what to set and how.
"""

import os
import shutil
import sys
from pathlib import Path


def fail(msg: str) -> None:
    print(f"❌ {msg}")
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"✅ {msg}")


def check_dir_var(name: str, *, writable: bool = False) -> None:
    """Assert an env var names an existing directory (optionally writable)."""
    value = os.environ.get(name)
    if not value:
        fail(f"{name} is not set (see set-env.sh.example).")
    if not Path(value).is_dir():
        fail(f"{name} ({value}) does not exist.")
    if writable and not os.access(value, os.W_OK):
        fail(f"{name} ({value}) is not writable.")
    ok(f"{name}={value}")


def main() -> None:
    # Required environment variables.
    check_dir_var("LARGE_ARTIFACTS_DIR", writable=True)
    check_dir_var("TMPDIR")

    # uv.
    uv_path = shutil.which("uv")
    if not uv_path:
        fail("uv is not installed — see https://docs.astral.sh/uv/.")
    ok(f"uv: {uv_path}")

    # Hugging Face login.
    try:
        from huggingface_hub import HfApi

        hf_user = HfApi().whoami()["name"]
    except Exception:
        fail("Hugging Face is not logged in — run: huggingface-cli login")
    ok(f"Hugging Face logged in as {hf_user}")

    # Weights & Biases login.
    try:
        import wandb

        wandb.Api()
    except Exception:
        fail("Weights & Biases is not logged in — run: wandb login")
    ok("Weights & Biases is logged in")

    print("✅ Environment looks good.")


if __name__ == "__main__":
    main()
