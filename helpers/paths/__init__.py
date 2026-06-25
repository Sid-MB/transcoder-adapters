"""Shared path constants for the project."""

import os
from pathlib import Path
from .output_path import generate_output_path

_LARGE_ARTIFACTS_DIR = os.environ.get("LARGE_ARTIFACTS_DIR")
if not _LARGE_ARTIFACTS_DIR:
    raise RuntimeError(
        "LARGE_ARTIFACTS_DIR is not set. It must point to a large, persistent volume "
        "where training results (checkpoints, feature data, logs, large .json/.pt files) "
        "are saved. Set it in your shell (see set-env.sh.example) and verify with check-env.py."
    )

# Root for all large output artifacts. Runs land under
# $LARGE_ARTIFACTS_DIR/transcoder-adapters/<category>/<run> (see generate_output_path).
PRODUCTS_DIR = Path(_LARGE_ARTIFACTS_DIR) / "transcoder-adapters"

SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID", "local")

generate_output_path = generate_output_path
