"""Shared path constants for the project."""

import os
from pathlib import Path
from .output_path import generate_output_path

USER = os.environ["USER"]

PRODUCTS_DIR = Path(f"/nlp/scr/{USER}/sparse-adaptation")

SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID", "local")

generate_output_path = generate_output_path
