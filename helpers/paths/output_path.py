from datetime import datetime
from pathlib import Path

def generate_output_path(category: str, specific_run_description: str | None = None, consistent: bool = False) -> Path:
	"""
	
	Args:
		category: The folder name under PRODUCTS_DIR, same for all similar runs. High-level category for the output (e.g. "attribution", "feature_collection")
		specific_run_description: Optional short description to include in the folder name for this run, to distinguish it from other runs. If None, will just use timestamp and SLURM_JOB_ID. 
		consistent: If True, will generate the same path for the same specific_run_description. Excludes slurm job id and timestamp when creating the path.
	"""
	from helpers.paths import PRODUCTS_DIR, SLURM_JOB_ID
	# Truncate model path: take last component
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	foldername_components = []
	if specific_run_description:
		foldername_components.append(specific_run_description)
	if not consistent:
		foldername_components += [timestamp, SLURM_JOB_ID]
	foldername = '_'.join(foldername_components)
	return PRODUCTS_DIR / category / foldername

