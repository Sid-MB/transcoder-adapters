#!/usr/bin/env bash

# Ex:
# ./sh/annotate/annotate_contrastive_runs.sh --source_data_dir /path/to/source/run --target_data_dir /path/to/target/run
#
# Ex with separate output:
# ./sh/annotate/annotate_contrastive_runs.sh --source_data_dir /path/to/source/run --target_data_dir /path/to/target/run --annotations_file /tmp/contrastive_annotations.json --top_k 50
#
# By default, annotations are saved to <target_data_dir>/feature_annotations.json.
# Default behavior updates only contrastive_run tags/scores in place.
# Use --replace_all to archive any existing annotations file to <target_data_dir>/archive/ and write a fresh file.

LOG_DIR="logs/annotate_contrastive_runs"
LOG_PREFIX="local"
source sh/common_logging.sh
source sh/annotate/common.sh

ensure_writable_uv_cache
uv run python -m analysis.features.annotate.annotate_contrastive_runs "$@"

# After, restart visualizer to see changes.
