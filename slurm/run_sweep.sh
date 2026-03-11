#!/usr/bin/env bash
set -euo pipefail

# Launch wandb sweep agents across all available GPUs.
#
# Usage:
#   ./run_sweep.sh --config <base_config> --sweep <sweep_config> [extra args...]
#
# With 1 GPU, runs a single agent. With multiple GPUs, creates the sweep first,
# then launches one agent per GPU using --sweep_id.
#
# Examples:
#   ./run_sweep.sh --config training/configs/gemma2_2b.yaml --sweep training/configs/sweeps/lr.yaml
#   ./run_sweep.sh --config training/configs/gemma2_2b.yaml --sweep training/configs/sweeps/lr.yaml --sweep_count 20

if command -v nvidia-smi >/dev/null 2>&1; then
  NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
else
  NUM_GPUS=1
fi

echo "Detected $NUM_GPUS GPU(s)"

if [ "$NUM_GPUS" -le 1 ]; then
  # Single GPU: just run directly, train.py handles sweep creation + agent
  uv run python -m training.train "$@"
  exit 0
fi

# Multiple GPUs: create sweep first, then launch agents with --sweep_id
# Extract --sweep value from args to create the sweep
SWEEP_FILE=""
REMAINING_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sweep)
      SWEEP_FILE="$2"
      shift 2
      ;;
    *)
      REMAINING_ARGS+=("$1")
      shift
      ;;
  esac
done

if [ -z "$SWEEP_FILE" ]; then
  echo "Error: --sweep <config.yaml> is required"
  exit 1
fi

# Extract --config values from remaining args to read wandb_project
CONFIG_FILES=()
for i in "${!REMAINING_ARGS[@]}"; do
  if [ "${REMAINING_ARGS[$i]}" = "--config" ]; then
    j=$((i + 1))
    while [ $j -lt ${#REMAINING_ARGS[@]} ] && [[ "${REMAINING_ARGS[$j]}" != --* ]]; do
      CONFIG_FILES+=("${REMAINING_ARGS[$j]}")
      j=$((j + 1))
    done
  fi
done

# Create the sweep (prints sweep ID to stdout)
CONFIG_ARGS=$(printf "'%s'," "${CONFIG_FILES[@]}")
SWEEP_ID=$(uv run python -c "
import yaml, sys
sys.path.insert(0, '.')
from training.config import load_config
with open('$SWEEP_FILE') as f:
    sc = yaml.safe_load(f)
config = load_config([${CONFIG_ARGS%,}])
import wandb
print(wandb.sweep(sweep=sc, project=config.wandb_project))
")

echo "Created sweep: $SWEEP_ID"
echo "Launching $NUM_GPUS agents..."

PIDS=()
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
  echo "  GPU $gpu_id: starting agent"
  CUDA_VISIBLE_DEVICES=$gpu_id uv run python -m training.train \
    --sweep_id "$SWEEP_ID" "${REMAINING_ARGS[@]}" &
  PIDS+=($!)
done

echo "All agents launched. Waiting..."
wait "${PIDS[@]}"
echo "All agents finished."
