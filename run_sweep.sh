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
#   ./run_sweep.sh --config training/configs/gemma2_2b.yaml --sweep training/configs/sweeps/lr_totalrows.yaml
#   ./run_sweep.sh --config training/configs/gemma2_2b.yaml --sweep training/configs/sweeps/lr_totalrows.yaml --sweep_count 20

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
NUM_GPUS=${NUM_GPUS:-1}

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

# Create the sweep (prints sweep ID to stdout)
SWEEP_ID=$(uv run python -c "
import yaml, wandb
with open('$SWEEP_FILE') as f:
    sc = yaml.safe_load(f)
print(wandb.sweep(sweep=sc, project='sparse-adaptation'))
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
