#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'USAGE'
Usage: submit_job.sh [--wait] <sh/slurm_batch_*.sh|slurm_batch_*.sh> [args...]

Submit one of this repo's Slurm wrapper scripts and print the captured job ID.

Options:
  --wait    Set SBATCH_WAIT=1 so sbatch waits for the job to terminate.

Examples:
  submit_job.sh sh/slurm_batch_train.sh --config training/configs/gemma2_2b.yaml
  submit_job.sh --wait slurm_batch_train.sh --config training/configs/gemma2_2b.yaml
USAGE
}

wait_for_job=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --wait)
      wait_for_job=1
      shift
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -eq 0 ]]; then
  usage >&2
  exit 2
fi

job_script="$1"
shift

if [[ "$job_script" != */* ]]; then
  job_script="sh/$job_script"
fi

if [[ "$job_script" != sh/slurm_batch_*.sh && "$job_script" != ./sh/slurm_batch_*.sh ]]; then
  echo "Error: expected a sh/slurm_batch_*.sh script, got: $job_script" >&2
  exit 2
fi

cmd=()
if [[ "$wait_for_job" -eq 1 ]]; then
  cmd+=(env SBATCH_WAIT=1)
fi
cmd+=("$job_script" "$@")

set +e
output=$("$SCRIPT_DIR/cluster_exec.sh" "${cmd[@]}" 2>&1)
status=$?
set -e

printf '%s\n' "$output"

job_id=$(printf '%s\n' "$output" | sed -nE 's/.*Submitted batch job ([0-9]+).*/\1/p' | tail -n 1)
if [[ -n "$job_id" ]]; then
  echo "job_id=$job_id"
fi

if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

if [[ -z "$job_id" ]]; then
  echo "Warning: could not parse a Slurm job ID from submission output." >&2
  exit 1
fi
