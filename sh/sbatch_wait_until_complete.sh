#!/usr/bin/env bash
# Given a Slurm job ID, wait until the job is complete, then print its exit code.

set -euo pipefail

usage() {
  echo "Usage: $0 <slurm-job-id>" >&2
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

JOB_ID="$1"
POLL_SECONDS="${SBATCH_WAIT_POLL_SECONDS:-10}"
ACCOUNTING_TIMEOUT_SECONDS="${SBATCH_WAIT_ACCOUNTING_TIMEOUT_SECONDS:-18000}"

if ! [[ "$JOB_ID" =~ ^[0-9]+([._][0-9]+)?$ ]]; then
  echo "Expected a Slurm job ID, got: $JOB_ID" >&2
  usage
  exit 2
fi

if ! [[ "$POLL_SECONDS" =~ ^[0-9]+$ ]] || [[ "$POLL_SECONDS" -eq 0 ]]; then
  echo "SBATCH_WAIT_POLL_SECONDS must be a positive integer" >&2
  exit 2
fi

if ! [[ "$ACCOUNTING_TIMEOUT_SECONDS" =~ ^[0-9]+$ ]]; then
  echo "SBATCH_WAIT_ACCOUNTING_TIMEOUT_SECONDS must be a non-negative integer" >&2
  exit 2
fi

if ! command -v squeue >/dev/null 2>&1; then
  echo "squeue is not available; run this on a Slurm login node" >&2
  exit 127
fi

if ! command -v sacct >/dev/null 2>&1; then
  echo "sacct is not available; run this on a Slurm login node" >&2
  exit 127
fi

while squeue --noheader --jobs="$JOB_ID" | grep -q .; do
  sleep "$POLL_SECONDS"
done

deadline=$((SECONDS + ACCOUNTING_TIMEOUT_SECONDS))

while true; do
  exit_code="$(
    sacct \
      --jobs="$JOB_ID" \
      --allocations \
      --noheader \
      --parsable2 \
      --format=ExitCode |
      awk 'NF { print $1; exit }'
  )"

  if [[ -n "$exit_code" ]]; then
    echo "$exit_code"

    status="${exit_code%%:*}"
    if [[ "$status" =~ ^[0-9]+$ ]] && [[ "$status" -le 255 ]]; then
      exit "$status"
    fi

    exit 1
  fi

  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for sacct to report exit code for job $JOB_ID" >&2
    exit 1
  fi

  sleep "$POLL_SECONDS"
done
