#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'USAGE'
Usage: find_job_logs.sh <job-id>

Find Slurm log files recursively under logs/ by matching the job ID suffix.
Expected names look like:
  <DATE>_<TIME>_<SLURM_JOB_ID>.out
  <DATE>_<TIME>_<SLURM_JOB_ID>.err

Example:
  find_job_logs.sh 123456
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 2
fi

job_id="$1"
if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
  echo "Error: job ID must be numeric, got: $job_id" >&2
  exit 2
fi

set +e
output=$(
  "$SCRIPT_DIR/cluster_exec.sh" \
    find logs -type f '(' -name "*_${job_id}.out" -o -name "*_${job_id}.err" ')' -print 2>&1
)
status=$?
set -e

if [[ "$status" -ne 0 ]]; then
  printf '%s\n' "$output" >&2
  exit "$status"
fi

if [[ -z "$output" ]]; then
  echo "No logs found for job_id=$job_id" >&2
  exit 1
fi

printf '%s\n' "$output" | sort
