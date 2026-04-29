#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAIL_LINES=200
MARKER_PATTERN='Traceback|Error|Exception|FAILED|CANCELLED|OutOfMemory|OOM|CUDA|No space left'

usage() {
  cat <<'USAGE'
Usage: summarize_job.sh [--tail N] <job-id>

Show Slurm status, matching log files, failure markers, and recent log tails.

Options:
  --tail N   Number of lines to read from each log file (default: 200)

Example:
  summarize_job.sh --tail 100 123456
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --tail)
      if [[ $# -lt 2 ]]; then
        echo "Error: --tail requires a line count." >&2
        exit 2
      fi
      TAIL_LINES="$2"
      shift 2
      ;;
    --tail=*)
      TAIL_LINES="${1#*=}"
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

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 2
fi

job_id="$1"
if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
  echo "Error: job ID must be numeric, got: $job_id" >&2
  exit 2
fi

if [[ ! "$TAIL_LINES" =~ ^[0-9]+$ || "$TAIL_LINES" -eq 0 ]]; then
  echo "Error: --tail must be a positive integer, got: $TAIL_LINES" >&2
  exit 2
fi

run_cluster() {
  local output
  local status

  set +e
  output=$("$SCRIPT_DIR/cluster_exec.sh" "$@" 2>&1)
  status=$?
  set -e

  printf '%s\n' "$output"
  return "$status"
}

echo "== Slurm summary: job_id=$job_id =="

echo
echo "-- squeue --"
if ! run_cluster squeue -j "$job_id"; then
  echo "(squeue failed)"
fi

echo
echo "-- sacct --"
if ! run_cluster sacct -j "$job_id" --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS; then
  echo "(sacct failed)"
fi

echo
echo "-- logs --"
set +e
logs_output=$("$SCRIPT_DIR/find_job_logs.sh" "$job_id" 2>&1)
logs_status=$?
set -e

if [[ "$logs_status" -ne 0 ]]; then
  printf '%s\n' "$logs_output"
  exit "$logs_status"
fi

printf '%s\n' "$logs_output"

echo
echo "-- failure markers --"
found_marker=0
while IFS= read -r log_file; do
  [[ -z "$log_file" ]] && continue
  echo ">>> $log_file"

  set +e
  grep_output=$("$SCRIPT_DIR/cluster_exec.sh" grep -nE "$MARKER_PATTERN" "$log_file" 2>&1)
  grep_status=$?
  set -e

  if [[ "$grep_status" -eq 0 ]]; then
    found_marker=1
    printf '%s\n' "$grep_output"
  elif [[ "$grep_status" -eq 1 ]]; then
    echo "(no markers)"
  else
    printf '%s\n' "$grep_output"
  fi
done <<< "$logs_output"

if [[ "$found_marker" -eq 0 ]]; then
  echo "No failure markers found in matched logs."
fi

echo
echo "-- tails --"
while IFS= read -r log_file; do
  [[ -z "$log_file" ]] && continue
  echo
  echo ">>> tail -n $TAIL_LINES $log_file"
  if ! run_cluster tail -n "$TAIL_LINES" "$log_file"; then
    echo "(tail failed)"
  fi
done <<< "$logs_output"
