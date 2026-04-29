#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: cluster_exec.sh <command> [args...]

Run a command from the canonical Stanford cluster checkout.

If sbatch is available locally, this runs directly after cd'ing to the cluster
repo directory. Otherwise it runs the command through ssh sc.stanford.edu.

Environment overrides:
  SLURM_SSH_HOST   SSH host to use when local (default: sc.stanford.edu)
  SLURM_REPO_DIR   Repo directory on the cluster (default: /nlp/u/siddharth/transcoder-adapters)

Examples:
  cluster_exec.sh squeue -j 123456
  cluster_exec.sh bash -lc 'find logs -type f | sort | tail'
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -eq 0 ]]; then
  usage >&2
  exit 2
fi

SSH_HOST="${SLURM_SSH_HOST:-sc.stanford.edu}"
REPO_DIR="${SLURM_REPO_DIR:-/nlp/u/siddharth/transcoder-adapters}"

quote_args() {
  local quoted=""
  local arg
  local q

  for arg in "$@"; do
    printf -v q '%q' "$arg"
    quoted+="${quoted:+ }$q"
  done

  printf '%s' "$quoted"
}

if command -v sbatch >/dev/null 2>&1; then
  cd "$REPO_DIR"
  exec "$@"
fi

remote_cmd="cd $(quote_args "$REPO_DIR") && $(quote_args "$@")"
exec ssh "$SSH_HOST" "$remote_cmd"
