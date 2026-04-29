#!/usr/bin/env bash
#
# Shared logging setup. Source this from Bash scripts after setting LOG_DIR.
#
# Optional:
#   LOG_PREFIX  Prefix for the log filename. Defaults to "local".

if [ -z "$LOG_DIR" ]; then
    echo "ERROR: LOG_DIR must be set before sourcing sh/common_logging.sh" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="${LOG_DIR}/${LOG_PREFIX:-local}_${TIMESTAMP}"
exec > >(tee "${LOGFILE}.out") 2> >(tee "${LOGFILE}.err" >&2)
