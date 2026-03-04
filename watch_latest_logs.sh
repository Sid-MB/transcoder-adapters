#!/bin/bash

LOG_DIR="logs"

# 1. Find the most recent .out file (handles both <digits>.out and <jobid>_<timestamp>.out)
OUT_FILE=$(ls -t "$LOG_DIR"/[0-9]*.out 2>/dev/null | head -1)

# 2. Check if a log was actually found
if [ -z "$OUT_FILE" ]; then
    echo "Error: No logs found in $LOG_DIR/"
    exit 1
fi

# Derive the matching .err file by swapping the extension
ERR_FILE="${OUT_FILE%.out}.err"
BASENAME=$(basename "$OUT_FILE" .out)

echo "Watching latest logs: $BASENAME (.out and .err)"

# 3. Launch tmux
# Create a new session named 'log_watcher', run the first tail, split it, and run the second tail.
# tmux new-session -d -s "log_watcher_$LATEST_NUM" "tail -f $OUT_FILE"
# tmux split-window -h "tail -f $ERR_FILE"
# tmux select-layout even-horizontal
# tmux attach-session -t "log_watcher_$LATEST_NUM"
tail -f "$OUT_FILE" "$ERR_FILE"
