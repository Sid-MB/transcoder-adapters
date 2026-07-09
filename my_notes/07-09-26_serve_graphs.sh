#!/usr/bin/env bash
# [implement: new 07/02 gemmascope transcoder experiments] 2026-07-09
# Serve the ORIGINAL vs FINE-TUNED GemmaScope transcoder circuit-tracer graphs side by side.
# Graphs were built by analysis/attribution/compare_finetuned_transcoder_graphs.py (job 16118628).
# See my_notes/07-09-26.md for the full context and the comparison table.
#
# Usage:
#   ./my_notes/07-09-26_serve_graphs.sh            # serve both (original :8050, finetuned :8051)
#   GRAPH_DIR=<other run dir> ./my_notes/07-09-26_serve_graphs.sh   # a different comparison run
#
# The servers block; original runs in the background, finetuned in the foreground (Ctrl-C stops
# both). From your laptop, forward the ports and open the two URLs:
#   ssh -L 8050:localhost:8050 -L 8051:localhost:8051 <this-node>
#   http://localhost:8050  (original)   http://localhost:8051  (fine-tuned)
set -euo pipefail

GRAPH_DIR="${GRAPH_DIR:-/nlp/scr/siddharth/transcoder-adapters/transcoder_finetune_graphs/L0-24-25_20260709_145430_16118628}"
ORIG_PORT="${ORIG_PORT:-8050}"
FT_PORT="${FT_PORT:-8051}"

if [[ ! -d "$GRAPH_DIR/original" || ! -d "$GRAPH_DIR/finetuned" ]]; then
  echo "ERROR: expected $GRAPH_DIR/{original,finetuned} to exist. Set GRAPH_DIR to a compare_finetuned_transcoder_graphs run dir." >&2
  exit 1
fi

cd "$(git rev-parse --show-toplevel)"

echo "[serve] original  graphs -> http://localhost:$ORIG_PORT  ($GRAPH_DIR/original)"
uv run --extra viz circuit-tracer start-server --graph_file_dir "$GRAPH_DIR/original" --port "$ORIG_PORT" &
ORIG_PID=$!
trap 'kill $ORIG_PID 2>/dev/null || true' EXIT

echo "[serve] finetuned graphs -> http://localhost:$FT_PORT  ($GRAPH_DIR/finetuned)"
echo "[serve] Ctrl-C to stop both."
uv run --extra viz circuit-tracer start-server --graph_file_dir "$GRAPH_DIR/finetuned" --port "$FT_PORT"
