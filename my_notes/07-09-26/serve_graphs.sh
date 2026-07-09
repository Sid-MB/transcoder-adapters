#!/usr/bin/env bash
# [implement: new 07/02 gemmascope transcoder experiments] 2026-07-09
# Serve the ORIGINAL vs FINE-TUNED GemmaScope transcoder circuit-tracer graphs side by side,
# WITH feature activation examples (click a feature node to see its top activating examples).
# Graphs built by analysis/attribution/compare_finetuned_transcoder_graphs.py (job 16118628).
# See ./README.md for the full context and the comparison table.
#
# Feature examples come from the latest base GemmaScope feature collection (20k samples,
# width_16k/average_l0_76, chat+web), which matches these graphs' scan exactly:
#   local: $LARGE_ARTIFACTS_DIR/transcoder-adapters/feature_data/base_ms20000_sharded
#   HF:    siddharthmb/2026.TA.features_gemma-2-2b_gemmascope_width_16k_average_l0_76_ms20000_ml1024_tk10_h5609948b210f
# NOTE: the fine-tuned run changed layers 0/24/25, so their examples are from the ORIGINAL
# weights (stale for those 3 layers; correct for the other 23). Collect features on the
# fine-tuned transcoders for fully-matched examples on all layers.
#
# Usage:
#   ./serve_graphs.sh                       # original :8050, finetuned :8051 (bundled graphs)
#   GRAPH_DIR=<dir> ./serve_graphs.sh       # a different compare run
#   FEATURES_DIR=<dir> ./serve_graphs.sh    # a different feature collection (…/circuit_tracer_features)
#   NO_FEATURES=1 ./serve_graphs.sh         # serve without feature examples
#
# The servers block; original runs in the background, finetuned in the foreground (Ctrl-C stops
# both). From your laptop, forward the ports:
#   ssh -L 8050:localhost:8050 -L 8051:localhost:8051 <this-node>
#   http://localhost:8050  (original)   http://localhost:8051  (fine-tuned)
# POSIX sh compatible (works under bash or dash / `sh script.sh`).
set -eu

SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
GRAPH_DIR="${GRAPH_DIR:-$SELF_DIR/graphs}"
ORIG_PORT="${ORIG_PORT:-8050}"
FT_PORT="${FT_PORT:-8051}"

# Feature examples: default to the latest base GemmaScope collection (matches these graphs' scan).
LARGE_ARTIFACTS_DIR="${LARGE_ARTIFACTS_DIR:-/nlp/scr/$USER}"
FEATURES_DIR="${FEATURES_DIR:-$LARGE_ARTIFACTS_DIR/transcoder-adapters/feature_data/base_ms20000_sharded/circuit_tracer_features}"

if [ ! -d "$GRAPH_DIR/original" ] || [ ! -d "$GRAPH_DIR/finetuned" ]; then
  echo "ERROR: expected $GRAPH_DIR/{original,finetuned} to exist. Set GRAPH_DIR to a compare_finetuned_transcoder_graphs run dir." >&2
  exit 1
fi

FEAT_ARGS=""
if [ -z "${NO_FEATURES:-}" ]; then
  if [ -f "$FEATURES_DIR/index.json.gz" ]; then
    FEAT_ARGS="--features_dir $FEATURES_DIR"
    echo "[serve] feature examples from: $FEATURES_DIR"
  else
    echo "[serve] WARNING: no feature collection at $FEATURES_DIR (index.json.gz missing); serving WITHOUT examples." >&2
    echo "[serve]          set FEATURES_DIR=<collection>/circuit_tracer_features or NO_FEATURES=1 to silence." >&2
  fi
fi

cd "$(git rev-parse --show-toplevel)"

echo "[serve] original  graphs -> http://localhost:$ORIG_PORT  ($GRAPH_DIR/original)"
# shellcheck disable=SC2086
uv run --extra viz circuit-tracer start-server --graph_file_dir "$GRAPH_DIR/original" --port "$ORIG_PORT" $FEAT_ARGS &
ORIG_PID=$!
trap 'kill $ORIG_PID 2>/dev/null || true' EXIT

echo "[serve] finetuned graphs -> http://localhost:$FT_PORT  ($GRAPH_DIR/finetuned)"
echo "[serve] Ctrl-C to stop both."
# shellcheck disable=SC2086
uv run --extra viz circuit-tracer start-server --graph_file_dir "$GRAPH_DIR/finetuned" --port "$FT_PORT" $FEAT_ARGS
