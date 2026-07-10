#!/usr/bin/env bash
# [implement: new 07/02 gemmascope transcoder experiments] 2026-07-09
# Serve the ORIGINAL vs FINE-TUNED GemmaScope transcoder circuit-tracer graphs side by side,
# WITH feature activation examples (click a feature node -> ~20 top activating examples/dataset).
# Graphs built by analysis/attribution/compare_finetuned_transcoder_graphs.py (job 16118628).
# See ./README.md for the full context and the comparison table.
#
# Feature examples: the graphs are tagged with the project's CURRENT-BEST base feature scan,
#   base_ms100000_dtk20 (full corpus: entire 100k lmsys val + 100k fineweb, top_k/domain_top_k 20)
#   HF: siddharthmb/2026.TA.features_gemma-2-2b_gemmascope_width_16k_average_l0_76_ms100000_ml1024_tk2_h12ad59325ffd
# which the frontend fetches DIRECTLY from HuggingFace (no local download). Your browser needs
# internet access (it does, over the ssh tunnel from your laptop).
#
# NOTE on the fine-tuned side (:8051): the re-fine-tune changed layers 0/24/25, but the HF
# collection only has ORIGINAL-weight examples. So those 3 layers' examples are approximate
# (nodes are flagged in the UI). Faithful re-collection on the fine-tuned weights was deemed too
# slow (dense 26-layer/425k-feature accumulation). To use a LOCAL collection instead, set
# FEATURES_DIR (a run's circuit_tracer_features dir) — the graphs must then be re-tagged to a
# local '/...' scan (see analysis/attribution/retag_graphs_for_local_features.py).
#
# Usage:
#   ./serve_graphs.sh                    # original :8050, finetuned :8051 (HF examples)
#   GRAPH_DIR=<dir> ./serve_graphs.sh    # a different compare run
#   FEATURES_DIR=<dir> ./serve_graphs.sh # serve local features instead (needs local-scan graphs)
# POSIX sh compatible (works under bash or dash / `sh script.sh`).
set -eu

SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
GRAPH_DIR="${GRAPH_DIR:-$SELF_DIR/graphs}"
ORIG_PORT="${ORIG_PORT:-8050}"
FT_PORT="${FT_PORT:-8051}"

if [ ! -d "$GRAPH_DIR/original" ] || [ ! -d "$GRAPH_DIR/finetuned" ]; then
  echo "ERROR: expected $GRAPH_DIR/{original,finetuned} to exist. Set GRAPH_DIR to a compare_finetuned_transcoder_graphs run dir." >&2
  exit 1
fi

# Feature examples come from HF (baked into the graph scan) by default; FEATURES_DIR overrides
# with a local collection (only used if the graphs are tagged with a local '/...' scan).
FEAT_ARGS=""
if [ -n "${FEATURES_DIR:-}" ]; then
  FEAT_ARGS="--features_dir $FEATURES_DIR"
  echo "[serve] local feature examples from: $FEATURES_DIR"
else
  echo "[serve] feature examples fetched from HuggingFace (base_ms100000_dtk20, ~20/feature)."
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
