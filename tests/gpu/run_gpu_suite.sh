#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

echo "Running GPU test suite."
uv run python -m unittest discover -v -s tests/gpu -p 'test_*.py'
