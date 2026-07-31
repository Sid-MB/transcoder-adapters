#!/bin/bash
cd /juice2/u/siddharth/transcoder-adapters
PYTHONPATH=/juice2/u/siddharth/transcoder-adapters uv run --extra viz python /nlp/scr/siddharth/transcoder-adapters/_refusal_conts/gen.py
