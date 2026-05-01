#!/usr/bin/env bash

ensure_writable_uv_cache() {
    if [ -z "${UV_CACHE_DIR:-}" ] || ! mkdir -p "$UV_CACHE_DIR" 2>/dev/null || [ ! -w "$UV_CACHE_DIR" ]; then
        export UV_CACHE_DIR="/tmp/uv_cache"
        mkdir -p "$UV_CACHE_DIR"
    fi
}
