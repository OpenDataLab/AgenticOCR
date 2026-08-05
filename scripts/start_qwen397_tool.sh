#!/bin/bash
# Launch the qwen397_tool wrapper service.
# Wraps Qwen3.5-397B with the AgenticOCR system prompt + OpenAI-compat
# /v1/chat/completions endpoint, so the existing AgenticOCR client can
# drive it via URL swap.
#
# Service is purely a proxy — no GPU needed. Run anywhere with network
# access to the upstream Qwen3.5-397B endpoint.

set -e

export QWEN_UPSTREAM_URL=${QWEN_UPSTREAM_URL:-http://localhost:20000/v1/}
export QWEN_UPSTREAM_KEY=${QWEN_UPSTREAM_KEY:-sk-placeholder}
export QWEN_UPSTREAM_MODEL=${QWEN_UPSTREAM_MODEL:-Qwen35-397B}
export DEFAULT_MAX_TOKENS=${DEFAULT_MAX_TOKENS:-8192}
export QWEN397_TOOL_PORT=${QWEN397_TOOL_PORT:-8006}

HOST=${HOST:-0.0.0.0}
PORT=${QWEN397_TOOL_PORT}

cd "$(dirname "$0")/.."

echo "Starting qwen397_tool wrapper on ${HOST}:${PORT}"
echo "  Upstream:        ${QWEN_UPSTREAM_URL}"
echo "  Upstream model:  ${QWEN_UPSTREAM_MODEL}"
echo "  Max tokens:      ${DEFAULT_MAX_TOKENS}"

python -m uvicorn src.agents.qwen397_tool_server:app \
    --host "${HOST}" \
    --port "${PORT}" \
    --workers 1
