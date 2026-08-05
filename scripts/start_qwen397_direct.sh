#!/bin/bash
# Launch the qwen397_direct wrapper service.
# Same proxy structure as qwen397_tool but injects the DirectExtractor
# system prompt instead, so the wrapped LLM emits final JSON in one turn.

set -e

export QWEN_UPSTREAM_URL=${QWEN_UPSTREAM_URL:-http://localhost:20000/v1/}
export QWEN_UPSTREAM_KEY=${QWEN_UPSTREAM_KEY:-sk-placeholder}
export QWEN_UPSTREAM_MODEL=${QWEN_UPSTREAM_MODEL:-Qwen35-397B}
export DEFAULT_MAX_TOKENS=${DEFAULT_MAX_TOKENS:-8192}
export QWEN397_DIRECT_PORT=${QWEN397_DIRECT_PORT:-8007}

HOST=${HOST:-0.0.0.0}
PORT=${QWEN397_DIRECT_PORT}

cd "$(dirname "$0")/.."

echo "Starting qwen397_direct wrapper on ${HOST}:${PORT}"
echo "  Upstream:        ${QWEN_UPSTREAM_URL}"
echo "  Upstream model:  ${QWEN_UPSTREAM_MODEL}"
echo "  Max tokens:      ${DEFAULT_MAX_TOKENS}"

python -m uvicorn src.agents.qwen397_direct_server:app \
    --host "${HOST}" \
    --port "${PORT}" \
    --workers 1
