"""Shared proxy core for qwen397_tool / qwen397_direct wrapper services.

Both wrappers behave the same way: receive an OpenAI-compatible Chat Completions
request, prepend a fixed system prompt to the messages, ensure max_tokens is
high enough to leave headroom for reasoning models, and forward the request
verbatim to an upstream OpenAI-compatible LLM endpoint.

The existing AgenticOCR Python client therefore drives any of these wrappers
through a single URL swap.
"""

import logging
import os
from typing import Any, Dict, List

import httpx
from fastapi import FastAPI, HTTPException

logger = logging.getLogger(__name__)


def make_app(
    *,
    title: str,
    system_prompt: str,
    upstream_url_env: str = "QWEN_UPSTREAM_URL",
    upstream_key_env: str = "QWEN_UPSTREAM_KEY",
    upstream_model_env: str = "QWEN_UPSTREAM_MODEL",
    default_max_tokens_env: str = "DEFAULT_MAX_TOKENS",
    default_max_tokens: int = 8192,
    request_timeout: int = 600,
) -> FastAPI:
    """Construct a FastAPI app that proxies /v1/chat/completions to upstream."""

    upstream_url = os.environ.get(upstream_url_env, "http://localhost:20000/v1/").rstrip("/")
    upstream_key = os.environ.get(upstream_key_env, "sk-placeholder")
    upstream_model = os.environ.get(upstream_model_env, "Qwen35-397B")
    max_tokens_default = int(os.environ.get(default_max_tokens_env, str(default_max_tokens)))

    app = FastAPI(title=title)

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "upstream_url": upstream_url,
            "upstream_model": upstream_model,
            "wrapper": title,
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: Dict[str, Any]):
        if not isinstance(req, dict):
            raise HTTPException(status_code=400, detail="Body must be a JSON object")

        messages: List[Dict[str, Any]] = list(req.get("messages") or [])
        # Replace any existing system message with our system prompt; if there
        # is none, prepend.
        if messages and messages[0].get("role") == "system":
            messages = messages[1:]
        messages = [{"role": "system", "content": system_prompt}] + messages

        forwarded = dict(req)
        forwarded["messages"] = messages
        forwarded["model"] = upstream_model

        # Reasoning models burn many tokens before producing the actual content
        # field. If the client didn't set a generous max_tokens, bump it up so
        # content isn't truncated to null.
        if "max_completion_tokens" in forwarded:
            mt = forwarded.get("max_completion_tokens") or 0
            if mt < max_tokens_default:
                forwarded["max_completion_tokens"] = max_tokens_default
        else:
            mt = forwarded.get("max_tokens") or 0
            if mt < max_tokens_default:
                forwarded["max_tokens"] = max_tokens_default

        headers = {
            "Authorization": f"Bearer {upstream_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=request_timeout) as client:
                resp = await client.post(
                    f"{upstream_url}/chat/completions",
                    json=forwarded,
                    headers=headers,
                )
        except httpx.HTTPError as e:
            logger.exception("Upstream call failed")
            raise HTTPException(status_code=502, detail=f"Upstream error: {e}") from e

        if resp.status_code != 200:
            logger.error("Upstream %d: %s", resp.status_code, resp.text[:500])
            raise HTTPException(
                status_code=resp.status_code,
                detail=f"Upstream returned {resp.status_code}: {resp.text[:300]}",
            )

        return resp.json()

    return app
