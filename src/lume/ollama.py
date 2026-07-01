from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from .config import get_settings


async def stream_chat(
    messages: list[dict],
    tools: list[dict] | None = None,
    options: dict | None = None,
) -> AsyncIterator[dict]:
    """Stream tokens from Ollama /api/chat. Yields dicts:
        {"type": "token", "content": "..."}
        {"type": "message", "content": "...", "tool_calls": [...]}   (final)
    """
    s = get_settings()
    payload: dict[str, Any] = {
        "model": s.model,
        "messages": messages,
        "stream": True,
    }
    if tools:
        payload["tools"] = tools
    if options:
        payload["options"] = options

    accumulated = ""
    tool_calls: list = []
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", f"{s.ollama_base_url}/api/chat", json=payload) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                msg = chunk.get("message") or {}
                piece = msg.get("content", "")
                if piece:
                    accumulated += piece
                    yield {"type": "token", "content": piece}
                if msg.get("tool_calls"):
                    tool_calls.extend(msg["tool_calls"])
                if chunk.get("done"):
                    yield {
                        "type": "message",
                        "content": accumulated,
                        "tool_calls": tool_calls,
                        "done_reason": chunk.get("done_reason", "stop"),
                    }
                    return