from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Any, Callable, Awaitable

BUILTIN_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run a shell command on the server. Use sparingly. Returns stdout+stderr and exit code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "The shell command to execute (bash)."},
                },
                "required": ["cmd"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "http_get",
            "description": "GET a URL and return up to 20000 chars of the response body as text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "max_chars": {"type": "integer", "default": 20000},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "now",
            "description": "Return the current server time in ISO format.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


@dataclass
class ToolCall:
    name: str
    args: dict
    raw_id: str | None


async def _tool_shell(args: dict) -> str:
    cmd = args.get("cmd", "")
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        out = (r.stdout or "") + (r.stderr or "")
        return f"[exit {r.returncode}]\n{out[:8000]}"
    except Exception as e:
        return f"[error] {e}"


async def _tool_http_get(args: dict) -> str:
    import httpx
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as c:
            r = await c.get(args["url"])
            txt = r.text
            return txt[: int(args.get("max_chars", 20000))]
    except Exception as e:
        return f"[error] {e}"


async def _tool_now(args: dict) -> str:
    import datetime
    return datetime.datetime.now().isoformat()


HANDLERS: dict[str, Callable[[dict], Awaitable[str]]] = {
    "shell": _tool_shell,
    "http_get": _tool_http_get,
    "now": _tool_now,
}


async def dispatch(name: str, args: dict) -> str:
    fn = HANDLERS.get(name)
    if fn is None:
        return f"[error] unknown tool: {name}"
    try:
        return await fn(args)
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"


def merge_tools(caller_tools: list[dict] | None) -> list[dict]:
    """Return the union of built-in and caller-supplied tools. Caller names may shadow built-ins."""
    if not caller_tools:
        return BUILTIN_TOOLS
    by_name: dict[str, dict] = {}
    for t in BUILTIN_TOOLS:
        by_name[t["function"]["name"]] = t
    for t in caller_tools:
        if isinstance(t, dict) and "function" in t and "name" in t["function"]:
            by_name[t["function"]["name"]] = t
    return list(by_name.values())


def caller_handler(caller_tools: list[dict] | None) -> dict[str, Callable[[dict], Awaitable[str]]]:
    """Built-in handlers are always available; caller-defined tools are documented but cannot be
    run server-side (caller must reply with tool_result events in-band — see chat route)."""
    return HANDLERS