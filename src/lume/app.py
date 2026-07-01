from __future__ import annotations

import json
import base64
import secrets
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from sse_starlette.sse import EventSourceResponse

from .attachments import (
    file_for_ollama,
    get_attachment,
    list_attachments,
    store_upload,
    text_from_attachment,
    transcribe_audio,
)
from .auth import auth_key, create_key, hash_key, list_keys, revoke_key
from .config import get_settings
from .db import init_db
from .ollama import stream_chat
from .threads import (
    append_message,
    create_thread,
    delete_thread,
    get_thread,
    list_messages,
    list_threads,
    rename_thread,
)
from .tools import dispatch, merge_tools


app = FastAPI(title="Lume", version="0.1.0",
              description="All-in-one local LLM gateway: chat, threads, attachments, voice, tools.")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ---------- auth dependency ----------

async def require_auth(authorization: str | None = Header(default=None)) -> dict:
    if not authorization:
        raise HTTPException(401, "missing Authorization header")
    if authorization.startswith("Bearer "):
        token = authorization[7:]
    else:
        token = authorization
    valid, is_admin = await auth_key(token)
    if not valid:
        raise HTTPException(401, "invalid api key")
    return {"valid": True, "is_admin": is_admin, "key": token}


async def require_admin(auth: dict = Depends(require_auth)) -> dict:
    if not auth["is_admin"]:
        raise HTTPException(403, "admin key required")
    return auth


# ---------- startup ----------

@app.on_event("startup")
async def _startup() -> None:
    await init_db()


# ---------- health ----------

@app.get("/health")
async def health() -> dict:
    s = get_settings()
    # probe ollama
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{s.ollama_base_url}/api/tags")
            ok = r.status_code == 200
    except Exception:
        ok = False
    return {"status": "ok", "ollama": ok, "model": s.model}


# ---------- threads ----------

@app.get("/v1/threads", dependencies=[Depends(require_auth)])
async def threads_list() -> dict:
    return {"threads": await list_threads()}


@app.post("/v1/threads", dependencies=[Depends(require_auth)])
async def threads_create(title: str = "Untitled thread", meta: dict | None = None) -> dict:
    return await create_thread(title=title, meta=meta)


@app.get("/v1/threads/{tid}", dependencies=[Depends(require_auth)])
async def threads_get(tid: str) -> dict:
    t = await get_thread(tid)
    if not t:
        raise HTTPException(404, "thread not found")
    msgs = await list_messages(tid)
    return {**t, "messages": msgs}


@app.patch("/v1/threads/{tid}", dependencies=[Depends(require_auth)])
async def threads_rename(tid: str, title: str) -> dict:
    t = await rename_thread(tid, title)
    if not t:
        raise HTTPException(404, "thread not found")
    return t


@app.delete("/v1/threads/{tid}", dependencies=[Depends(require_auth)])
async def threads_delete(tid: str) -> dict:
    ok = await delete_thread(tid)
    if not ok:
        raise HTTPException(404, "thread not found")
    return {"deleted": True}


@app.get("/v1/threads/{tid}/messages", dependencies=[Depends(require_auth)])
async def messages_list(tid: str) -> dict:
    t = await get_thread(tid)
    if not t:
        raise HTTPException(404, "thread not found")
    return {"messages": await list_messages(tid)}


# ---------- attachments (upload, fetch) ----------

@app.post("/v1/attachments", dependencies=[Depends(require_auth)])
async def attachments_upload(file: UploadFile = File(...)) -> dict:
    data = await file.read()
    return await store_upload(data, file.filename or "upload.bin", file.content_type)


@app.get("/v1/attachments/{aid}", dependencies=[Depends(require_auth)])
async def attachments_get(aid: int) -> Any:
    att = await get_attachment(aid)
    if not att:
        raise HTTPException(404, "attachment not found")
    return FileResponse(att["path"], media_type=att["mime"], filename=att["filename"])


@app.get("/v1/attachments/{aid}/meta", dependencies=[Depends(require_auth)])
async def attachments_meta(aid: int) -> dict:
    att = await get_attachment(aid)
    if not att:
        raise HTTPException(404, "attachment not found")
    return att


# ---------- chat (the heart of it) ----------

def _sse(event: dict) -> str:
    return json.dumps(event, ensure_ascii=False)


async def _build_context(
    thread_id: str | None,
    prompt: str,
    attachment_ids: list[int],
    system: str | None,
) -> list[dict]:
    """Construct ollama messages list. If thread given, replay history; else one-shot."""
    msgs: list[dict] = []
    if system:
        msgs.append({"role": "system", "content": system})
    else:
        msgs.append({"role": "system", "content": _default_system()})

    if thread_id:
        history = await list_messages(thread_id)
        for m in history:
            ollama_msg = _history_msg_to_ollama(m)
            if ollama_msg is not None:
                msgs.append(ollama_msg)

    user_text_parts: list[str] = []
    if prompt:
        user_text_parts.append(prompt)
    image_b64s: list[str] = []
    atts = await list_attachments(attachment_ids)
    for att in atts:
        if att["kind"] == "audio":
            # transcribe server-side, append text
            try:
                t = transcribe_audio(att["path"])
                user_text_parts.append(f"[voice message]\n{t}")
            except Exception as e:
                user_text_parts.append(f"[voice transcription failed: {e}]")
        elif att["kind"] == "text":
            inline = text_from_attachment(att)
            if inline:
                user_text_parts.append(f"[attachment: {att['filename']}]\n{inline[:60000]}")
            else:
                user_text_parts.append(f"[attachment: {att['filename']}]")
        elif att["kind"] == "image":
            try:
                b64 = base64.b64encode(Path(att["path"]).read_bytes()).decode("ascii")
                image_b64s.append(b64)
            except Exception:
                user_text_parts.append(f"[image unreadable: {att['filename']}]")
        else:
            user_text_parts.append(
                f"[binary attachment: {att['filename']}, {att['size']} bytes]"
            )

    if not user_text_parts and not image_b64s:
        raise HTTPException(400, "empty request: provide prompt or attachment")

    user_msg: dict[str, Any] = {"role": "user", "content": "\n\n".join(user_text_parts)}
    if image_b64s:
        user_msg["images"] = image_b64s
    msgs.append(user_msg)
    return msgs


def _history_msg_to_ollama(m: dict) -> dict | None:
    role = m["role"]
    if role == "tool":
        return {
            "role": "tool",
            "content": m["content"],
            "name": m.get("tool_name") or "",
        }
    out: dict[str, Any] = {"role": role}
    if m["content"]:
        out["content"] = m["content"]
    if m.get("tool_calls"):
        out["tool_calls"] = m["tool_calls"]
    # images carried as paths; ollama needs base64 or path — we'll skip image replay for brevity
    return out if ("content" in out or "tool_calls" in out) else None


def _default_system(self=None) -> str:
    return (
        "You are Lume, a helpful agentic assistant running locally. "
        "You can answer with text and call tools. When tools are available, use them as needed. "
        "Be concise and direct."
    )


async def _emit_loop(
    messages: list[dict],
    tools: list[dict] | None,
    voice: bool,
    options: dict | None,
) -> AsyncIterator[str]:
    """SSE generator: drives ollama + tool dispatch loop, yields typed events."""
    s = get_settings()
    rounds = 0
    cur_messages = list(messages)
    last_assistant_content = ""
    while rounds <= s.max_tool_iterations:
        rounds += 1
        assistant_content = ""
        assistant_tool_calls: list = []
        async for ev in stream_chat(cur_messages, tools=tools, options=options):
            if ev["type"] == "token":
                assistant_content += ev["content"]
                yield _sse({"type": "token", "content": ev["content"]})
            elif ev["type"] == "message":
                assistant_content = ev["content"]
                assistant_tool_calls = ev["tool_calls"]
                last_assistant_content = assistant_content
        # save assistant turn later by caller; here we just stream
        if not assistant_tool_calls:
            break
        # emit tool_calls event
        yield _sse({"type": "tool_calls", "tool_calls": assistant_tool_calls})
        # append assistant message with tool_calls for the next round
        cur_messages.append({
            "role": "assistant",
            "content": assistant_content or "",
            "tool_calls": assistant_tool_calls,
        })
        # dispatch each tool call
        for tc in assistant_tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"_raw": args}
            tc_id = tc.get("id") or name
            # only built-in tools run server-side; caller-defined tools return a placeholder
            from .tools import HANDLERS
            if name in HANDLERS:
                result = await dispatch(name, args)
                yield _sse({"type": "tool_result", "tool_call_id": tc_id, "name": name, "result": result})
                cur_messages.append({
                    "role": "tool",
                    "content": result,
                    "name": name,
                })
            else:
                # caller-defined tool that the server can't execute — surface a placeholder result
                # so the model can continue; the caller may intercept the tool_calls event and
                # feed a better result via a follow-up message in the thread.
                note = (f"[server cannot execute tool '{name}' (caller-defined); "
                        f"send its result in a follow-up user message if needed]")
                yield _sse({"type": "tool_result", "tool_call_id": tc_id, "name": name,
                            "result": note, "note": "caller_defined"})
                cur_messages.append({"role": "tool", "content": note, "name": name})
        # loop continues → model gets tool results and may produce final text or more tool calls

    # final: optional voice
    if voice and last_assistant_content:
        audio = await _tts(last_assistant_content)
        if audio:
            yield _sse({"type": "voice", "url": f"/v1/voice/{audio['name']}", "name": audio["name"]})
    yield _sse({"type": "done"})


async def _tts(text: str) -> dict | None:
    s = get_settings()
    if not s.supertonic_enabled:
        return None
    try:
        from supertonic import TTS  # type: ignore
    except Exception:
        # fall back to marking voice as unavailable
        return None
    try:
        tts = TTS(auto_download=True)
        style = tts.get_voice_style(voice_name=s.supertonic_voice)
        wav, _ = tts.synthesize(
            text=text, lang=s.supertonic_lang, voice_style=style,
            total_steps=s.supertonic_steps, speed=s.supertonic_speed,
        )
        name = f"{uuid.uuid4().hex}.wav"
        out = s.voice_dir / name
        tts.save_audio(wav, str(out))
        return {"name": name, "path": str(out)}
    except Exception:
        return None


# --- chat endpoints ---

@app.post("/v1/chat")
async def chat(
    request: Request,
    prompt: str | None = Form(default=None),
    thread_id: str | None = Form(default=None),
    system: str | None = Form(default=None),
    voice: bool = Form(default=False),
    tools_json: str | None = Form(default=None, alias="tools"),
    options_json: str | None = Form(default=None, alias="options"),
    attachment_ids: str | None = Form(default=None),
    files: list[UploadFile] | None = None,
    auth: dict = Depends(require_auth),
) -> StreamingResponse:
    """Chat endpoint. multipart/form-data. Streams SSE typed events.

    Form fields:
      prompt            str    user message text (optional if attachments exist)
      thread_id         str    optional; if absent, creates an ephemeral one-shot
      system            str    optional system prompt override
      voice             bool   if true, also synthesizes TTS of the final reply
      tools             str    JSON array of OpenAI-style tool schemas to merge with built-ins
      options           str    JSON ollama options (temperature, etc.)
      attachment_ids    str    JSON list of previously uploaded attachment ids to include
      files             files  zero or more fresh files to upload-and-include in this turn

    SSE events (each line: `data: <json>`):
      {"type":"token","content":"..."}             incremental text
      {"type":"tool_calls","tool_calls":[...]}     model wants to call tools
      {"type":"tool_result","tool_call_id":"...","name":"...","result":"..."}
      {"type":"voice","url":"/v1/voice/<name>","name":"..."}
      {"type":"done"}                              always last
    """
    s = get_settings()

    # parse JSON form fields
    try:
        caller_tools = json.loads(tools_json) if tools_json else None
    except json.JSONDecodeError:
        raise HTTPException(400, "tools must be JSON")
    try:
        options = json.loads(options_json) if options_json else None
    except json.JSONDecodeError:
        raise HTTPException(400, "options must be JSON")
    try:
        att_ids: list[int] = json.loads(attachment_ids) if attachment_ids else []
    except json.JSONDecodeError:
        raise HTTPException(400, "attachment_ids must be JSON list")

    # attach fresh uploaded files
    if files:
        for f in files:
            data = await f.read()
            att = await store_upload(data, f.filename or "upload.bin", f.content_type)
            att_ids.append(att["id"])

    # thread handling
    ephemeral = False
    if thread_id:
        t = await get_thread(thread_id)
        if not t:
            raise HTTPException(404, "thread not found")
    else:
        t = await create_thread(title=(prompt or "Voice / attachment chat")[:80])
        thread_id = t["id"]
        ephemeral = True

    messages = await _build_context(thread_id, prompt or "", att_ids, system)
    tools = merge_tools(caller_tools)

    # persist the user turn
    await append_message(thread_id, "user", content=prompt or "", attachment_ids=att_ids)

    async def gen() -> AsyncIterator[str]:
        assistant_text = ""
        assistant_tool_calls: list = []
        try:
            async for raw in _emit_loop(messages, tools, voice, options):
                # we need to parse our own SSE payload back to track state — simpler: re-emit as-is
                # but capture the assistant text for persistence.
                # _emit_loop emits pre-serialized JSON strings; parse here to track.
                ev = json.loads(raw)
                if ev["type"] == "token":
                    assistant_text += ev["content"]
                elif ev["type"] == "tool_calls":
                    assistant_tool_calls = ev["tool_calls"]
                yield raw
            # persist assistant turn
            await append_message(
                thread_id, "assistant",
                content=assistant_text,
                tool_calls=assistant_tool_calls or None,
            )
        except httpx.HTTPStatusError as e:
            try:
                await e.response.aread()
                body = e.response.text[:500]
            except Exception:
                body = "<unreadable>"
            yield _sse({"type": "error", "message": f"ollama error: {e.response.status_code} {body}"})
            yield _sse({"type": "done"})
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield _sse({"type": "error", "message": f"{type(e).__name__}: {e}"})
            yield _sse({"type": "done"})

    headers = {
        "X-Lume-Thread-Id": thread_id,
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }
    return EventSourceResponse(gen(), media_type="text/event-stream", headers=headers)


@app.post("/v1/chat/json", dependencies=[Depends(require_auth)])
async def chat_json(
    body: dict,
) -> dict:
    """Non-streaming JSON variant. Body:
    {prompt, thread_id?, system?, voice?, tools?, options?, attachment_ids?, file_ids?}
    Returns {thread_id, content, tool_calls, voice?}."""
    prompt = body.get("prompt", "")
    thread_id = body.get("thread_id")
    system = body.get("system")
    voice = bool(body.get("voice"))
    caller_tools = body.get("tools")
    options = body.get("options")
    att_ids = body.get("attachment_ids", []) or []

    if thread_id:
        if not await get_thread(thread_id):
            raise HTTPException(404, "thread not found")
    else:
        t = await create_thread(title=(prompt or "chat")[:80])
        thread_id = t["id"]

    messages = await _build_context(thread_id, prompt, list(att_ids), system)
    tools = merge_tools(caller_tools)
    await append_message(thread_id, "user", content=prompt, attachment_ids=list(att_ids))

    s = get_settings()
    rounds = 0
    cur = list(messages)
    assistant_text = ""
    assistant_tool_calls: list = []
    while rounds <= s.max_tool_iterations:
        rounds += 1
        assistant_text = ""
        assistant_tool_calls = []
        async for ev in stream_chat(cur, tools=tools, options=options):
            if ev["type"] == "token":
                assistant_text += ev["content"]
            elif ev["type"] == "message":
                assistant_text = ev["content"]
                assistant_tool_calls = ev["tool_calls"]
        if not assistant_tool_calls:
            break
        cur.append({"role": "assistant", "content": assistant_text or "",
                    "tool_calls": assistant_tool_calls})
        from .tools import HANDLERS
        for tc in assistant_tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            if name in HANDLERS:
                result = await dispatch(name, args)
            else:
                result = f"[server cannot execute tool '{name}']"
            cur.append({"role": "tool", "content": result, "name": name})

    await append_message(thread_id, "assistant", content=assistant_text,
                          tool_calls=assistant_tool_calls or None)
    out: dict[str, Any] = {"thread_id": thread_id, "content": assistant_text,
                          "tool_calls": assistant_tool_calls}
    if voice and assistant_text:
        audio = await _tts(assistant_text)
        if audio:
            out["voice"] = {"url": f"/v1/voice/{audio['name']}", "name": audio["name"]}
    return out


# ---------- voice fetch ----------

@app.get("/v1/voice/{name}", dependencies=[Depends(require_auth)])
async def voice_get(name: str) -> Any:
    # guard against path traversal
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(400, "bad name")
    s = get_settings()
    p = s.voice_dir / name
    if not p.exists():
        raise HTTPException(404, "voice not found")
    return FileResponse(str(p), media_type="audio/wav", filename=name)


# ---------- admin: keys ----------

@app.get("/v1/admin/keys", dependencies=[Depends(require_admin)])
async def admin_keys_list() -> dict:
    return {"keys": await list_keys()}


@app.post("/v1/admin/keys", dependencies=[Depends(require_admin)])
async def admin_keys_create(name: str = "client", is_admin: bool = False) -> dict:
    plaintext = await create_key(name, is_admin=is_admin)
    return {"key": plaintext, "name": name, "is_admin": is_admin}


@app.delete("/v1/admin/keys/{kid}", dependencies=[Depends(require_admin)])
async def admin_keys_revoke(kid: int) -> dict:
    ok = await revoke_key(kid)
    if not ok:
        raise HTTPException(404, "key not found or already revoked")
    return {"revoked": True, "id": kid}


# ---------- one-shot audio transcription (utility) ----------

@app.post("/v1/transcribe", dependencies=[Depends(require_auth)])
async def transcribe(file: UploadFile = File(...), language: str | None = None) -> dict:
    data = await file.read()
    s = get_settings()
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename or "in.wav").suffix, delete=False) as tf:
        tf.write(data)
        tmp = tf.name
    try:
        text = transcribe_audio(tmp, language=language)
    finally:
        Path(tmp).unlink(missing_ok=True)
    return {"text": text}


def main() -> None:
    import uvicorn
    s = get_settings()
    uvicorn.run("lume.app:app", host=s.bind_host, port=s.bind_port, reload=False)


if __name__ == "__main__":
    main()