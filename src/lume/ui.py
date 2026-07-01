"""Lume — Streamlit chat UI.

A daily-driver web frontend for the Lume API. Supports:
  - Per-client API key (entered once, stored in session state)
  - Thread sidebar: list, create, rename, delete, switch
  - Streaming chat with token-by-token typewriter (st.write_stream)
  - File + audio attachments (st.chat_input accept_file/accept_audio)
  - Voice replies (server-side Supertonic TTS, autoplayed via st.audio)
  - Tool-call visibility (st.status per tool round)
  - System prompt + voice-out toggle in the sidebar

Run with:  uv run streamlit run src/lume/ui.py
  or:       uv run lume-ui
"""
from __future__ import annotations

import io
import json
import os
from typing import Any, Iterator

import requests
import streamlit as st

DEFAULT_BASE_URL = os.environ.get("LUME_UI_BASE_URL", "http://127.0.0.1:8000")

# ---- session state bootstrap ----

SS = st.session_state
for key, default in {
    "api_key": "",
    "base_url": DEFAULT_BASE_URL,
    "current_thread": None,
    "messages": [],          # list[dict] cached render rows: {role, content, tool_calls, attachments, voice}
    "threads": [],
    "system_prompt": "",
    "voice_out": False,
    "authed": False,
    "auth_error": None,
}.items():
    if key not in SS:
        SS[key] = default


# ---- API client (sync) ----

class Lume:
    def __init__(self, base_url: str, key: str) -> None:
        self.base = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {key}"}

    def _check(self, r: requests.Response) -> None:
        if r.status_code == 401:
            SS.auth_error = "Invalid API key."
            SS.authed = False
            st.rerun()
        if not r.ok:
            st.error(f"API error {r.status_code}: {r.text[:300]}")
            st.stop()

    def health(self) -> dict:
        try:
            r = requests.get(f"{self.base}/health", timeout=5)
            return r.json() if r.ok else {"status": "down", "ollama": False}
        except Exception:
            return {"status": "down", "ollama": False}

    def list_threads(self) -> list[dict]:
        r = requests.get(f"{self.base}/v1/threads", headers=self.headers, timeout=10)
        self._check(r)
        return r.json().get("threads", [])

    def create_thread(self, title: str = "New chat") -> dict:
        r = requests.post(
            f"{self.base}/v1/threads", headers=self.headers,
            params={"title": title}, timeout=10,
        )
        self._check(r)
        return r.json()

    def rename_thread(self, tid: str, title: str) -> dict:
        r = requests.patch(
            f"{self.base}/v1/threads/{tid}", headers=self.headers,
            params={"title": title}, timeout=10,
        )
        self._check(r)
        return r.json()

    def delete_thread(self, tid: str) -> None:
        r = requests.delete(f"{self.base}/v1/threads/{tid}", headers=self.headers, timeout=10)
        self._check(r)

    def get_thread(self, tid: str) -> dict:
        r = requests.get(f"{self.base}/v1/threads/{tid}", headers=self.headers, timeout=10)
        self._check(r)
        return r.json()

    def upload(self, name: str, data: bytes, mime: str) -> int:
        r = requests.post(
            f"{self.base}/v1/attachments", headers=self.headers,
            files={"file": (name, data, mime)}, timeout=120,
        )
        self._check(r)
        return int(r.json()["id"])

    def fetch_bytes(self, path: str) -> bytes:
        r = requests.get(f"{self.base}{path}", headers=self.headers, timeout=120)
        self._check(r)
        return r.content

    def chat_stream(
        self,
        prompt: str,
        thread_id: str | None,
        attachment_ids: list[int],
        system: str | None,
        voice: bool,
    ) -> tuple[requests.Response, dict[str, str]]:
        data: dict[str, Any] = {
            "prompt": prompt,
            "voice": "true" if voice else "false",
        }
        if thread_id:
            data["thread_id"] = thread_id
        if system:
            data["system"] = system
        if attachment_ids:
            data["attachment_ids"] = json.dumps(attachment_ids)
        r = requests.post(
            f"{self.base}/v1/chat", headers=self.headers,
            data=data, stream=True, timeout=None,
        )
        self._check(r)
        thread_id_hdr = r.headers.get("X-Lume-Thread-Id", thread_id or "")
        return r, {"X-Lume-Thread-Id": thread_id_hdr}


def sse_lines(resp: requests.Response) -> Iterator[dict]:
    """Yield parsed SSE event dicts from a streaming response."""
    buf = b""
    for chunk in resp.iter_content(chunk_size=4096):
        if not chunk:
            continue
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line:
                continue
            if line.startswith(b"data:"):
                payload = line[5:].strip()
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError:
                    continue


# ---- helpers ----

def client() -> Lume:
    return Lume(SS.base_url, SS.api_key)


def switch_thread(tid: str | None) -> None:
    SS.current_thread = tid
    SS.messages = []
    if tid:
        try:
            t = client().get_thread(tid)
            for m in t.get("messages", []):
                row = {
                    "role": m["role"],
                    "content": m.get("content", ""),
                    "tool_calls": m.get("tool_calls", []) or [],
                    "attachments": [],
                    "voice": None,
                }
                SS.messages.append(row)
        except Exception:
            SS.messages = []
    st.rerun()


def refresh_threads() -> None:
    try:
        SS.threads = client().list_threads()
    except Exception:
        SS.threads = []


# ---- auth gate ----

def auth_screen() -> None:
    st.markdown(
        "<div style='text-align:center; padding: 2rem 0 1rem;'>"
        "<div style='font-size:3rem;'>✨</div>"
        "<h1 style='margin:0.2rem 0;'>Lume</h1>"
        "<p style='color:#9b9bb5; margin-top:0;'>All-in-one local LLM gateway</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    with st.form("auth"):
        base = st.text_input("Server URL", value=SS.base_url, help="Where Lume is running")
        key = st.text_input("API key", value=SS.api_key, type="password",
                            placeholder="lume_…")
        col1, col2 = st.columns([1, 3])
        with col1:
            submitted = st.form_submit_button("Connect", use_container_width=True)
        with col2:
            health = Lume(base, key or "x").health() if base else {"status": "down"}
            st.caption(
                f"{'🟢 reachable' if health.get('ollama') else '🔴 unreachable'} · "
                f"model: `{health.get('model', '?')}`"
            )
        if submitted:
            SS.base_url = base
            SS.api_key = key
            SS.auth_error = None
            try:
                threads = Lume(base, key).list_threads()
                SS.threads = threads
                SS.authed = True
                st.rerun()
            except Exception as e:
                SS.auth_error = str(e)
                st.error(f"Could not connect: {e}")
    if SS.auth_error:
        st.error(SS.auth_error)


# ---- sidebar ----

def sidebar() -> None:
    with st.sidebar:
        st.markdown("### ✨ Lume")
        st.caption(f"`{SS.base_url}`")
        if st.button("🔌 Disconnect", use_container_width=True):
            SS.authed = False
            SS.api_key = ""
            SS.current_thread = None
            SS.messages = []
            st.rerun()

        st.divider()
        st.markdown("##### Threads")
        if st.button("➕ New chat", use_container_width=True, key="new_thread"):
            t = client().create_thread("New chat")
            refresh_threads()
            switch_thread(t["id"])
        refresh_threads()
        for t in SS.threads:
            tid = t["id"]
            title = t.get("title", "Untitled")
            active = SS.current_thread == tid
            cols = st.columns([5, 1, 1])
            with cols[0]:
                if st.button(
                    f"{'● ' if active else ''}{title}",
                    key=f"th_{tid}",
                    use_container_width=True,
                    help=f"Created {t.get('created_at','')}",
                ):
                    switch_thread(tid)
            with cols[1]:
                if st.button("✏️", key=f"rn_{tid}", help="Rename"):
                    new_title = st.text_input(
                        "new title", value=title, key=f"rni_{tid}",
                        label_visibility="collapsed",
                    )
                    if new_title and new_title != title:
                        client().rename_thread(tid, new_title)
                        refresh_threads()
                        st.rerun()
            with cols[2]:
                if st.button("🗑", key=f"del_{tid}", help="Delete"):
                    client().delete_thread(tid)
                    if SS.current_thread == tid:
                        switch_thread(None)
                    refresh_threads()
                    st.rerun()

        st.divider()
        st.markdown("##### Options")
        SS.system_prompt = st.text_area(
            "System prompt", value=SS.system_prompt, height=80,
            placeholder="Leave blank for Lume default",
            label_visibility="collapsed",
        )
        SS.voice_out = st.toggle("🔊 Voice replies", value=SS.voice_out, help="Synthesize TTS for each reply")


# ---- chat rendering ----

def render_history() -> None:
    for row in SS.messages:
        role = row["role"]
        if role == "tool":
            continue  # tool messages are shown inline under the assistant turn that issued them
        avatar = "🧑‍💻" if role == "user" else "✨"
        with st.chat_message(role, avatar=avatar):
            # attachments
            for att in row.get("attachments", []):
                kind = att.get("kind", "file")
                name = att.get("filename", "attachment")
                if kind == "image":
                    try:
                        st.image(client().fetch_bytes(f"/v1/attachments/{att['id']}"))
                    except Exception:
                        st.caption(f"🖼️ {name}")
                elif kind == "audio":
                    st.caption(f"🎙️ voice message: {name}")
                else:
                    st.caption(f"📎 {name}")
            if row.get("content"):
                st.markdown(row["content"])
            # voice reply player
            if row.get("voice"):
                try:
                    audio_bytes = client().fetch_bytes(row["voice"])
                    st.audio(audio_bytes, format="audio/wav", autoplay=True)
                except Exception:
                    st.caption("🔊 (voice unavailable)")
            # tool calls + results
            for tc in row.get("tool_calls", []):
                fn = tc.get("function", {})
                name = fn.get("name", "?")
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        pass
                with st.status(f"🔧 Tool: `{name}`", state="complete", expanded=False):
                    st.code(json.dumps(args, indent=2, ensure_ascii=False), language="json")
                    result = tc.get("_result")
                    if result:
                        st.code(str(result)[:4000], language="text")


# ---- main chat flow ----

def do_chat(prompt_text: str, files: list, audio: Any) -> None:
    # 1. upload any attachments first
    att_ids: list[int] = []
    att_meta: list[dict] = []
    for f in files or []:
        data = f.getvalue()
        aid = client().upload(f.name, data, f.type or "application/octet-stream")
        att_ids.append(aid)
        kind = "image" if (f.type or "").startswith("image/") else (
            "audio" if (f.type or "").startswith("audio/") else "file"
        )
        att_meta.append({"id": aid, "kind": kind, "filename": f.name})
    if audio is not None:
        data = audio.getvalue()
        aid = client().upload(audio.name or "voice.wav", data, "audio/wav")
        att_ids.append(aid)
        att_meta.append({"id": aid, "kind": "audio", "filename": audio.name or "voice.wav"})

    # 2. render user turn immediately
    user_row = {
        "role": "user",
        "content": prompt_text,
        "tool_calls": [],
        "attachments": att_meta,
        "voice": None,
    }
    SS.messages.append(user_row)
    with st.chat_message("user", avatar="🧑‍💻"):
        for att in att_meta:
            if att["kind"] == "image":
                try:
                    st.image(client().fetch_bytes(f"/v1/attachments/{att['id']}"))
                except Exception:
                    st.caption(f"🖼️ {att['filename']}")
            elif att["kind"] == "audio":
                st.caption(f"🎙️ voice message: {att['filename']}")
            else:
                st.caption(f"📎 {att['filename']}")
        if prompt_text:
            st.markdown(prompt_text)

    # 3. stream assistant reply
    with st.chat_message("assistant", avatar="✨"):
        token_gen = _stream_generator(
            prompt_text, att_ids, files_meta=att_meta,
        )
        full_text = st.write_stream(token_gen)

    # 4. finalize row (already appended inside _stream_generator)
    # nothing else needed; rerun to render cleanly with history
    refresh_threads()
    st.rerun()


def _stream_generator(prompt_text: str, att_ids: list[int], files_meta: list[dict]) -> Iterator[str]:
    """Yield token strings; capture tool calls / voice into the assistant row."""
    # prepare assistant row up-front so we can fill it as we go
    asst_row = {
        "role": "assistant",
        "content": "",
        "tool_calls": [],
        "attachments": [],
        "voice": None,
    }
    SS.messages.append(asst_row)

    resp, hdrs = client().chat_stream(
        prompt=prompt_text,
        thread_id=SS.current_thread,
        attachment_ids=att_ids,
        system=SS.system_prompt or None,
        voice=SS.voice_out,
    )
    # capture the (possibly new) thread id
    new_tid = hdrs.get("X-Lume-Thread-Id") or ""
    if new_tid and new_tid != SS.current_thread:
        SS.current_thread = new_tid

    pending_tool_calls: list = []
    try:
        for ev in sse_lines(resp):
            t = ev.get("type")
            if t == "token":
                piece = ev.get("content", "")
                asst_row["content"] += piece
                yield piece
            elif t == "tool_calls":
                pending_tool_calls = ev.get("tool_calls", [])
                asst_row["tool_calls"] = pending_tool_calls
            elif t == "tool_result":
                tc_id = ev.get("tool_call_id")
                result = ev.get("result", "")
                # attach the result back onto the matching tool call for later display
                for tc in pending_tool_calls:
                    if (tc.get("id") or tc.get("function", {}).get("name")) == tc_id:
                        tc["_result"] = result
                # emit a short status line into the stream so the user sees activity
                yield f"\n\n> 🔧 `{ev.get('name','?')}` → see details below\n\n"
            elif t == "voice":
                url = ev.get("url", "")
                asst_row["voice"] = url
            elif t == "error":
                msg = ev.get("message", "unknown error")
                asst_row["content"] += f"\n\n⚠️ {msg}"
                yield f"\n\n⚠️ {msg}"
            elif t == "done":
                break
    except Exception as e:
        err = f"\n\n⚠️ connection error: {e}"
        asst_row["content"] += err
        yield err


# ---- entrypoint ----

def main() -> None:
    st.set_page_config(
        page_title="Lume",
        page_icon="✨",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    # tiny css polish: brand the chat input + hide streamlit chrome
    st.markdown(
        """
        <style>
        .stChatInput { border-radius: 14px; }
        .stChatInput textarea { font-size: 15px; }
        [data-testid="stChatMessageContent"] { font-size: 15px; line-height: 1.55; }
        .stApp > header { background: transparent; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if not SS.authed:
        auth_screen()
        return

    sidebar()
    st.title("✨ Lume")
    if SS.current_thread:
        st.caption(f"thread `{SS.current_thread[:8]}…`")
    else:
        st.caption("no thread selected — a new one will be created on first message")

    render_history()

    prompt = st.chat_input(
        "Message Lume…  (attach files 📎 or record 🎙️)",
        accept_file="multiple",
        accept_audio=True,
        file_type=None,
        key="lume_chat_input",
    )

    if prompt is not None:
        text = ""
        files = []
        audio = None
        # prompt is dict-like when accept_file/accept_audio is set
        if isinstance(prompt, dict) or hasattr(prompt, "text"):
            text = getattr(prompt, "text", "") or prompt.get("text", "")
            files = getattr(prompt, "files", None) or prompt.get("files", []) or []
            audio = getattr(prompt, "audio", None) or prompt.get("audio", None)
        else:
            text = str(prompt)
        if text or files or audio:
            do_chat(text, files, audio)


if __name__ == "__main__":
    main()