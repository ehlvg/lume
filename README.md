# Lume

All-in-one local LLM gateway. One model, one server, one key per client — chat, threads,
attachments, voice, and tools in a tiny, non-OpenAI HTTP API.

Lume fronts an always-running local Ollama (`gemma4:12b-mlx` by default) and exposes a
deliberately small surface area: callers never pick a model, never deal with Ollama's
options, and never have to assemble message history — they pass a prompt (optionally with
files/voice) inside a thread, and Lume streams typed SSE events back.

A bundled Streamlit web UI (`ui.py`) gives you a ChatGPT-like daily driver on top of the
API — threads sidebar, file + audio attachments, streaming replies, tool-call visibility,
and voice output.

## Features

- **Single model** — configured server-side, never sent by the client.
- **Threads** — start, list, fetch, rename, delete. History is replayed automatically.
- **Attachments of any kind** — images, PDFs, code, text, audio. PDFs/text are extracted
  and inlined into context; images are passed to Ollama's vision; audio is transcribed
  server-side with Whisper and inlined as text.
- **Voice in** — send a `.mp3`/`.wav`/`.m4a`/… as an attachment, Lume runs
  `openai-whisper` and feeds the transcript to the model.
- **Voice out** — pass `voice=true` and Lume synthesizes the final reply with
  [Supertonic](https://github.com/supertone-inc/supertonic) (`lang="na"` auto-language),
  returning a `/v1/voice/<name>` URL in the SSE stream.
- **Tool calling** — built-in tools (`shell`, `http_get`, `now`) always available,
  caller-supplied tool schemas are merged in. Lume runs the tool dispatch loop and streams
  `tool_calls` / `tool_result` events.
- **Per-client API keys** — stored in SQLite, revocable, with usage tracking. Bootstrap
  an admin key with `LUME_ADMIN_BOOTSTRAP_KEY`, then create per-client keys via the admin
  API.

## Quick start

```bash
# 1. install deps (core + voice)
uv sync --extra voice
# or just core (TTS/STT disabled):
uv sync

# 2. configure
cp .env.example .env
# edit .env: set LUME_ADMIN_BOOTSTRAP_KEY to a long random string

# 3. ensure ollama has the model loaded
ollama pull gemma4:12b-mlx   # if not already present
ollama run   gemma4:12b-mlx  # keep it warm in another terminal

# 4. run
uv run lume         # or: uv run python main.py
```

## Web UI

A Streamlit chat app lives in `src/lume/ui.py`. It talks to the Lume API over HTTP and
gives you a daily-driver chat interface with:

- **Threads sidebar** — create, rename, delete, switch; history auto-restores on switch.
- **Streaming replies** — token-by-token typewriter via `st.write_stream`.
- **File + audio attachments** — `st.chat_input` with `accept_file="multiple"` and
  `accept_audio=True`; uploads go to the API, which transcribes voice server-side.
- **Tool-call visibility** — each tool round renders as a collapsible `st.status` showing
  the call args and result.
- **Voice replies** — toggle 🔊 in the sidebar; the API synthesizes TTS and the UI
  autoplays the returned WAV.
- **System prompt** — editable in the sidebar, sent with each request.
- **Themed** — Lume brand colors in `.streamlit/config.toml`, emoji avatars (🧑‍💻 user,
  ✨ assistant).

```bash
# start the API first (see Quick start), then:
uv run streamlit run src/lume/ui.py
# or with the console script:
uv run lume-ui
```

Open http://localhost:8501, paste your API key, and chat. The server URL defaults to
`http://127.0.0.1:8000` (override with `LUME_UI_BASE_URL`).

## API surface

All endpoints require `Authorization: Bearer <key>` (except `/health`).

### Chat (streaming SSE) — `POST /v1/chat`

`multipart/form-data`:

| field           | type   | notes                                                          |
|-----------------|--------|----------------------------------------------------------------|
| `prompt`        | str    | user message text (optional if you only send attachments)       |
| `thread_id`     | str    | optional; if omitted, an ephemeral thread is created           |
| `system`        | str    | optional system-prompt override                                 |
| `voice`         | bool   | also synthesize TTS of the final reply                          |
| `tools`         | str    | JSON array of OpenAI-style tool schemas, merged with built-ins   |
| `options`       | str    | JSON ollama options (temperature, num_ctx, …)                   |
| `attachment_ids`| str    | JSON list of attachment ids previously uploaded                 |
| `files`         | files  | zero or more new files to upload-and-include in this turn       |

Response: `text/event-stream`. The `X-Lume-Thread-Id` response header carries the thread
id (useful for ephemeral threads). Events:

```
data: {"type":"token","content":"..."}            # incremental text
data: {"type":"tool_calls","tool_calls":[...]}     # model wants to call tools
data: {"type":"tool_result","tool_call_id":"...","name":"...","result":"..."}
data: {"type":"voice","url":"/v1/voice/<name>","name":"..."}
data: {"type":"done"}
```

### Non-streaming variant — `POST /v1/chat/json`

JSON body `{prompt, thread_id?, system?, voice?, tools?, options?, attachment_ids?}`.
Returns `{thread_id, content, tool_calls, voice?}`.

### Threads

| Method   | Path                          | Body / Query          |
|----------|-------------------------------|-----------------------|
| `GET`    | `/v1/threads`                 | —                     |
| `POST`   | `/v1/threads`                 | `title`               |
| `GET`    | `/v1/threads/{tid}`           | —                     |
| `PATCH`  | `/v1/threads/{tid}`           | `title`               |
| `DELETE`| `/v1/threads/{tid}`           | —                     |
| `GET`    | `/v1/threads/{tid}/messages`  | —                     |

### Attachments

| Method | Path                          | Notes                |
|--------|-------------------------------|----------------------|
| `POST` | `/v1/attachments`             | multipart `file`     |
| `GET`  | `/v1/attachments/{aid}`       | raw bytes            |
| `GET`  | `/v1/attachments/{aid}/meta`   | json metadata        |

Attachments are content-addressed by sha256 (deduped on disk). PDFs and text-like files are
extracted into plain text and inlined into context; images go to Ollama's vision; audio is
transcribed with Whisper.

### Voice

| Method | Path                | Notes                                |
|--------|---------------------|--------------------------------------|
| `POST` | `/v1/transcribe`   | multipart `file`, optional `language` |
| `GET`  | `/v1/voice/{name}`  | fetch a synthesized reply             |

### Admin (requires an admin key)

| Method   | Path                  | Notes                          |
|----------|-----------------------|--------------------------------|
| `GET`    | `/v1/admin/keys`      | list keys (no plaintext)       |
| `POST`   | `/v1/admin/keys`      | `name`, `is_admin` → `{key}`    |
| `DELETE`| `/v1/admin/keys/{id}` | revoke                          |

## Built-in tools

Server-defined tools the model can call (always available, can be shadowed by caller
tools of the same name):

- `shell { cmd }` — run a shell command on the server, return stdout/stderr + exit code.
- `http_get { url, max_chars? }` — GET a URL, return up to `max_chars` of body text.
- `now { }` — current server time in ISO format.

Caller-supplied tools (passed in the `tools` form field) are advertised to the model. If a
caller tool name collides with a built-in, the caller's schema wins but the server can't
execute it (caller-defined) — the model will see a placeholder tool result and may ask
again; the caller can inject the real result by sending a follow-up user message in the
same thread.

## Example

```bash
# create a client key (with an admin key)
ADMIN=lume_xxx
curl -X POST http://localhost:8000/v1/admin/keys?name=phone \
  -H "Authorization: Bearer $ADMIN" -d is_admin=false
# → {"key":"lume_yyy", ...}

KEY=lume_yyy

# start a thread and chat
curl -N http://localhost:8000/v1/chat \
  -H "Authorization: Bearer $KEY" \
  -F prompt="Summarize this PDF" \
  -F files=@paper.pdf

# the response stream carries X-Lume-Thread-Id; reuse it to continue
TID=...
curl -N http://localhost:8000/v1/chat \
  -H "Authorization: Bearer $KEY" \
  -F prompt="Now give me the 3 main takeaways" \
  -F thread_id=$TID

# voice reply
curl -N http://localhost:8000/v1/chat \
  -H "Authorization: Bearer $KEY" \
  -F prompt="Read me the summary aloud" \
  -F thread_id=$TID \
  -F voice=true

# send a voice message
curl -N http://localhost:8000/v1/chat \
  -H "Authorization: Bearer $KEY" \
  -F files=@memo.m4a \
  -F thread_id=$TID
```

## Configuration

All settings via environment variables (prefix `LUME_`), see `.env.example`.

## Project layout

```
src/lume/
  app.py            FastAPI routes, SSE loop, tool dispatch
  ollama.py         streaming ollama client
  threads.py        thread + message persistence
  attachments.py    uploads, dedup, text extraction, whisper transcription
  tools.py          built-in tool registry + dispatcher
  auth.py           per-client API keys
  db.py             sqlite schema + connection helper
  config.py         pydantic settings
  ui.py             Streamlit web chat UI
.streamlit/
  config.toml       Lume theme (colors, fonts, emoji avatars)
data/
  lume.db
  attachments/      content-addressed files
  voice/            whisper txt outputs + supertonic wav outputs
```

## Notes & limits

- History is replayed verbatim; images sent in prior turns are *not* re-attached (Ollama
  message format limitation). Send the image again if the model needs to see it.
- The tool loop is bounded by `LUME_MAX_TOOL_ITERATIONS` (default 6).
- `shell` tool runs on the server as the Lume process owner. Only grant keys to people you
  trust, or disable it by editing `tools.py`.