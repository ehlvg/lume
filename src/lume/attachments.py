from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .config import get_settings
from .db import get_db

# Heuristic: order matters — checked before generic.
_TEXT_EXTS = {".txt", ".md", ".rst", ".log", ".csv", ".tsv", ".json", ".yaml", ".yml",
              ".toml", ".ini", ".cfg", ".py", ".js", ".ts", ".tsx", ".jsx", ".go",
              ".rs", ".c", ".h", ".cpp", ".hpp", ".cc", ".java", ".kt", ".swift",
              ".rb", ".php", ".sh", ".bash", ".zsh", ".sql", ".lua", ".dart",
              ".html", ".htm", ".css", ".scss", ".vue", ".svelte"}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif"}
_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma"}


def _guess_kind(filename: str, mime: str | None) -> str:
    ext = Path(filename).suffix.lower()
    if ext in _IMAGE_EXTS or (mime and mime.startswith("image/")):
        return "image"
    if ext in _AUDIO_EXTS or (mime and mime.startswith("audio/")):
        return "audio"
    if ext in _TEXT_EXTS or (mime and mime.startswith("text/")):
        return "text"
    if ext == ".pdf" or (mime == "application/pdf"):
        return "text"
    return "file"


async def store_upload(data: bytes, filename: str, mime: str | None) -> dict[str, Any]:
    s = get_settings()
    sha = hashlib.sha256(data).hexdigest()
    dest = s.attachments_dir / f"{sha}{Path(filename).suffix or '.bin'}"

    db = await get_db()
    try:
        row = await (await db.execute(
            "SELECT id, sha256, size, mime, filename, path, kind FROM attachments WHERE sha256=?",
            (sha,),
        )).fetchone()
        if row is None:
            if not dest.exists():
                dest.write_bytes(data)
            kind = _guess_kind(filename, mime)
            await db.execute(
                "INSERT INTO attachments (sha256, size, mime, filename, path, kind) "
                "VALUES (?,?,?,?,?,?)",
                (sha, len(data), mime or _mime_from_ext(filename), filename, str(dest), kind),
            )
            await db.commit()
            row = await (await db.execute(
                "SELECT id, sha256, size, mime, filename, path, kind FROM attachments WHERE sha256=?",
                (sha,),
            )).fetchone()
        return dict(row)
    finally:
        await db.close()


def _mime_from_ext(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    return {
        ".txt": "text/plain", ".md": "text/markdown", ".json": "application/json",
        ".pdf": "application/pdf", ".png": "image/png", ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp",
        ".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4",
    }.get(ext, "application/octet-stream")


async def get_attachment(attachment_id: int) -> dict | None:
    db = await get_db()
    try:
        row = await (await db.execute(
            "SELECT id, sha256, size, mime, filename, path, kind FROM attachments WHERE id=?",
            (attachment_id,),
        )).fetchone()
        return dict(row) if row else None
    finally:
        await db.close()


async def list_attachments(ids: list[int]) -> list[dict]:
    if not ids:
        return []
    q = ",".join("?" * len(ids))
    db = await get_db()
    try:
        rows = await (await db.execute(
            f"SELECT id, sha256, size, mime, filename, path, kind FROM attachments "
            f"WHERE id IN ({q}) ORDER BY id",
            ids,
        )).fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


def text_from_attachment(att: dict) -> str:
    """Extract plain text for inclusion in the model context. Returns '' for binary blobs."""
    path = Path(att["path"])
    kind = att["kind"]
    mime = att["mime"]
    try:
        if kind == "text" and mime == "application/pdf":
            return _pdf_to_text(path)
        if kind == "text":
            return path.read_text(encoding="utf-8", errors="replace")
        return ""
    except Exception:
        return ""


def _pdf_to_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        return ""
    try:
        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception:
        return ""


def file_for_ollama(att: dict) -> dict | None:
    """Return the ollama-format attachment dict, or None if the file should be text-inlined."""
    if att["kind"] in ("image",):
        return {"image_path": att["path"]}
    return None


async def cleanup_orphans() -> int:
    """Remove attachment files not referenced by any message."""
    s = get_settings()
    db = await get_db()
    try:
        rows = await (await db.execute("SELECT id, path FROM attachments")).fetchall()
        # collect all attachment_ids referenced by messages
        used = set()
        msg_rows = await (await db.execute("SELECT attachment_ids FROM messages")).fetchall()
        for m in msg_rows:
            for i in json.loads(m["attachment_ids"] or "[]"):
                used.add(int(i))
        removed = 0
        for r in rows:
            if int(r["id"]) not in used:
                p = Path(r["path"])
                if p.exists():
                    p.unlink(missing_ok=True)
                await db.execute("DELETE FROM attachments WHERE id=?", (r["id"],))
                removed += 1
        await db.commit()
        return removed
    finally:
        await db.close()


# --- Voice (whisper transcription) ---

def transcribe_audio(path: str | Path, language: str | None = None) -> str:
    """Transcribe an audio file with openai-whisper. Tries the CLI first, then the python package."""
    s = get_settings()
    p = Path(path)
    # Try the CLI binary (cheaper to invoke, no model-load in our process).
    if Path(s.whisper_bin).exists():
        out_dir = s.voice_dir
        cmd = [
            s.whisper_bin, str(p),
            "--model", s.whisper_model, "--device", s.whisper_device,
            "--output_dir", str(out_dir), "--output_format", "txt", "--verbose", "False",
        ]
        if language:
            cmd += ["--language", language]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=600)
            out_txt = out_dir / f"{p.stem}.txt"
            if out_txt.exists():
                return out_txt.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            pass  # fall through to package
    # Fall back to the python package (load model in-process).
    import whisper  # type: ignore
    model = whisper.load_model(s.whisper_model, device=s.whisper_device)
    result = model.transcribe(str(p), language=language)
    return (result.get("text") or "").strip()