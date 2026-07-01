from __future__ import annotations

import json
import uuid
from typing import Any

from .db import get_db


async def create_thread(title: str = "Untitled thread", meta: dict | None = None) -> dict:
    tid = str(uuid.uuid4())
    db = await get_db()
    try:
        await db.execute(
            "INSERT INTO threads (id, title, meta) VALUES (?,?,?)",
            (tid, title, json.dumps(meta or {})),
        )
        await db.commit()
    finally:
        await db.close()
    return await get_thread(tid)


async def get_thread(tid: str) -> dict | None:
    db = await get_db()
    try:
        row = await (await db.execute("SELECT * FROM threads WHERE id=?", (tid,))).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["meta"] = json.loads(d["meta"] or "{}")
        return d
    finally:
        await db.close()


async def list_threads() -> list[dict]:
    db = await get_db()
    try:
        rows = await (await db.execute(
            "SELECT id, created_at, updated_at, title, meta FROM threads "
            "ORDER BY updated_at DESC"
        )).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["meta"] = json.loads(d["meta"] or "{}")
            out.append(d)
        return out
    finally:
        await db.close()


async def rename_thread(tid: str, title: str) -> dict | None:
    db = await get_db()
    try:
        await db.execute(
            "UPDATE threads SET title=?, updated_at=datetime('now') WHERE id=?",
            (title, tid),
        )
        await db.commit()
    finally:
        await db.close()
    return await get_thread(tid)


async def delete_thread(tid: str) -> bool:
    db = await get_db()
    try:
        cur = await db.execute("DELETE FROM threads WHERE id=?", (tid,))
        await db.commit()
        return cur.rowcount > 0
    finally:
        await db.close()


async def append_message(
    thread_id: str,
    role: str,
    content: str = "",
    tool_calls: list | None = None,
    tool_name: str | None = None,
    tool_call_id: str | None = None,
    attachment_ids: list[int] | None = None,
) -> dict:
    db = await get_db()
    try:
        seq_row = await (await db.execute(
            "SELECT COALESCE(MAX(seq), -1)+1 AS next FROM messages WHERE thread_id=?",
            (thread_id,),
        )).fetchone()
        seq = int(seq_row["next"])
        await db.execute(
            "INSERT INTO messages (thread_id, seq, role, content, tool_calls, tool_name, "
            "tool_call_id, attachment_ids) VALUES (?,?,?,?,?,?,?,?)",
            (
                thread_id, seq, role, content,
                json.dumps(tool_calls or []),
                tool_name, tool_call_id,
                json.dumps(attachment_ids or []),
            ),
        )
        await db.execute("UPDATE threads SET updated_at=datetime('now') WHERE id=?", (thread_id,))
        await db.commit()
        row = await (await db.execute("SELECT * FROM messages WHERE thread_id=? AND seq=?",
                                     (thread_id, seq))).fetchone()
        return _row_to_msg(row)
    finally:
        await db.close()


def _row_to_msg(row) -> dict:
    d = dict(row)
    d["tool_calls"] = json.loads(d["tool_calls"] or "[]")
    d["attachment_ids"] = json.loads(d["attachment_ids"] or "[]")
    return d


async def list_messages(thread_id: str, limit: int = 1000) -> list[dict]:
    db = await get_db()
    try:
        rows = await (await db.execute(
            "SELECT * FROM messages WHERE thread_id=? ORDER BY seq ASC LIMIT ?",
            (thread_id, limit),
        )).fetchall()
        return [_row_to_msg(r) for r in rows]
    finally:
        await db.close()


async def trim_to_context(thread_id: str, max_messages: int = 50) -> list[dict]:
    msgs = await list_messages(thread_id, limit=max_messages)
    return msgs