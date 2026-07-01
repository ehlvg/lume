from __future__ import annotations

import aiosqlite

from .config import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS keys (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    key_hash    TEXT NOT NULL UNIQUE,        -- sha256 of plaintext key
    prefix      TEXT NOT NULL,                -- first 8 chars, for display only
    name        TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    revoked_at  TEXT,
    last_used_at TEXT,
    req_count   INTEGER NOT NULL DEFAULT 0,
    is_admin    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS threads (
    id          TEXT PRIMARY KEY,             -- uuid
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now')),
    title       TEXT NOT NULL DEFAULT 'Untitled thread',
    meta        TEXT NOT NULL DEFAULT '{}'    -- JSON blob
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id   TEXT NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
    seq         INTEGER NOT NULL,             -- monotonic within thread, 0-based
    role        TEXT NOT NULL,                -- user | assistant | tool
    content     TEXT NOT NULL DEFAULT '',     -- text content
    tool_calls  TEXT NOT NULL DEFAULT '[]',   -- JSON list
    tool_name   TEXT,                         -- set when role='tool'
    tool_call_id TEXT,                         -- links a tool result to a tool_call
    attachment_ids TEXT NOT NULL DEFAULT '[]', -- JSON list of attachment rowids
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (thread_id, seq)
);

CREATE TABLE IF NOT EXISTS attachments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    sha256      TEXT NOT NULL UNIQUE,
    size        INTEGER NOT NULL,
    mime        TEXT NOT NULL,
    filename    TEXT NOT NULL,
    path        TEXT NOT NULL,                -- absolute path on disk
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    kind        TEXT NOT NULL DEFAULT 'file'   -- file | image | audio | text
);

CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id, seq);
"""


async def init_db() -> None:
    s = get_settings()
    db = await aiosqlite.connect(str(s.db_path))
    try:
        await db.executescript(SCHEMA)
        await _seed_admin(db)
        await db.commit()
    finally:
        await db.close()


async def _seed_admin(db: aiosqlite.Connection) -> None:
    from .auth import hash_key

    s = get_settings()
    if not s.admin_bootstrap_key:
        return
    h = hash_key(s.admin_bootstrap_key)
    exists = await (await db.execute("SELECT 1 FROM keys WHERE key_hash=?", (h,))).fetchone()
    if exists:
        return
    prefix = s.admin_bootstrap_key[:8]
    await db.execute(
        "INSERT INTO keys (key_hash, prefix, name, is_admin) VALUES (?,?,?,1)",
        (h, prefix, "admin"),
    )


async def get_db() -> aiosqlite.Connection:
    s = get_settings()
    db = await aiosqlite.connect(str(s.db_path))
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA foreign_keys=ON")
    return db