from __future__ import annotations

import hashlib
import secrets

from .db import get_db

BEARER_PREFIX = "Bearer "


def hash_key(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def gen_key() -> str:
    return "lume_" + secrets.token_urlsafe(32)


async def auth_key(plaintext: str) -> tuple[bool, bool]:
    """Return (valid, is_admin)."""
    if not plaintext:
        return False, False
    h = hash_key(plaintext)
    db = await get_db()
    try:
        row = await (await db.execute(
            "SELECT id, is_admin, revoked_at FROM keys WHERE key_hash=?", (h,)
        )).fetchone()
        if row is None:
            return False, False
        if row["revoked_at"] is not None:
            return False, False
        await db.execute(
            "UPDATE keys SET last_used_at=datetime('now'), req_count=req_count+1 WHERE id=?",
            (row["id"],),
        )
        await db.commit()
        return True, bool(row["is_admin"])
    finally:
        await db.close()


async def create_key(name: str, is_admin: bool = False) -> str:
    plaintext = gen_key()
    h = hash_key(plaintext)
    db = await get_db()
    try:
        await db.execute(
            "INSERT INTO keys (key_hash, prefix, name, is_admin) VALUES (?,?,?,?)",
            (h, plaintext[:8], name, 1 if is_admin else 0),
        )
        await db.commit()
    finally:
        await db.close()
    return plaintext


async def list_keys() -> list[dict]:
    db = await get_db()
    try:
        rows = await (await db.execute(
            "SELECT id, prefix, name, is_admin, created_at, last_used_at, req_count, revoked_at "
            "FROM keys ORDER BY created_at DESC"
        )).fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def revoke_key(key_id: int) -> bool:
    db = await get_db()
    try:
        cur = await db.execute(
            "UPDATE keys SET revoked_at=datetime('now') WHERE id=? AND revoked_at IS NULL",
            (key_id,),
        )
        await db.commit()
        return cur.rowcount > 0
    finally:
        await db.close()