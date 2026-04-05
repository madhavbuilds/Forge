"""SQLite prompt/response cache keyed by sha256(prompt + context)."""

from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path
def _digest(prompt: str, context: str) -> str:
    h = hashlib.sha256()
    h.update(prompt.encode("utf-8", errors="replace"))
    h.update(b"\n")
    h.update(context.encode("utf-8", errors="replace"))
    return h.hexdigest()


class PromptCache:
    def __init__(self, db_path: Path, max_entries: int, enabled: bool) -> None:
        self.db_path = db_path
        self.max_entries = max_entries
        self.enabled = enabled
        if enabled:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as cx:
            cx.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created REAL NOT NULL
                )
                """
            )
            cx.commit()

    def get(self, prompt: str, context: str) -> str | None:
        if not self.enabled:
            return None
        key = _digest(prompt, context)
        with sqlite3.connect(self.db_path) as cx:
            row = cx.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set(self, prompt: str, context: str, value: str) -> None:
        if not self.enabled:
            return
        key = _digest(prompt, context)
        now = time.time()
        with sqlite3.connect(self.db_path) as cx:
            cx.execute(
                "INSERT OR REPLACE INTO cache (key, value, created) VALUES (?, ?, ?)",
                (key, value, now),
            )
            n = cx.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            if n > self.max_entries:
                cx.execute(
                    """
                    DELETE FROM cache WHERE key IN (
                        SELECT key FROM cache ORDER BY created ASC LIMIT ?
                    )
                    """,
                    (n - self.max_entries,),
                )
            cx.commit()

    def remember_session(self, snippets: list[str]) -> None:
        """Store session lines for debugging / future use (minimal)."""
        if not self.enabled or not snippets:
            return
        with sqlite3.connect(self.db_path) as cx:
            cx.execute(
                """
                CREATE TABLE IF NOT EXISTS session (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    line TEXT NOT NULL,
                    created REAL NOT NULL
                )
                """
            )
            now = time.time()
            for s in snippets[-500:]:
                cx.execute("INSERT INTO session (line, created) VALUES (?, ?)", (s, now))
            cx.commit()
