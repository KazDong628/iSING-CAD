"""SQLite job journal; a browser reconnect never discards completed stages."""
from __future__ import annotations
import json
import sqlite3
import threading
import os
from pathlib import Path

class RuntimeLease:
    """An OS-held lock; stale lock files do not prevent process restart."""
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.stream = (directory / "server.lock").open("a+b")
        self.stream.seek(0)
        self.stream.write(b"0")
        self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.stream.close()
            raise ValueError("已有工作台使用此运行目录，请使用现有服务或独立CONTOUR运行目录。") from None

    def close(self):
        if self.stream.closed:
            return
        self.stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        self.stream.close()

class JobStore:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "jobs.sqlite3"
        self.lock = threading.RLock()
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, updated TEXT NOT NULL, document TEXT NOT NULL)")

    def connect(self):
        return sqlite3.connect(self.path, timeout=15)

    def save(self, job: dict):
        raw = json.dumps(job, ensure_ascii=False, allow_nan=False)
        with self.lock, self.connect() as db:
            db.execute("INSERT INTO jobs VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET updated=excluded.updated,document=excluded.document", (job["id"], job["updated_at"], raw))

    def get(self, job_id: str) -> dict:
        with self.connect() as db:
            row = db.execute("SELECT document FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return json.loads(row[0])

    def list(self, limit=100):
        with self.connect() as db:
            rows = db.execute("SELECT document FROM jobs ORDER BY updated DESC LIMIT ?", (limit,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def delete(self, job_id: str):
        """Delete one durable job record and report whether it existed."""
        with self.lock, self.connect() as db:
            cursor = db.execute("DELETE FROM jobs WHERE id=?", (job_id,))
            return cursor.rowcount == 1
