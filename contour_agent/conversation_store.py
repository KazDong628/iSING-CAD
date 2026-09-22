"""Small durable conversation journal for the browser agent workbench."""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _now():
    return datetime.now(timezone.utc).isoformat()


class ConversationStore:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()

    def _path(self, conversation_id: str):
        if len(conversation_id) != 32 or any(ch not in "0123456789abcdef" for ch in conversation_id):
            raise KeyError(conversation_id)
        return self.directory / f"{conversation_id}.json"

    def save(self, conversation: dict):
        path = self._path(conversation["id"])
        conversation["updated_at"] = _now()
        temporary = path.with_suffix(".json.tmp")
        with self.lock:
            temporary.write_text(json.dumps(conversation, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(path)
        return conversation

    def create(self, title="新建主轮廓任务"):
        conversation_id = uuid.uuid4().hex
        now = _now()
        value = {
            "id": conversation_id,
            "title": str(title).strip()[:80] or "新建主轮廓任务",
            "created_at": now,
            "updated_at": now,
            "messages": [],
            "memory": {"last_job_id": None, "job_ids": [], "turn_count": 0},
        }
        return self.save(value)

    def get(self, conversation_id: str):
        path = self._path(conversation_id)
        if not path.is_file():
            raise KeyError(conversation_id)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            raise KeyError(conversation_id) from None
        if not isinstance(value, dict) or value.get("id") != conversation_id:
            raise KeyError(conversation_id)
        return value

    def list(self, limit=100):
        rows = []
        for path in self.directory.glob("*.json"):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(value, dict) and value.get("id") == path.stem:
                    rows.append(value)
            except (OSError, ValueError, TypeError):
                continue
        rows.sort(key=lambda row: row.get("updated_at", ""), reverse=True)
        return rows[: max(1, min(200, int(limit)))]

    def add_message(self, conversation_id: str, role: str, text: str, **extra):
        if role not in {"user", "assistant", "system"}:
            raise ValueError("Unsupported conversation role")
        value = self.get(conversation_id)
        message = {"id": uuid.uuid4().hex, "role": role, "text": str(text)[:4000], "time": _now(), **extra}
        value.setdefault("messages", []).append(message)
        value.setdefault("memory", {})["turn_count"] = sum(row.get("role") == "user" for row in value["messages"])
        self.save(value)
        return message, value

    def remember_job(self, conversation_id: str, job_id: str):
        value = self.get(conversation_id)
        memory = value.setdefault("memory", {})
        ids = memory.setdefault("job_ids", [])
        if job_id not in ids:
            ids.append(job_id)
        memory["last_job_id"] = job_id
        self.save(value)
        return value

    def delete(self, conversation_id: str):
        """Delete one conversation journal and report whether it existed."""
        path = self._path(conversation_id)
        with self.lock:
            if not path.is_file():
                raise KeyError(conversation_id)
            path.unlink()
        return True
