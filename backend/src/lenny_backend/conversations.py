from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from ai_pm_contracts import AnswerProcessTrace
from pydantic import BaseModel, ConfigDict


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class Conversation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    title: str
    created_at: str
    updated_at: str


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    conversation_id: str
    role: Literal["user", "assistant"]
    content: str
    created_at: str
    citations: tuple[dict[str, Any], ...] = ()
    mode: Literal["direct", "agentic"] | None = None
    process_trace: AnswerProcessTrace | None = None


class ConversationStore:
    """Small connection-per-operation SQLite store safe for FastAPI threads."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                ) STRICT;

                CREATE TABLE IF NOT EXISTS messages (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT NOT NULL UNIQUE,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    citations_json TEXT NOT NULL DEFAULT '[]',
                    mode TEXT CHECK (mode IS NULL OR mode IN ('direct', 'agentic')),
                    process_trace_json TEXT NOT NULL DEFAULT 'null'
                ) STRICT;

                CREATE INDEX IF NOT EXISTS messages_conversation_sequence
                ON messages(conversation_id, sequence);
                """
            )
            columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(messages)")
            }
            if "mode" not in columns:
                connection.execute(
                    "ALTER TABLE messages ADD COLUMN mode TEXT "
                    "CHECK (mode IS NULL OR mode IN ('direct', 'agentic'))"
                )
            if "process_trace_json" not in columns:
                connection.execute(
                    "ALTER TABLE messages ADD COLUMN process_trace_json TEXT "
                    "NOT NULL DEFAULT 'null'"
                )

    def create(self, title: str | None = None) -> Conversation:
        timestamp = _now()
        conversation = Conversation(
            id=str(uuid.uuid4()),
            title=(title or "新对话").strip() or "新对话",
            created_at=timestamp,
            updated_at=timestamp,
        )
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO conversations(id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (
                    conversation.id,
                    conversation.title,
                    conversation.created_at,
                    conversation.updated_at,
                ),
            )
        return conversation

    def list(self) -> tuple[Conversation, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, title, created_at, updated_at "
                "FROM conversations ORDER BY updated_at DESC, id DESC"
            ).fetchall()
        return tuple(Conversation.model_validate(dict(row)) for row in rows)

    def get(self, conversation_id: str) -> Conversation | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
        return None if row is None else Conversation.model_validate(dict(row))

    def rename(self, conversation_id: str, title: str) -> Conversation | None:
        normalized = " ".join(title.split())
        if not normalized or len(normalized) > 200:
            raise ValueError("conversation title must contain 1 to 200 characters")
        timestamp = _now()
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (normalized, timestamp, conversation_id),
            )
        if cursor.rowcount == 0:
            return None
        return self.get(conversation_id)

    def delete(self, conversation_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM conversations WHERE id = ?", (conversation_id,)
            )
        return cursor.rowcount > 0

    def add_message(
        self,
        conversation_id: str,
        role: Literal["user", "assistant"],
        content: str,
        citations: tuple[dict[str, Any], ...] = (),
        mode: Literal["direct", "agentic"] | None = None,
        process_trace: AnswerProcessTrace | None = None,
    ) -> Message:
        message = Message(
            id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            role=role,
            content=content,
            created_at=_now(),
            citations=citations,
            mode=mode,
            process_trace=process_trace,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO messages(
                    id, conversation_id, role, content, created_at, citations_json, mode,
                    process_trace_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.id,
                    message.conversation_id,
                    message.role,
                    message.content,
                    message.created_at,
                    json.dumps(message.citations, ensure_ascii=False, separators=(",", ":")),
                    message.mode,
                    json.dumps(
                        message.process_trace.model_dump(mode="json")
                        if message.process_trace
                        else None,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                ),
            )
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (message.created_at, conversation_id),
            )
        return message

    def messages(self, conversation_id: str) -> tuple[Message, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, conversation_id, role, content, created_at, citations_json, mode,
                       process_trace_json
                FROM messages WHERE conversation_id = ? ORDER BY sequence ASC
                """,
                (conversation_id,),
            ).fetchall()
        return tuple(self._message_from_row(row) for row in rows)

    @staticmethod
    def _message_from_row(row: sqlite3.Row) -> Message:
        citations = json.loads(str(row["citations_json"]))
        raw_trace = json.loads(str(row["process_trace_json"]))
        return Message(
            id=row["id"],
            conversation_id=row["conversation_id"],
            role=row["role"],
            content=row["content"],
            created_at=row["created_at"],
            citations=tuple(citations),
            mode=row["mode"],
            process_trace=(
                AnswerProcessTrace.model_validate(raw_trace)
                if raw_trace is not None
                else None
            ),
        )
