from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from ai_pm_contracts import AnswerProcessTrace
from pydantic import BaseModel, ConfigDict, Field

AnswerMode = Literal["direct", "agentic"]
AnswerKind = Literal["grounded", "insufficient_evidence"]
AnswerContractVersion = Literal["v1", "v2"]
ResponseKind = Literal[
    "grounded_answer",
    "general_guidance",
    "clarification",
    "insufficient_evidence",
]
RunStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _lease_deadline(seconds: int) -> str:
    return (
        (datetime.now(UTC) + timedelta(seconds=seconds))
        .isoformat()
        .replace("+00:00", "Z")
    )


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
    claims: tuple[dict[str, Any], ...] = ()
    mode: AnswerMode | None = None
    answer_kind: AnswerKind | None = None
    contract_version: AnswerContractVersion = "v1"
    response_kind: ResponseKind | None = None
    response_payload: dict[str, Any] = Field(default_factory=dict)
    process_trace: AnswerProcessTrace | None = None


class AnswerRun(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    conversation_id: str
    idempotency_key: str | None = None
    question: str
    mode: AnswerMode
    status: RunStatus
    user_message_id: str
    assistant_message_id: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool = False
    owner_id: str | None = None
    lease_expires_at: str | None = None
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None


class ConversationBusyError(RuntimeError):
    def __init__(self, active_run: AnswerRun):
        super().__init__("conversation already has an active answer run")
        self.active_run = active_run


class IdempotencyConflictError(RuntimeError):
    pass


class ConversationNotFoundError(RuntimeError):
    pass


class ConversationStore:
    """Connection-per-operation SQLite store with durable answer-run state."""

    def __init__(self, path: str | Path, *, lease_seconds: int = 360):
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        self.path = Path(path)
        self.instance_id = str(uuid.uuid4())
        self.lease_seconds = lease_seconds
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    @contextmanager
    def _connection(
        self, *, write: bool = False, immediate: bool = False
    ) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            if immediate:
                connection.execute("BEGIN IMMEDIATE")
            yield connection
            if write:
                connection.commit()
        except BaseException:
            if write:
                connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection(write=True) as connection:
            self._enable_wal(connection)
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
                    process_trace_json TEXT NOT NULL DEFAULT 'null',
                    claims_json TEXT NOT NULL DEFAULT '[]',
                    answer_kind TEXT CHECK (
                        answer_kind IS NULL OR answer_kind IN (
                            'grounded', 'insufficient_evidence'
                        )
                    ),
                    contract_version TEXT NOT NULL DEFAULT 'v1' CHECK (
                        contract_version IN ('v1', 'v2')
                    ),
                    response_kind TEXT CHECK (
                        response_kind IS NULL OR response_kind IN (
                            'grounded_answer', 'general_guidance', 'clarification',
                            'insufficient_evidence'
                        )
                    ),
                    response_payload_json TEXT NOT NULL DEFAULT '{}'
                ) STRICT;
                CREATE INDEX IF NOT EXISTS messages_conversation_sequence
                ON messages(conversation_id, sequence);
                CREATE TABLE IF NOT EXISTS answer_runs (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT NOT NULL UNIQUE,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    idempotency_key TEXT,
                    question TEXT NOT NULL,
                    mode TEXT NOT NULL CHECK (mode IN ('direct', 'agentic')),
                    status TEXT NOT NULL CHECK (
                        status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')
                    ),
                    user_message_id TEXT NOT NULL REFERENCES messages(id),
                    assistant_message_id TEXT REFERENCES messages(id),
                    error_code TEXT,
                    error_message TEXT,
                    retryable INTEGER NOT NULL DEFAULT 0 CHECK (retryable IN (0, 1)),
                    owner_id TEXT,
                    lease_expires_at TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    UNIQUE(conversation_id, idempotency_key)
                ) STRICT;
                CREATE INDEX IF NOT EXISTS answer_runs_conversation_sequence
                ON answer_runs(conversation_id, sequence DESC);
                CREATE UNIQUE INDEX IF NOT EXISTS answer_runs_one_active_conversation
                ON answer_runs(conversation_id)
                WHERE status IN ('queued', 'running');
                """
            )
            # executescript commits its own DDL transaction. Acquire the write
            # lock before inspecting columns so concurrent workers cannot both
            # attempt the same legacy migration.
            connection.execute("BEGIN IMMEDIATE")
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
            if "claims_json" not in columns:
                connection.execute(
                    "ALTER TABLE messages ADD COLUMN claims_json TEXT "
                    "NOT NULL DEFAULT '[]'"
                )
            if "answer_kind" not in columns:
                connection.execute(
                    "ALTER TABLE messages ADD COLUMN answer_kind TEXT "
                    "CHECK (answer_kind IS NULL OR answer_kind IN "
                    "('grounded', 'insufficient_evidence'))"
                )
            if "contract_version" not in columns:
                connection.execute(
                    "ALTER TABLE messages ADD COLUMN contract_version TEXT "
                    "NOT NULL DEFAULT 'v1' CHECK (contract_version IN ('v1', 'v2'))"
                )
            if "response_kind" not in columns:
                connection.execute(
                    "ALTER TABLE messages ADD COLUMN response_kind TEXT "
                    "CHECK (response_kind IS NULL OR response_kind IN ("
                    "'grounded_answer', 'general_guidance', 'clarification', "
                    "'insufficient_evidence'))"
                )
            if "response_payload_json" not in columns:
                connection.execute(
                    "ALTER TABLE messages ADD COLUMN response_payload_json TEXT "
                    "NOT NULL DEFAULT '{}'"
                )
            run_columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(answer_runs)")
            }
            if "owner_id" not in run_columns:
                connection.execute("ALTER TABLE answer_runs ADD COLUMN owner_id TEXT")
            if "lease_expires_at" not in run_columns:
                connection.execute(
                    "ALTER TABLE answer_runs ADD COLUMN lease_expires_at TEXT"
                )
            connection.execute(
                """UPDATE answer_runs SET lease_expires_at = ?
                WHERE status IN ('queued', 'running') AND lease_expires_at IS NULL""",
                (_lease_deadline(self.lease_seconds),),
            )
            self._expire_runs(connection, _now())

    @staticmethod
    def _enable_wal(connection: sqlite3.Connection) -> None:
        for attempt in range(20):
            try:
                connection.execute("PRAGMA journal_mode = WAL").fetchone()
                return
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).casefold() or attempt == 19:
                    raise
                time.sleep(min(0.01 * (attempt + 1), 0.1))

    def create(self, title: str | None = None) -> Conversation:
        timestamp = _now()
        conversation = Conversation(
            id=str(uuid.uuid4()),
            title=(title or "新对话").strip() or "新对话",
            created_at=timestamp,
            updated_at=timestamp,
        )
        with self._connection(write=True) as connection:
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

    def list(self, *, limit: int = 100) -> tuple[Conversation, ...]:
        _validate_limit(limit)
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT id, title, created_at, updated_at "
                "FROM conversations ORDER BY updated_at DESC, id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return tuple(Conversation.model_validate(dict(row)) for row in rows)

    def get(self, conversation_id: str) -> Conversation | None:
        with self._connection() as connection:
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
        with self._connection(write=True) as connection:
            cursor = connection.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (normalized, timestamp, conversation_id),
            )
        if cursor.rowcount == 0:
            return None
        return self.get(conversation_id)

    def delete(self, conversation_id: str) -> bool:
        with self._connection(write=True, immediate=True) as connection:
            self._expire_runs(connection, _now())
            active = connection.execute(
                """SELECT * FROM answer_runs
                WHERE conversation_id = ? AND status IN ('queued', 'running')
                ORDER BY sequence DESC LIMIT 1""",
                (conversation_id,),
            ).fetchone()
            if active is not None:
                raise ConversationBusyError(self._run_from_row(active))
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
        mode: AnswerMode | None = None,
        process_trace: AnswerProcessTrace | None = None,
        *,
        claims: tuple[dict[str, Any], ...] = (),
        answer_kind: AnswerKind | None = None,
        contract_version: AnswerContractVersion = "v1",
        response_kind: ResponseKind | None = None,
        response_payload: dict[str, Any] | None = None,
    ) -> Message:
        message = Message(
            id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            role=role,
            content=content,
            created_at=_now(),
            citations=citations,
            claims=claims,
            mode=mode,
            answer_kind=answer_kind,
            contract_version=contract_version,
            response_kind=response_kind,
            response_payload=response_payload or {},
            process_trace=process_trace,
        )
        with self._connection(write=True) as connection:
            self._insert_message(connection, message)
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (message.created_at, conversation_id),
            )
        return message

    def messages(
        self, conversation_id: str, *, limit: int = 500
    ) -> tuple[Message, ...]:
        _validate_limit(limit, maximum=1_000)
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT id, conversation_id, role, content, created_at, citations_json,
                       claims_json, mode, answer_kind, contract_version,
                       response_kind, response_payload_json, process_trace_json
                FROM (
                    SELECT sequence, id, conversation_id, role, content, created_at,
                           citations_json, claims_json, mode, answer_kind,
                           contract_version, response_kind, response_payload_json,
                           process_trace_json
                    FROM messages WHERE conversation_id = ?
                    ORDER BY sequence DESC LIMIT ?
                ) ORDER BY sequence ASC
                """,
                (conversation_id, limit),
            ).fetchall()
        return tuple(self._message_from_row(row) for row in rows)

    def message(self, message_id: str) -> Message | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT id, conversation_id, role, content, created_at, citations_json,
                       claims_json, mode, answer_kind, contract_version,
                       response_kind, response_payload_json, process_trace_json
                FROM messages WHERE id = ?
                """,
                (message_id,),
            ).fetchone()
        return None if row is None else self._message_from_row(row)

    def start_run(
        self,
        conversation_id: str,
        question: str,
        mode: AnswerMode,
        *,
        idempotency_key: str | None = None,
    ) -> tuple[AnswerRun, bool]:
        timestamp = _now()
        with self._connection(write=True, immediate=True) as connection:
            self._expire_runs(connection, timestamp)
            conversation = connection.execute(
                "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            if conversation is None:
                raise ConversationNotFoundError(
                    f"unknown conversation: {conversation_id}"
                )
            if idempotency_key is not None:
                existing = connection.execute(
                    """SELECT * FROM answer_runs
                    WHERE conversation_id = ? AND idempotency_key = ?""",
                    (conversation_id, idempotency_key),
                ).fetchone()
                if existing is not None:
                    run = self._run_from_row(existing)
                    if run.question != question or run.mode != mode:
                        raise IdempotencyConflictError(
                            "idempotency key was already used for a different request"
                        )
                    return run, False
            active = connection.execute(
                """SELECT * FROM answer_runs
                WHERE conversation_id = ? AND status IN ('queued', 'running')
                ORDER BY sequence DESC LIMIT 1""",
                (conversation_id,),
            ).fetchone()
            if active is not None:
                raise ConversationBusyError(self._run_from_row(active))
            user_message = Message(
                id=str(uuid.uuid4()),
                conversation_id=conversation_id,
                role="user",
                content=question,
                created_at=timestamp,
                mode=mode,
            )
            self._insert_message(connection, user_message)
            run = AnswerRun(
                id=str(uuid.uuid4()),
                conversation_id=conversation_id,
                idempotency_key=idempotency_key,
                question=question,
                mode=mode,
                status="running",
                user_message_id=user_message.id,
                owner_id=self.instance_id,
                lease_expires_at=_lease_deadline(self.lease_seconds),
                created_at=timestamp,
                started_at=timestamp,
            )
            connection.execute(
                """INSERT INTO answer_runs(
                    id, conversation_id, idempotency_key, question, mode, status,
                    user_message_id, assistant_message_id, error_code, error_message,
                    retryable, owner_id, lease_expires_at, created_at, started_at,
                    completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, 0, ?, ?, ?, ?, NULL)""",
                (
                    run.id,
                    run.conversation_id,
                    run.idempotency_key,
                    run.question,
                    run.mode,
                    run.status,
                    run.user_message_id,
                    run.owner_id,
                    run.lease_expires_at,
                    run.created_at,
                    run.started_at,
                ),
            )
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (timestamp, conversation_id),
            )
        return run, True

    def complete_run(
        self,
        run_id: str,
        *,
        answer: str,
        citations: tuple[dict[str, Any], ...],
        claims: tuple[dict[str, Any], ...],
        answer_kind: AnswerKind | None,
        process_trace: AnswerProcessTrace,
        contract_version: AnswerContractVersion = "v1",
        response_kind: ResponseKind | None = None,
        response_payload: dict[str, Any] | None = None,
    ) -> tuple[AnswerRun, Message]:
        timestamp = _now()
        with self._connection(write=True, immediate=True) as connection:
            self._expire_runs(connection, timestamp)
            row = connection.execute(
                "SELECT * FROM answer_runs WHERE id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown answer run: {run_id}")
            run = self._run_from_row(row)
            if run.status != "running":
                raise RuntimeError(f"cannot complete answer run in state {run.status}")
            assistant = Message(
                id=str(uuid.uuid4()),
                conversation_id=run.conversation_id,
                role="assistant",
                content=answer,
                created_at=timestamp,
                citations=citations,
                claims=claims,
                mode=run.mode,
                answer_kind=answer_kind,
                contract_version=contract_version,
                response_kind=response_kind,
                response_payload=response_payload or {},
                process_trace=process_trace,
            )
            self._insert_message(connection, assistant)
            cursor = connection.execute(
                """UPDATE answer_runs
                SET status = 'succeeded', assistant_message_id = ?, completed_at = ?
                WHERE id = ? AND status = 'running' AND owner_id = ?""",
                (assistant.id, timestamp, run_id, self.instance_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("answer run changed state while completing")
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (timestamp, run.conversation_id),
            )
            completed = connection.execute(
                "SELECT * FROM answer_runs WHERE id = ?", (run_id,)
            ).fetchone()
            assert completed is not None
        return self._run_from_row(completed), assistant

    def fail_run(
        self,
        run_id: str,
        *,
        code: str,
        message: str,
        retryable: bool,
        cancelled: bool = False,
    ) -> AnswerRun:
        timestamp = _now()
        status = "cancelled" if cancelled else "failed"
        with self._connection(write=True, immediate=True) as connection:
            self._expire_runs(connection, timestamp)
            connection.execute(
                """UPDATE answer_runs
                SET status = ?, error_code = ?, error_message = ?, retryable = ?,
                    completed_at = ?
                WHERE id = ? AND status IN ('queued', 'running') AND owner_id = ?""",
                (
                    status,
                    code,
                    message,
                    int(retryable),
                    timestamp,
                    run_id,
                    self.instance_id,
                ),
            )
            row = connection.execute(
                "SELECT * FROM answer_runs WHERE id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown answer run: {run_id}")
        return self._run_from_row(row)

    def get_run(self, run_id: str) -> AnswerRun | None:
        with self._connection(write=True, immediate=True) as connection:
            self._expire_runs(connection, _now())
            row = connection.execute(
                "SELECT * FROM answer_runs WHERE id = ?", (run_id,)
            ).fetchone()
        return None if row is None else self._run_from_row(row)

    def runs(self, conversation_id: str, *, limit: int = 100) -> tuple[AnswerRun, ...]:
        _validate_limit(limit)
        with self._connection(write=True, immediate=True) as connection:
            self._expire_runs(connection, _now())
            rows = connection.execute(
                """SELECT * FROM answer_runs WHERE conversation_id = ?
                ORDER BY sequence DESC LIMIT ?""",
                (conversation_id, limit),
            ).fetchall()
        return tuple(self._run_from_row(row) for row in rows)

    @staticmethod
    def _insert_message(connection: sqlite3.Connection, message: Message) -> None:
        connection.execute(
            """INSERT INTO messages(
                id, conversation_id, role, content, created_at, citations_json,
                mode, process_trace_json, claims_json, answer_kind,
                contract_version, response_kind, response_payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                message.id,
                message.conversation_id,
                message.role,
                message.content,
                message.created_at,
                json.dumps(
                    message.citations, ensure_ascii=False, separators=(",", ":")
                ),
                message.mode,
                json.dumps(
                    message.process_trace.model_dump(mode="json")
                    if message.process_trace
                    else None,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                json.dumps(message.claims, ensure_ascii=False, separators=(",", ":")),
                message.answer_kind,
                message.contract_version,
                message.response_kind,
                json.dumps(
                    message.response_payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            ),
        )

    @staticmethod
    def _message_from_row(row: sqlite3.Row) -> Message:
        citations = json.loads(str(row["citations_json"]))
        claims = json.loads(str(row["claims_json"]))
        raw_trace = json.loads(str(row["process_trace_json"]))
        response_payload = json.loads(str(row["response_payload_json"]))
        return Message(
            id=row["id"],
            conversation_id=row["conversation_id"],
            role=row["role"],
            content=row["content"],
            created_at=row["created_at"],
            citations=tuple(citations),
            claims=tuple(claims),
            mode=row["mode"],
            answer_kind=row["answer_kind"],
            contract_version=row["contract_version"],
            response_kind=row["response_kind"],
            response_payload=response_payload,
            process_trace=(
                AnswerProcessTrace.model_validate(raw_trace)
                if raw_trace is not None
                else None
            ),
        )

    @staticmethod
    def _run_from_row(row: sqlite3.Row) -> AnswerRun:
        return AnswerRun(
            id=row["id"],
            conversation_id=row["conversation_id"],
            idempotency_key=row["idempotency_key"],
            question=row["question"],
            mode=row["mode"],
            status=row["status"],
            user_message_id=row["user_message_id"],
            assistant_message_id=row["assistant_message_id"],
            error_code=row["error_code"],
            error_message=row["error_message"],
            retryable=bool(row["retryable"]),
            owner_id=row["owner_id"],
            lease_expires_at=row["lease_expires_at"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
        )

    @staticmethod
    def _expire_runs(connection: sqlite3.Connection, timestamp: str) -> None:
        connection.execute(
            """UPDATE answer_runs
            SET status = 'failed', error_code = 'ANSWER_RUN_LEASE_EXPIRED',
                error_message = 'Answer run lease expired before completion',
                retryable = 1, completed_at = ?
            WHERE status IN ('queued', 'running')
              AND lease_expires_at IS NOT NULL AND lease_expires_at <= ?""",
            (timestamp, timestamp),
        )


def _validate_limit(limit: int, *, maximum: int = 200) -> None:
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= maximum
    ):
        raise ValueError(f"limit must be between 1 and {maximum}")
