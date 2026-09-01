import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import pytest
from ai_pm_contracts import AnswerProcessPhase, AnswerProcessTrace
from lenny_backend.conversations import (
    ConversationBusyError,
    ConversationNotFoundError,
    ConversationStore,
)


def test_conversation_and_message_crud(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "conversations.sqlite3")
    conversation = store.create("Research")

    assert store.get(conversation.id) == conversation
    assert store.list() == (conversation,)

    renamed = store.rename(conversation.id, "  Retention   research  ")
    assert renamed is not None
    assert renamed.title == "Retention research"
    assert store.rename("missing", "Still valid") is None
    with pytest.raises(ValueError):
        store.rename(conversation.id, "   ")

    user = store.add_message(conversation.id, "user", "Question", mode="direct")
    assistant = store.add_message(
        conversation.id,
        "assistant",
        "Answer",
        ({"evidence_id": "source:one"},),
        mode="direct",
        process_trace=AnswerProcessTrace(
            phases=(AnswerProcessPhase(phase="retrieve", message="Searching"),),
            provider_counts={"keyword": 3, "vector": 2},
            fused_count=4,
            evidence_count=1,
        ),
    )
    assert store.messages(conversation.id) == (user, assistant)
    assert store.get(conversation.id).updated_at == assistant.created_at  # type: ignore[union-attr]

    assert store.delete(conversation.id) is True
    assert store.get(conversation.id) is None
    assert store.messages(conversation.id) == ()
    assert store.delete(conversation.id) is False

    with sqlite3.connect(tmp_path / "conversations.sqlite3") as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(messages)")}
    assert "reasoning_content" not in columns
    assert "mode" in columns
    assert "process_trace_json" in columns
    assert assistant.process_trace is not None
    assert assistant.process_trace.provider_counts == {"keyword": 3, "vector": 2}


def test_message_requires_existing_conversation(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "conversations.sqlite3")
    with pytest.raises(sqlite3.IntegrityError):
        store.add_message("missing", "user", "Question")


def test_existing_message_database_is_migrated_to_store_mode(tmp_path: Path) -> None:
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            ) STRICT;
            CREATE TABLE messages (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                citations_json TEXT NOT NULL DEFAULT '[]'
            ) STRICT;
            """
        )

    store = ConversationStore(path)
    conversation = store.create()
    stored = store.add_message(conversation.id, "user", "Question", mode="agentic")

    assert conversation.title == "新对话"
    assert stored.mode == "agentic"
    assert stored.process_trace is None
    assert store.messages(conversation.id)[0].mode == "agentic"
    assert store.messages(conversation.id)[0].process_trace is None
    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(messages)")}
    assert "process_trace_json" in columns


def test_legacy_migration_is_serialized_across_store_initializers(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy-concurrent.sqlite3"
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.executescript(
            """
            CREATE TABLE conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            ) STRICT;
            CREATE TABLE messages (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                citations_json TEXT NOT NULL DEFAULT '[]'
            ) STRICT;
            """
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        stores = tuple(executor.map(lambda _: ConversationStore(path), range(2)))

    assert all(store.list() == () for store in stores)
    with closing(sqlite3.connect(path)) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(messages)")}
    assert {"mode", "process_trace_json", "claims_json", "answer_kind"} <= columns


def test_answer_runs_enforce_conversation_lock_and_idempotency(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "conversations.sqlite3")
    conversation = store.create("Runs")

    run, created = store.start_run(
        conversation.id,
        "Question",
        "direct",
        idempotency_key="request-one",
    )
    replay, replay_created = store.start_run(
        conversation.id,
        "Question",
        "direct",
        idempotency_key="request-one",
    )

    assert created is True
    assert replay_created is False
    assert replay == run
    assert len(store.messages(conversation.id)) == 1
    with pytest.raises(ConversationBusyError) as caught:
        store.start_run(conversation.id, "Another question", "agentic")
    assert caught.value.active_run.id == run.id
    with pytest.raises(ConversationBusyError):
        store.delete(conversation.id)

    failed = store.fail_run(
        run.id,
        code="TEST_FAILURE",
        message="failed",
        retryable=True,
    )
    assert failed.status == "failed"
    next_run, next_created = store.start_run(
        conversation.id, "Another question", "agentic"
    )
    assert next_created is True
    assert next_run.status == "running"


def test_start_run_checks_conversation_inside_transaction(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "conversations.sqlite3")
    with pytest.raises(ConversationNotFoundError):
        store.start_run("missing", "Question", "direct")


def test_answer_run_completion_is_atomic_and_restart_recovers_orphans(
    tmp_path: Path,
) -> None:
    path = tmp_path / "conversations.sqlite3"
    store = ConversationStore(path)
    conversation = store.create("Runs")
    run, _ = store.start_run(conversation.id, "Question", "direct")
    trace = AnswerProcessTrace(evidence_count=1)

    completed, assistant = store.complete_run(
        run.id,
        answer="Answer",
        citations=({"evidence_id": "source:one"},),
        claims=(
            {
                "claim_id": "claim-1",
                "text": "Answer",
                "evidence_ids": ["source:one"],
            },
        ),
        answer_kind="grounded",
        process_trace=trace,
    )
    assert completed.status == "succeeded"
    assert completed.assistant_message_id == assistant.id
    assert store.messages(conversation.id)[-1].claims[0]["claim_id"] == "claim-1"

    orphan, _ = store.start_run(conversation.id, "Next", "agentic")
    restarted = ConversationStore(path)
    recovered = restarted.get_run(orphan.id)
    assert recovered is not None
    assert recovered.status == "running"

    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            "UPDATE answer_runs SET lease_expires_at = ? WHERE id = ?",
            ("2000-01-01T00:00:00Z", orphan.id),
        )
    after_expiry = ConversationStore(path)
    recovered = after_expiry.get_run(orphan.id)
    assert recovered is not None
    assert recovered.status == "failed"
    assert recovered.error_code == "ANSWER_RUN_LEASE_EXPIRED"
    assert recovered.retryable is True

    moved = tmp_path / "moved.sqlite3"
    path.replace(moved)
    assert moved.is_file()
