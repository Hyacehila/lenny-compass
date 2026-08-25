import sqlite3
from pathlib import Path

import pytest
from ai_pm_contracts import AnswerProcessPhase, AnswerProcessTrace
from lenny_backend.conversations import ConversationStore


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
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(messages)")
        }
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
