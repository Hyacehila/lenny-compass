from pathlib import Path

import pytest
from lenny_backend.settings import Settings
from pydantic import ValidationError


def test_settings_defaults_and_environment(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("APP_PORT", "4999")
    monkeypatch.setenv("AGENTIC_RAG_ENABLED", "true")
    settings = Settings(_env_file=None, repository_root=tmp_path)

    assert settings.app_host == "127.0.0.1"
    assert settings.app_port == 4999
    assert settings.agentic_rag_enabled is True
    assert settings.llm_wiki_enabled is False
    assert settings.resolved_chat_db_path == tmp_path / "runtime/backend/conversations.sqlite3"
    assert settings.resolved_retrieval_manifest_path == tmp_path / "retrieval/manifests/build.json"


def test_absolute_paths_are_preserved(tmp_path: Path) -> None:
    absolute = tmp_path / "chat.sqlite3"
    settings = Settings(
        _env_file=None,
        repository_root=tmp_path / "repository",
        chat_db_path=absolute,
    )
    assert settings.resolved_chat_db_path == absolute


@pytest.mark.parametrize(
    "host", ["0.0.0.0", "192.168.1.10", "example.com", "localhost", "::1"]
)
def test_app_host_rejects_non_loopback(host: str) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, app_host=host)

def test_app_host_accepts_only_required_ipv4_loopback() -> None:
    assert Settings(_env_file=None, app_host="127.0.0.1").app_host == "127.0.0.1"
