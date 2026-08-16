from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Runtime configuration loaded from the repository-local ignored ``.env``."""

    model_config = SettingsConfigDict(
        env_file=REPOSITORY_ROOT / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        case_sensitive=False,
    )

    repository_root: Path = Field(default=REPOSITORY_ROOT, exclude=True)
    app_host: str = "127.0.0.1"
    app_port: int = Field(default=4317, ge=1, le=65535)

    facts_root: Path = Path("facts")
    retrieval_manifest_path: Path = Path("retrieval/manifests/build.json")
    lexical_db_path: Path = Path("retrieval/lexical/fts.sqlite3")
    semantic_db_path: Path = Path("retrieval/semantic/vec.sqlite3")
    chat_db_path: Path = Path("runtime/backend/conversations.sqlite3")
    frontend_dist_path: Path = Path("frontend/dist")
    query_rewrite_prompt_path: Path = Path("config/agent/query_rewrite.txt")
    answer_prompt_path: Path = Path("config/agent/direct_answer.txt")
    agentic_prompt_path: Path = Path("config/agent/agentic_answer.txt")
    vision_prompt_path: Path = Path("config/agent/vision_observation.txt")

    agentic_rag_enabled: bool = False
    llm_wiki_enabled: bool = False
    llm_wiki_api_base_url: str = "http://127.0.0.1:19828/api/v1"
    llm_wiki_api_token: str | None = None
    llm_wiki_project_id: str | None = None
    llm_wiki_allow_remote: bool = False
    deepseek_api_key: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_answer_model: str = "deepseek-v4-flash"
    deepseek_rewrite_model: str = "deepseek-v4-flash"
    zhipu_api_key: str | None = None
    zhipu_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    zhipu_embedding_model: str = "embedding-3"
    zhipu_embedding_dimensions: int = Field(default=2048, ge=2048, le=2048)
    zhipu_vision_model: str = "glm-5v-turbo"
    provider_timeout_seconds: float = Field(default=60.0, gt=0, le=300)

    @field_validator("app_host")
    @classmethod
    def validate_loopback_host(cls, value: str) -> str:
        value = value.strip().casefold()
        if value != "127.0.0.1":
            raise ValueError("APP_HOST must be 127.0.0.1")
        return value

    def resolve_path(self, value: Path) -> Path:
        return value if value.is_absolute() else self.repository_root / value

    @property
    def resolved_facts_root(self) -> Path:
        return self.resolve_path(self.facts_root)

    @property
    def resolved_retrieval_manifest_path(self) -> Path:
        return self.resolve_path(self.retrieval_manifest_path)

    @property
    def resolved_lexical_db_path(self) -> Path:
        return self.resolve_path(self.lexical_db_path)

    @property
    def resolved_semantic_db_path(self) -> Path:
        return self.resolve_path(self.semantic_db_path)

    @property
    def resolved_chat_db_path(self) -> Path:
        return self.resolve_path(self.chat_db_path)

    @property
    def resolved_frontend_dist_path(self) -> Path:
        return self.resolve_path(self.frontend_dist_path)

    @property
    def resolved_query_rewrite_prompt_path(self) -> Path:
        return self.resolve_path(self.query_rewrite_prompt_path)

    @property
    def resolved_answer_prompt_path(self) -> Path:
        return self.resolve_path(self.answer_prompt_path)

    @property
    def resolved_vision_prompt_path(self) -> Path:
        return self.resolve_path(self.vision_prompt_path)

    @property
    def resolved_agentic_prompt_path(self) -> Path:
        return self.resolve_path(self.agentic_prompt_path)
