from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from admissions_conversation_engine.domain.tenant_config import TenantConfig


class LanguageDetectorConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: Literal["fasttext", "llm"] = "fasttext"
    fasttext_model_path: str = "/app/models/lid.176.ftz"


class RagVectorStoreConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: str
    dsn: str
    collection: str
    top_k: int = 4


class RagEmbeddingsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: Literal["openai"] = "openai"
    api_key: str = Field(min_length=1)
    model: str
    batch_size: int = 128


class RagConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    vector_store: RagVectorStoreConfig
    embeddings: RagEmbeddingsConfig


class LlmProfileConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: Literal["openai"] = "openai"
    api_key: str | None = None
    model: str
    temperature: float = 0.0


class LlmConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    default: LlmProfileConfig
    guardrail: LlmProfileConfig
    react: LlmProfileConfig
    translator: LlmProfileConfig


class CheckpointerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: str
    dsn: str


class ObservabilityConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: Literal["langfuse"] = "langfuse"
    public_key: str | None = None
    secret_key: str | None = None
    base_url: str | None = None


class AuthConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        frozen=True,
        extra="ignore",
        env_nested_delimiter="__",
    )

    rag: RagConfig
    llm: LlmConfig
    checkpointer: CheckpointerConfig
    observability: ObservabilityConfig
    tenant: TenantConfig
    auth: AuthConfig
    language_detector: LanguageDetectorConfig = Field(default_factory=LanguageDetectorConfig)

    def __str__(self) -> str:
        return self.model_dump_json(indent=2, ensure_ascii=False)
