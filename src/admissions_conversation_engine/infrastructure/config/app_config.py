from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict


class RagVectorStoreConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: str
    dsn: str
    collection: str
    top_k: int = 4


class RagEmbeddingsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: Literal["openai"] = "openai"
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


class TenantConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    institution: str
    terms_of_service: str
    allowed_topics: str
    tone: str
    language_fallback: str
    allowed_languages: str


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
