from __future__ import annotations

import pytest

from admissions_conversation_engine.infrastructure.config.config_source import ConfigSource
from admissions_conversation_engine.infrastructure.config.vault_config_source import (
    VaultConfigSource,
)


class FakeNotFoundError(Exception):
    pass


class FakeSecret:
    def __init__(self, value: str) -> None:
        self.value = value


class FakeSecretClient:
    def __init__(self, values_by_name: dict[str, str]) -> None:
        self._values_by_name = values_by_name

    def get_secret(self, name: str) -> FakeSecret:
        if name not in self._values_by_name:
            raise FakeNotFoundError()
        return FakeSecret(self._values_by_name[name])


class StaticConfigSource(ConfigSource):
    def __init__(self, values: dict) -> None:
        self._values = values

    def load_configuration_values(self) -> dict:
        return self._values


def _make_source_with_fake_client(base_values: dict, vault_values: dict, monkeypatch) -> VaultConfigSource:
    monkeypatch.setenv("AZURE_KEY_VAULT_URL", "https://example.vault.azure.net/")
    source = VaultConfigSource(base_source=StaticConfigSource(base_values))
    monkeypatch.setattr(
        source,
        "_create_secret_client",
        lambda _: (FakeSecretClient(vault_values), FakeNotFoundError),
    )
    return source


def test_vault_config_source_resolves_from_key_vault_references(monkeypatch) -> None:
    base_values = {
        "RAG__VECTOR_STORE__DSN": "from_key_vault/admission-rag-vector-store-dsn",
        "LLM__DEFAULT__API_KEY": "from_key_vault/admissions-llm-default-api-key",
        "OBSERVABILITY__PUBLIC_KEY": "from_key_vault/admissions-observability-public-key",
        "CHECKPOINTER__DSN": "postgresql://local-db",
    }
    vault_values = {
        "admission-rag-vector-store-dsn": "postgresql://from-vault",
        "admissions-llm-default-api-key": "openai-key-from-vault",
        "admissions-observability-public-key": "pk-test",
    }

    source = _make_source_with_fake_client(base_values, vault_values, monkeypatch)
    result = source.load_configuration_values()

    assert result["RAG__VECTOR_STORE__DSN"] == "postgresql://from-vault"
    assert result["LLM__DEFAULT__API_KEY"] == "openai-key-from-vault"
    assert result["OBSERVABILITY__PUBLIC_KEY"] == "pk-test"
    assert result["CHECKPOINTER__DSN"] == "postgresql://local-db"


def test_vault_config_source_skips_vault_when_no_references() -> None:
    base_values = {
        "RAG__VECTOR_STORE__DSN": "postgresql://local-db",
        "LLM__DEFAULT__API_KEY": "local-key",
    }

    source = VaultConfigSource(base_source=StaticConfigSource(base_values))
    # no AZURE_KEY_VAULT_URL set, no vault call expected
    result = source.load_configuration_values()

    assert result == base_values


def test_vault_config_source_requires_vault_url_when_references_exist(monkeypatch) -> None:
    monkeypatch.delenv("AZURE_KEY_VAULT_URL", raising=False)

    base_values = {"RAG__VECTOR_STORE__DSN": "from_key_vault/my-secret"}
    source = VaultConfigSource(base_source=StaticConfigSource(base_values))

    with pytest.raises(RuntimeError, match="AZURE_KEY_VAULT_URL is required"):
        source.load_configuration_values()


def test_vault_config_source_raises_when_secret_not_found(monkeypatch) -> None:
    base_values = {"RAG__VECTOR_STORE__DSN": "from_key_vault/missing-secret"}
    source = _make_source_with_fake_client(base_values, vault_values={}, monkeypatch=monkeypatch)

    with pytest.raises(RuntimeError, match="missing-secret"):
        source.load_configuration_values()
