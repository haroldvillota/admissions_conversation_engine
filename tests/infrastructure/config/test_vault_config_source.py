from __future__ import annotations

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


def test_vault_config_source_loads_explicit_secret_name_mapping(monkeypatch) -> None:
    monkeypatch.setenv("AZURE_KEY_VAULT_URL", "https://example.vault.azure.net/")

    fake_client = FakeSecretClient(
        {
            "admissions-rag-vector-store-dsn": "postgresql://from-vault",
            "admissions-llm-default-api-key": "openai-key-from-vault",
            "admissions-observability-public-key": "pk-test",
        }
    )

    source = VaultConfigSource(vault_path="admissions")
    monkeypatch.setattr(
        source,
        "_create_secret_client",
        lambda _: (fake_client, FakeNotFoundError),
    )

    result = source.load_configuration_values()

    assert result == {
        "RAG__VECTOR_STORE__DSN": "postgresql://from-vault",
        "LLM__DEFAULT__API_KEY": "openai-key-from-vault",
        "OBSERVABILITY__PUBLIC_KEY": "pk-test",
    }


def test_vault_config_source_requires_vault_url(monkeypatch) -> None:
    monkeypatch.delenv("AZURE_KEY_VAULT_URL", raising=False)
    source = VaultConfigSource(vault_path="secret/myapp")

    try:
        source.load_configuration_values()
        assert False, "Expected RuntimeError when AZURE_KEY_VAULT_URL is missing."
    except RuntimeError as error:
        assert "AZURE_KEY_VAULT_URL is required" in str(error)
