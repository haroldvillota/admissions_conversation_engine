from __future__ import annotations

import os
from typing import Any, Mapping

from .config_source import ConfigSource


class VaultConfigSource(ConfigSource):
    def __init__(self, vault_path: str) -> None:
        self._vault_path = vault_path

    def load_configuration_values(self) -> Mapping[str, Any]:
        vault_url = os.getenv("AZURE_KEY_VAULT_URL", "").strip()
        if not vault_url:
            raise RuntimeError(
                "AZURE_KEY_VAULT_URL is required when USE_VAULT=1 "
                "(example: https://<vault-name>.vault.azure.net/)."
            )

        secret_client, resource_not_found_error = self._create_secret_client(vault_url)
        secret_name_by_config_key = self._secret_name_by_config_key()

        values: dict[str, Any] = {}

        for config_key, secret_name in secret_name_by_config_key.items():
            secret_value = self._read_secret_value(
                secret_client=secret_client,
                secret_name=secret_name,
                resource_not_found_error=resource_not_found_error,
            )
            
            if secret_value not in (None, ""):
                values[config_key] = secret_value

        return values

    def _create_secret_client(self, vault_url: str) -> tuple[Any, type[Exception]]:
        try:
            from azure.core.exceptions import ResourceNotFoundError
            from azure.identity import DefaultAzureCredential
            from azure.keyvault.secrets import SecretClient
        except ImportError as import_error:
            raise RuntimeError(
                "Azure Key Vault dependencies are not installed. "
                "Add 'azure-identity' and 'azure-keyvault-secrets' to project dependencies."
            ) from import_error

        credential = DefaultAzureCredential()
        client = SecretClient(vault_url=vault_url, credential=credential)
        return client, ResourceNotFoundError

    def _secret_name_by_config_key(self) -> dict[str, str]:
        prefix = self._vault_path.strip() or "admissions"
        return {
            "RAG__VECTOR_STORE__DSN": f"{prefix}-rag-vector-store-dsn",
            "RAG__EMBEDDINGS__API_KEY": f"{prefix}-rag-embeddings-api-key",
            "CHECKPOINTER__DSN": f"{prefix}-checkpointer-dsn",
            "LLM__DEFAULT__API_KEY": f"{prefix}-llm-default-api-key",
            "OBSERVABILITY__PUBLIC_KEY": f"{prefix}-observability-public-key",
            "OBSERVABILITY__SECRET_KEY": f"{prefix}-observability-secret-key",
        }

    @staticmethod
    def _read_secret_value(
        secret_client: Any,
        secret_name: str,
        resource_not_found_error: type[Exception],
    ) -> str | None:
        try:
            return secret_client.get_secret(secret_name).value
        except resource_not_found_error:
            return None
