from __future__ import annotations

import os
from typing import Any, Mapping

from .config_source import ConfigSource

_FROM_KEY_VAULT_PREFIX = "from_key_vault/"


class VaultConfigSource(ConfigSource):
    """
    Wraps a base config source and resolves any value that starts with
    'from_key_vault/<secret-name>' by fetching the actual secret from Azure Key Vault.

    Example:
        RAG__VECTOR_STORE__DSN=from_key_vault/admission-rag-vector-store-dsn
    """

    def __init__(self, base_source: ConfigSource) -> None:
        self._base_source = base_source

    def load_configuration_values(self) -> Mapping[str, Any]:
        base_values = dict(self._base_source.load_configuration_values())

        vault_refs = {
            key: value[len(_FROM_KEY_VAULT_PREFIX):]
            for key, value in base_values.items()
            if isinstance(value, str) and value.startswith(_FROM_KEY_VAULT_PREFIX)
        }

        if not vault_refs:
            return base_values

        vault_url = os.getenv("AZURE_KEY_VAULT_URL", "").strip()
        if not vault_url:
            raise RuntimeError(
                "AZURE_KEY_VAULT_URL is required when using 'from_key_vault/' references "
                "(example: https://<vault-name>.vault.azure.net/)."
            )

        secret_client, resource_not_found_error = self._create_secret_client(vault_url)

        resolved = dict(base_values)
        for key, secret_name in vault_refs.items():
            secret_value = self._read_secret_value(
                secret_client=secret_client,
                secret_name=secret_name,
                resource_not_found_error=resource_not_found_error,
            )
            if secret_value is None:
                raise RuntimeError(
                    f"Secret '{secret_name}' not found in Azure Key Vault "
                    f"(referenced by env var '{key}')."
                )
            resolved[key] = secret_value

        return resolved

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
