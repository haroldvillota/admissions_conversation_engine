from __future__ import annotations

import os
import re
from typing import Any, Mapping

from .config_source import ConfigSource


class VaultConfigSource(ConfigSource):
    _SECRET_KEYS: tuple[str, ...] = (
        "RAG__VECTOR_STORE__DSN",
        "RAG__EMBEDDINGS__API_KEY",
        "CHECKPOINTER__DSN",
        "LLM__DEFAULT__API_KEY",
        "OBSERVABILITY__PUBLIC_KEY",
        "OBSERVABILITY__SECRET_KEY",
    )

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
        vault_prefix = self._normalize_secret_name(self._vault_path)

        values: dict[str, Any] = {}

        for config_key in self._SECRET_KEYS:
            candidate_secret_names = self._candidate_secret_names(config_key, vault_prefix=vault_prefix)
            secret_value = self._read_first_available_secret(
                secret_client=secret_client,
                candidate_secret_names=candidate_secret_names,
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

    @classmethod
    def _candidate_secret_names(cls, config_key: str, vault_prefix: str) -> tuple[str, ...]:
        normalized_key = cls._normalize_secret_name(config_key)
        if not vault_prefix:
            return (normalized_key,)
        return (f"{vault_prefix}--{normalized_key}", normalized_key)

    @staticmethod
    def _read_first_available_secret(
        secret_client: Any,
        candidate_secret_names: tuple[str, ...],
        resource_not_found_error: type[Exception],
    ) -> str | None:
        for secret_name in candidate_secret_names:
            try:
                return secret_client.get_secret(secret_name).value
            except resource_not_found_error:
                continue

        return None

    @staticmethod
    def _normalize_secret_name(value: str) -> str:
        normalized = value.strip().replace("__", "--").replace("/", "--").replace("_", "-").lower()
        normalized = re.sub(r"[^a-z0-9-]", "-", normalized)
        normalized = re.sub(r"-{2,}", "--", normalized)
        return normalized.strip("-")
