from __future__ import annotations

from typing import Any, Mapping

from .config_source import ConfigSource


class VaultConfigSource(ConfigSource):
    def __init__(self, vault_path: str) -> None:
        self._vault_path = vault_path

    def load_configuration_values(self) -> Mapping[str, Any]:
        """
        Devuelve SOLO secretos con las mismas keys env-style:
        - RAG__VECTOR_STORE__DSN
        - CHECKPOINTER__DSN
        - LLM__DEFAULT__API_KEY
        - OBSERVABILITY__PUBLIC_KEY
        - OBSERVABILITY__SECRET_KEY
        """
        raise NotImplementedError("Implement vault client here.")
