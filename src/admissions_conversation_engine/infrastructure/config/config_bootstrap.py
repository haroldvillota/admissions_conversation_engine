from __future__ import annotations

import os
from dotenv import load_dotenv

from dataclasses import dataclass
from functools import lru_cache

from typing import Any

from pydantic import ValidationError

from .app_config import AppConfig, LlmConfig
from .config_source import ConfigSource
from .env_config_source import EnvironmentVariableConfigSource
from .merged_config_source import MergedConfigSource
from .vault_config_source import VaultConfigSource


@dataclass(frozen=True)
class AppConfigBootstrapper:
    config_source: ConfigSource

    def load_app_config(self) -> AppConfig:
        raw_values = dict(self.config_source.load_configuration_values())
        raw_values = self._normalize_config_shape(raw_values)
        raw_values = self._apply_rag_embeddings_api_key_fallback(raw_values)
        try:
            app_config = AppConfig.model_validate(raw_values)
            return self._fill_llm_profiles_from_default_config(app_config)
        except ValidationError as validation_error:
            raise RuntimeError(f"Invalid application configuration:\n{validation_error}") from validation_error

    @staticmethod
    def _normalize_config_shape(raw_values: dict[str, Any]) -> dict[str, Any]:
        """
        Config sources can return:
        - nested dicts (tests, file-based sources)
        - flat env-var style dicts (EnvironmentVariableConfigSource, Vault)

        Normalize to the nested dict shape expected by AppConfig.
        """

        if any(top_key in raw_values for top_key in ("rag", "llm", "checkpointer", "observability", "tenant")):
            return raw_values

        if not any("__" in key for key in raw_values.keys()):
            return raw_values

        normalized: dict[str, Any] = {}
        for key, value in raw_values.items():
            if "__" not in key:
                continue
            parts = key.split("__")
            cursor: dict[str, Any] = normalized
            for part in parts[:-1]:
                part_key = part.lower()
                existing = cursor.get(part_key)
                if not isinstance(existing, dict):
                    cursor[part_key] = {}
                cursor = cursor[part_key]
            cursor[parts[-1].lower()] = value

        return normalized

    @staticmethod
    def _apply_rag_embeddings_api_key_fallback(raw_values: dict[str, Any]) -> dict[str, Any]:
        """
        README: RAG__EMBEDDINGS__API_KEY is optional.
        If it's missing/blank, fallback to LLM__DEFAULT__API_KEY.

        This is applied at bootstrap time so the rest of the system only sees a fully
        resolved AppConfig.
        """

        llm_default_api_key = (
            raw_values.get("llm", {}).get("default", {}).get("api_key")  # type: ignore[union-attr]
        )
        rag_embeddings = raw_values.get("rag", {}).get("embeddings", {})  # type: ignore[union-attr]

        if (
            isinstance(rag_embeddings, dict)
            and rag_embeddings.get("api_key") in (None, "")
            and llm_default_api_key not in (None, "")
        ):
            rag_embeddings["api_key"] = llm_default_api_key

        return raw_values

    @staticmethod
    def _fill_llm_profiles_from_default_config(app_config: AppConfig) -> AppConfig:
        default_profile = app_config.llm.default

        def with_defaults(profile_name: str) -> dict[str, Any]:
            profile = getattr(app_config.llm, profile_name)
            profile_values = profile.model_dump()
            default_values = default_profile.model_dump()

            for key, default_value in default_values.items():
                if profile_values.get(key) in (None, "") and default_value not in (None, ""):
                    profile_values[key] = default_value

            return profile_values

        llm_values = app_config.llm.model_dump()
        llm_values["guardrail"] = with_defaults("guardrail")
        llm_values["react"] = with_defaults("react")
        llm_values["translator"] = with_defaults("translator")

        llm_config = LlmConfig.model_validate(llm_values)
        return app_config.model_copy(update={"llm": llm_config})


@lru_cache(maxsize=1)
def get_app_config() -> AppConfig:
    load_dotenv()

    use_vault = os.getenv("USE_VAULT", "0") == "1"
    vault_path = os.getenv("VAULT_PATH", "secret/myapp")

    env_source = EnvironmentVariableConfigSource()

    if not use_vault:
        return AppConfigBootstrapper(config_source=env_source).load_app_config()

    secret_override_keys = [
        "RAG__VECTOR_STORE__DSN",
        "RAG__EMBEDDINGS__API_KEY",
        "CHECKPOINTER__DSN",
        "LLM__DEFAULT__API_KEY",
        "OBSERVABILITY__PUBLIC_KEY",
        "OBSERVABILITY__SECRET_KEY",
    ]

    merged_source = MergedConfigSource(
        base_source=env_source,
        override_source=VaultConfigSource(vault_path=vault_path),
        override_keys=secret_override_keys,
    )

    return AppConfigBootstrapper(config_source=merged_source).load_app_config()
