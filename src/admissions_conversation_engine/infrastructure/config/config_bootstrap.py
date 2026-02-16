from __future__ import annotations

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
        raw_values = self.config_source.load_configuration_values()
        try:
            app_config = AppConfig.model_validate(raw_values)
            return self._fill_llm_profiles_from_default_config(app_config)
        except ValidationError as validation_error:
            raise RuntimeError(f"Invalid application configuration:\n{validation_error}") from validation_error

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
def get_app_config(*, use_vault: bool, vault_path: str) -> AppConfig:
    env_source = EnvironmentVariableConfigSource()

    if not use_vault:
        return AppConfigBootstrapper(config_source=env_source).load_app_config()

    secret_override_keys = [
        "RAG__VECTOR_STORE__DSN",
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
