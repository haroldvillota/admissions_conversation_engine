from admissions_conversation_engine.infrastructure.config.app_config import (
    AppConfig,
    LlmConfig,
    LlmProfileConfig,
)
from admissions_conversation_engine.infrastructure.config.config_bootstrap import (
    AppConfigBootstrapper,
)
from admissions_conversation_engine.infrastructure.config.config_source import ConfigSource
from admissions_conversation_engine.infrastructure.config.env_config_source import (
    EnvironmentVariableConfigSource,
)


class StaticConfigSource(ConfigSource):
    def __init__(self, values: dict):
        self._values = values

    def load_configuration_values(self) -> dict:
        return self._values


def _valid_nested_raw_values() -> dict:
    return {
        "rag": {
            "vector_store": {
                "kind": "postgresql",
                "dsn": "postgresql://u:p@h:5432/db",
                "collection": "rag_docs",
                "top_k": 4,
            },
            "embeddings": {
                "provider": "openai",
                "api_key": "embeddings-key",
                "model": "text-embedding-3-large",
                "batch_size": 128,
            },
        },
        "llm": {
            "default": {
                "provider": "openai",
                "api_key": "default-key",
                "model": "gpt-4.1-mini",
                "temperature": 0.0,
            },
            "guardrail": {
                "provider": "openai",
                "api_key": None,
                "model": "",
                "temperature": 0.2,
            },
            "react": {
                "provider": "openai",
                "api_key": "",
                "model": "gpt-4.1",
                "temperature": 0.0,
            },
            "translator": {
                "provider": "openai",
                "api_key": None,
                "model": "",
                "temperature": 0.0,
            },
        },
        "checkpointer": {
            "kind": "postgresql",
            "dsn": "postgresql://u:p@h:5432/db",
        },
        "observability": {
            "provider": "langfuse",
            "public_key": "pk",
            "secret_key": "sk",
            "base_url": "https://cloud.langfuse.com",
        },
        "tenant": {
            "institution": "Universidad Europea",
            "terms_of_service": "https://example.com/tos",
            "allowed_topics": "Admisiones",
            "tone": "Empatico",
            "language_fallback": "en-US",
            "allowed_languages": "es-ES,en-US",
        },
    }

def _flatten_nested_to_env(values: dict, *, prefix: str = "") -> dict:
    out: dict[str, str] = {}
    for key, value in values.items():
        composed = f"{prefix}__{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten_nested_to_env(value, prefix=composed))
        else:
            # env vars come as strings; pydantic will coerce types as needed
            out[composed.upper()] = "" if value is None else str(value)
    return out


def test_load_app_config_fills_missing_llm_values_from_default() -> None:
    bootstrapper = AppConfigBootstrapper(config_source=StaticConfigSource(_valid_nested_raw_values()))

    app_config = bootstrapper.load_app_config()

    assert app_config.llm.guardrail.api_key == "default-key"
    assert app_config.llm.guardrail.model == "gpt-4.1-mini"
    assert app_config.llm.react.api_key == "default-key"
    assert app_config.llm.translator.model == "gpt-4.1-mini"

def test_load_app_config_uses_llm_default_api_key_as_rag_embeddings_fallback_nested() -> None:
    values = _valid_nested_raw_values()
    del values["rag"]["embeddings"]["api_key"]
    bootstrapper = AppConfigBootstrapper(config_source=StaticConfigSource(values))

    app_config = bootstrapper.load_app_config()

    assert app_config.rag.embeddings.api_key == "default-key"


def test_load_app_config_uses_llm_default_api_key_as_rag_embeddings_fallback_env_form(monkeypatch) -> None:
    env_values = _flatten_nested_to_env(_valid_nested_raw_values())
    env_values.pop("RAG__EMBEDDINGS__API_KEY", None)

    for key, value in env_values.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("RAG__EMBEDDINGS__API_KEY", raising=False)

    bootstrapper = AppConfigBootstrapper(config_source=EnvironmentVariableConfigSource())
    app_config = bootstrapper.load_app_config()

    assert app_config.rag.embeddings.api_key == "default-key"


def test_load_app_config_raises_runtime_error_for_invalid_config() -> None:
    invalid_values = _valid_nested_raw_values()
    invalid_values["llm"]["default"]["provider"] = "invalid-provider"
    bootstrapper = AppConfigBootstrapper(config_source=StaticConfigSource(invalid_values))

    try:
        bootstrapper.load_app_config()
        assert False, "Expected RuntimeError for invalid configuration"
    except RuntimeError as error:
        assert "Invalid application configuration:" in str(error)


def test_fill_llm_profiles_from_default_config_keeps_profile_specific_values() -> None:
    app_config = AppConfig(
        rag=_valid_nested_raw_values()["rag"],
        llm=LlmConfig(
            default=LlmProfileConfig(
                provider="openai",
                api_key="default-key",
                model="gpt-4.1-mini",
                temperature=0.0,
            ),
            guardrail=LlmProfileConfig(
                provider="openai",
                api_key=None,
                model="guardrail-model",
                temperature=0.0,
            ),
            react=LlmProfileConfig(
                provider="openai",
                api_key=None,
                model="react-model",
                temperature=0.0,
            ),
            translator=LlmProfileConfig(
                provider="openai",
                api_key=None,
                model="translator-model",
                temperature=0.0,
            ),
        ),
        checkpointer=_valid_nested_raw_values()["checkpointer"],
        observability=_valid_nested_raw_values()["observability"],
        tenant=_valid_nested_raw_values()["tenant"],
    )

    updated = AppConfigBootstrapper._fill_llm_profiles_from_default_config(app_config)

    assert updated.llm.guardrail.model == "guardrail-model"
    assert updated.llm.react.model == "react-model"
    assert updated.llm.translator.model == "translator-model"
    assert updated.llm.guardrail.api_key == "default-key"
