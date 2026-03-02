from admissions_conversation_engine.infrastructure.config.env_config_source import (
    EnvironmentVariableConfigSource,
)


def test_env_config_source_returns_process_environment(monkeypatch) -> None:
    # Verifica que la fuente de configuración por entorno expone las variables del proceso actual.
    monkeypatch.setenv("TEST_ENV_CONFIG_SOURCE", "ok")
    source = EnvironmentVariableConfigSource()

    result = source.load_configuration_values()

    assert result["TEST_ENV_CONFIG_SOURCE"] == "ok"
