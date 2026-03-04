from admissions_conversation_engine.infrastructure.config.app_config import LlmProfileConfig
from admissions_conversation_engine.infrastructure.llm_factory import LLMFactory


def _llm_config(api_key: str | None = "secret") -> LlmProfileConfig:
    return LlmProfileConfig(
        provider="openai",
        api_key=api_key,
        model="gpt-4.1-mini",
        temperature=0.0,
    )


def test_llm_factory_build_llm_uses_model_and_api_key(monkeypatch) -> None:
    # Verifica que LLMFactory construye el modelo de chat pasando el nombre de modelo y la API key del perfil.
    captured = {}

    def fake_init_chat_model(model, api_key=None):
        captured["model"] = model
        captured["api_key"] = api_key
        return "fake-llm"

    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.llm_factory.init_chat_model",
        fake_init_chat_model,
    )

    factory = LLMFactory(_llm_config())

    llm = factory.build_llm()

    assert llm == "fake-llm"
    assert captured == {"model": "gpt-4.1-mini", "api_key": "secret"}


def test_probe_connection_succeeds_when_api_is_reachable(monkeypatch) -> None:
    # Verifica que probe_connection no lanza excepción cuando la API del LLM responde correctamente.
    import admissions_conversation_engine.infrastructure.llm_factory as mod

    class FakeModels:
        def list(self, limit: int) -> None:
            pass

    class FakeOpenAI:
        def __init__(self, api_key: str) -> None:
            self.models = FakeModels()

    monkeypatch.setattr(mod.openai, "OpenAI", FakeOpenAI)

    factory = LLMFactory(_llm_config())
    factory.probe_connection()  # No debe lanzar


def test_probe_connection_raises_runtime_error_when_api_key_is_missing() -> None:
    # Verifica que probe_connection lanza RuntimeError inmediatamente cuando no hay API key configurada.
    factory = LLMFactory(_llm_config(api_key=None))

    try:
        factory.probe_connection()
        assert False, "Expected RuntimeError"
    except RuntimeError as error:
        assert "API key no configurada" in str(error)
        assert "gpt-4.1-mini" in str(error)


def test_probe_connection_raises_runtime_error_when_api_is_unreachable(monkeypatch) -> None:
    # Verifica que probe_connection lanza RuntimeError con mensaje claro cuando la API del LLM falla.
    import admissions_conversation_engine.infrastructure.llm_factory as mod

    class FakeModels:
        def list(self, limit: int) -> None:
            raise Exception("Connection refused")

    class FakeOpenAI:
        def __init__(self, api_key: str) -> None:
            self.models = FakeModels()

    monkeypatch.setattr(mod.openai, "OpenAI", FakeOpenAI)

    factory = LLMFactory(_llm_config())

    try:
        factory.probe_connection()
        assert False, "Expected RuntimeError"
    except RuntimeError as error:
        assert "API del modelo LLM" in str(error)
        assert "gpt-4.1-mini" in str(error)
        assert "Connection refused" in str(error)
