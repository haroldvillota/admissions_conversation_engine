from admissions_conversation_engine.infrastructure.config.app_config import LlmProfileConfig
from admissions_conversation_engine.infrastructure.llm_factory import LLMFactory


def test_llm_factory_build_llm_uses_model_and_api_key(monkeypatch) -> None:
    captured = {}

    def fake_init_chat_model(model, api_key=None):
        captured["model"] = model
        captured["api_key"] = api_key
        return "fake-llm"

    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.llm_factory.init_chat_model",
        fake_init_chat_model,
    )

    factory = LLMFactory(
        LlmProfileConfig(
            provider="openai",
            api_key="secret",
            model="gpt-4.1-mini",
            temperature=0.0,
        )
    )

    llm = factory.build_llm()

    assert llm == "fake-llm"
    assert captured == {"model": "gpt-4.1-mini", "api_key": "secret"}
