from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from admissions_conversation_engine.application.llm_language_detector_node import (
    LlmLanguageDetectorNode,
)
from admissions_conversation_engine.domain.agent_state import ContextSchema
from admissions_conversation_engine.infrastructure.config.app_config import TenantConfig


def _runtime() -> SimpleNamespace:
    context = ContextSchema(
        chat_id="chat-1",
        channel_id="wa",
        case="off_hours",
        user_name="Pedro",
    )
    return SimpleNamespace(context=context)


def _llm_with_content(content: str) -> RunnableLambda:
    return RunnableLambda(lambda _: AIMessage(content=content))


def _tenant_config() -> TenantConfig:
    return TenantConfig(
        institution="Universidad Europea",
        terms_of_service="https://example.com/tos",
        allowed_topics="Admisiones",
        tone="empatico",
        language_fallback="en-US",
        allowed_languages="es-ES, en-US",
    )


def test_parse_allowed_languages_strips_values() -> None:
    node = LlmLanguageDetectorNode(
        llm=_llm_with_content('{"language":"es-ES","confidence":0.9}'),
        prompt="detector",
        config=_tenant_config(),
    )

    result = node._parse_allowed_languages()

    assert result == ["es-ES", "en-US"]


def test_language_detector_returns_detected_language_and_clamped_confidence() -> None:
    node = LlmLanguageDetectorNode(
        llm=_llm_with_content('{"language":"es-ES","confidence":1.7}'),
        prompt="detector",
        config=_tenant_config(),
    )
    state = {"messages": [HumanMessage(content="Hola")]}

    result = node(state=state, runtime=_runtime())  # type: ignore[arg-type]

    assert result == {"language": "es-ES", "language_confidence": 1.0}


def test_language_detector_uses_fallback_for_invalid_json() -> None:
    node = LlmLanguageDetectorNode(
        llm=_llm_with_content("not-json"),
        prompt="detector",
        config=_tenant_config(),
    )
    state = {"messages": [HumanMessage(content="Hola")]}

    result = node(state=state, runtime=_runtime())  # type: ignore[arg-type]

    assert result == {"language": "en-US", "language_confidence": 0.0}


def test_language_detector_uses_fallback_for_language_outside_allowed_list() -> None:
    node = LlmLanguageDetectorNode(
        llm=_llm_with_content('{"language":"fr-FR","confidence":0.8}'),
        prompt="detector",
        config=_tenant_config(),
    )
    state = {"messages": [HumanMessage(content="Bonjour")]}

    result = node(state=state, runtime=_runtime())  # type: ignore[arg-type]

    assert result == {"language": "en-US", "language_confidence": 0.8}


def test_language_detector_defaults_confidence_to_zero_when_invalid() -> None:
    node = LlmLanguageDetectorNode(
        llm=_llm_with_content('{"language":"es-ES","confidence":"not-a-number"}'),
        prompt="detector",
        config=_tenant_config(),
    )
    state = {"messages": [HumanMessage(content="Hola")]}

    result = node(state=state, runtime=_runtime())  # type: ignore[arg-type]

    assert result == {"language": "es-ES", "language_confidence": 0.0}
