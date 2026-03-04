from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from admissions_conversation_engine.domain.agent_state import ContextSchema
from admissions_conversation_engine.domain.tenant_config import TenantConfig


def _runtime() -> SimpleNamespace:
    context = ContextSchema(
        chat_id="chat-1",
        channel_id="wa",
        case="off_hours",
        user_name="Pedro",
    )
    return SimpleNamespace(context=context)


def _tenant_config() -> TenantConfig:
    return TenantConfig(
        institution="Universidad Europea",
        terms_of_service="https://example.com/tos",
        allowed_topics="Admisiones",
        tone="empatico",
        language_fallback="en-US",
        allowed_languages="es-ES, en-US",
    )


def _make_node(labels: list[str], probs: list[float], config: TenantConfig | None = None):
    """Crea un FasttextLanguageDetectorNode con el modelo fasttext mockeado."""
    from admissions_conversation_engine.application.fasttext_language_detector_node import (
        FasttextLanguageDetectorNode,
    )

    mock_model = MagicMock()
    mock_model.predict.return_value = (labels, probs)

    with patch("fasttext.load_model", return_value=mock_model):
        node = FasttextLanguageDetectorNode(config or _tenant_config(), model_path="/fake/model.ftz")

    return node


def test_parse_allowed_languages_strips_values() -> None:
    # Verifica que la lista de idiomas permitidos se parsea correctamente.
    node = _make_node(["__label__es"], [0.99])
    assert node._parse_allowed_languages() == ["es-ES", "en-US"]


def test_match_language_maps_iso_to_bcp47() -> None:
    # Verifica que el código ISO de fasttext se mapea al BCP-47 permitido.
    node = _make_node(["__label__es"], [0.99])
    assert node._match_language("es", ["es-ES", "en-US"]) == "es-ES"
    assert node._match_language("en", ["es-ES", "en-US"]) == "en-US"


def test_match_language_returns_none_when_not_found() -> None:
    # Verifica que se retorna None si el código no tiene coincidencia.
    node = _make_node(["__label__fr"], [0.95])
    assert node._match_language("fr", ["es-ES", "en-US"]) is None


@pytest.mark.asyncio
async def test_fasttext_detects_spanish_correctly() -> None:
    # Verifica detección correcta de español.
    node = _make_node(["__label__es"], [0.98])
    state = {"messages": [HumanMessage(content="Hola, ¿cómo puedo inscribirme?")]}

    result = await node(state=state, runtime=_runtime())  # type: ignore[arg-type]

    assert result == {"language": "es-ES", "language_confidence": 0.98}


@pytest.mark.asyncio
async def test_fasttext_detects_english_correctly() -> None:
    # Verifica detección correcta de inglés.
    node = _make_node(["__label__en"], [0.95])
    state = {"messages": [HumanMessage(content="Hello, how can I apply?")]}

    result = await node(state=state, runtime=_runtime())  # type: ignore[arg-type]

    assert result == {"language": "en-US", "language_confidence": 0.95}


@pytest.mark.asyncio
async def test_fasttext_uses_fallback_for_unsupported_language() -> None:
    # Verifica que se usa el fallback cuando el idioma detectado no está en la lista.
    node = _make_node(["__label__fr"], [0.92])
    state = {"messages": [HumanMessage(content="Bonjour, comment puis-je m'inscrire?")]}

    result = await node(state=state, runtime=_runtime())  # type: ignore[arg-type]

    assert result == {"language": "en-US", "language_confidence": 0.92}


@pytest.mark.asyncio
async def test_fasttext_clamps_confidence_above_one() -> None:
    # Verifica que la confianza se limita al rango [0.0, 1.0].
    node = _make_node(["__label__es"], [1.5])
    state = {"messages": [HumanMessage(content="Hola")]}

    result = await node(state=state, runtime=_runtime())  # type: ignore[arg-type]

    assert result["language_confidence"] == 1.0


@pytest.mark.asyncio
async def test_fasttext_clamps_confidence_below_zero() -> None:
    # Verifica que la confianza nunca es negativa.
    node = _make_node(["__label__es"], [-0.1])
    state = {"messages": [HumanMessage(content="Hola")]}

    result = await node(state=state, runtime=_runtime())  # type: ignore[arg-type]

    assert result["language_confidence"] == 0.0


@pytest.mark.asyncio
async def test_fasttext_handles_empty_message() -> None:
    # Verifica que un mensaje vacío no rompe el nodo; usa el fallback.
    node = _make_node(["__label__fr"], [0.5])
    state = {"messages": [HumanMessage(content="")]}

    result = await node(state=state, runtime=_runtime())  # type: ignore[arg-type]

    assert result["language"] == "en-US"
