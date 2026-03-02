from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from admissions_conversation_engine.application.simple_llm_node import SimpleLLMNode
from admissions_conversation_engine.domain.agent_state import ContextSchema


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


@pytest.mark.asyncio
async def test_simple_llm_node_returns_ai_message_in_messages_key() -> None:
    # Verifica que el nodo LLM simple devuelve la respuesta del modelo en la clave "messages" del estado.
    node = SimpleLLMNode(
        llm=_llm_with_content("Respuesta simple del asistente"),
        prompt="Eres un asistente de admisiones.",
    )
    state = {"messages": [HumanMessage(content="Hola")]}

    result = await node(state=state, runtime=_runtime())  # type: ignore[arg-type]

    assert len(result["messages"]) == 1
    assert result["messages"][0].content == "Respuesta simple del asistente"


@pytest.mark.asyncio
async def test_simple_llm_node_injects_context_fields_into_prompt() -> None:
    # Verifica que el contexto de ejecución (user_name, chat_id, etc.) se inyecta en el system prompt renderizado.
    captured = {}

    def fake_llm(inputs):
        captured["messages"] = inputs.to_messages()
        return AIMessage(content="ok")

    node = SimpleLLMNode(
        llm=RunnableLambda(fake_llm),
        prompt="Hola {user_name}, tu chat es {chat_id}.",
    )
    state = {"messages": [HumanMessage(content="Hola")]}

    await node(state=state, runtime=_runtime())  # type: ignore[arg-type]

    system_messages = [m for m in captured["messages"] if m.type == "system"]
    assert any("Pedro" in m.content for m in system_messages)
    assert any("chat-1" in m.content for m in system_messages)


@pytest.mark.asyncio
async def test_simple_llm_node_passes_conversation_history_to_llm() -> None:
    # Verifica que el historial de mensajes del estado se incluye en los inputs enviados al LLM.
    captured = {}

    def fake_llm(inputs):
        captured["messages"] = inputs.to_messages()
        return AIMessage(content="respuesta")

    node = SimpleLLMNode(
        llm=RunnableLambda(fake_llm),
        prompt="system prompt",
    )
    state = {"messages": [HumanMessage(content="¿Cuáles son los requisitos?")]}

    await node(state=state, runtime=_runtime())  # type: ignore[arg-type]

    human_messages = [m for m in captured["messages"] if m.type == "human"]
    assert any("requisitos" in m.content for m in human_messages)
