from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from admissions_conversation_engine.application.react_node import ReactNode
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


def test_react_node_returns_ai_message_in_messages_key() -> None:
    node = ReactNode(
        llm=_llm_with_content("Respuesta del asistente"),
        prompt="system prompt",
    )
    state = {"messages": [HumanMessage(content="Hola")]}

    result = node(state=state, runtime=_runtime())  # type: ignore[arg-type]

    assert len(result["messages"]) == 1
    assert result["messages"][0].content == "Respuesta del asistente"


def test_react_node_uses_knowledge_tool_context() -> None:
    captured = {"messages": None}

    def fake_llm(inputs):
        captured["messages"] = inputs.to_messages()
        return AIMessage(content="Respuesta con contexto")

    class FakeKnowledgeTool:
        def search(self, query: str) -> str:
            assert query == "Hola"
            return "dato relevante"

    node = ReactNode(
        llm=RunnableLambda(fake_llm),
        prompt="system prompt",
        knowledge_tool=FakeKnowledgeTool(),
    )
    state = {"messages": [HumanMessage(content="Hola")]}

    result = node(state=state, runtime=_runtime())  # type: ignore[arg-type]

    assert result["messages"][0].content == "Respuesta con contexto"
    assert captured["messages"] is not None
    system_contents = [message.content for message in captured["messages"] if message.type == "system"]
    assert any("dato relevante" in content for content in system_contents)
