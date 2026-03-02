from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from admissions_conversation_engine.application.tool_node import ToolNode
from admissions_conversation_engine.domain.agent_state import ContextSchema


def _runtime() -> SimpleNamespace:
    context = ContextSchema(
        chat_id="chat-1",
        channel_id="wa",
        case="off_hours",
        user_name="Pedro",
    )
    return SimpleNamespace(context=context)


@pytest.mark.asyncio
async def test_tool_node_returns_tool_messages_for_each_tool_call() -> None:
    # Verifica que el nodo de herramientas crea un ToolMessage por cada tool_call presente en el último mensaje.
    class FakeKnowledgeTool:
        name = "search_knowledge"

        async def ainvoke(self, args: dict) -> str:
            return f"resultado para: {args['query']}"

    last_message = AIMessage(
        content="",
        tool_calls=[
            {"id": "tc-1", "name": "search_knowledge", "args": {"query": "admisiones"}},
        ],
    )
    state = {"messages": [last_message]}

    node = ToolNode(llm=object(), knowledge_tool=FakeKnowledgeTool())

    result = await node(state=state, runtime=_runtime())  # type: ignore[arg-type]

    assert len(result["messages"]) == 1
    tool_msg = result["messages"][0]
    assert tool_msg.name == "search_knowledge"
    assert tool_msg.tool_call_id == "tc-1"
    assert "admisiones" in tool_msg.content


@pytest.mark.asyncio
async def test_tool_node_returns_multiple_tool_messages_for_multiple_calls() -> None:
    # Verifica que el nodo procesa correctamente múltiples tool_calls y devuelve un ToolMessage por cada uno.
    call_log: list[str] = []

    class FakeKnowledgeTool:
        name = "search_knowledge"

        async def ainvoke(self, args: dict) -> str:
            call_log.append(args["query"])
            return f"info sobre {args['query']}"

    last_message = AIMessage(
        content="",
        tool_calls=[
            {"id": "tc-1", "name": "search_knowledge", "args": {"query": "requisitos"}},
            {"id": "tc-2", "name": "search_knowledge", "args": {"query": "plazos"}},
        ],
    )
    state = {"messages": [last_message]}

    node = ToolNode(llm=object(), knowledge_tool=FakeKnowledgeTool())

    result = await node(state=state, runtime=_runtime())  # type: ignore[arg-type]

    assert len(result["messages"]) == 2
    assert result["messages"][0].tool_call_id == "tc-1"
    assert result["messages"][1].tool_call_id == "tc-2"
    assert call_log == ["requisitos", "plazos"]


@pytest.mark.asyncio
async def test_tool_node_invokes_tool_with_correct_args() -> None:
    # Verifica que el nodo pasa los argumentos del tool_call exactamente como los recibió al invocar la herramienta.
    received_args: dict = {}

    class FakeKnowledgeTool:
        name = "search_knowledge"

        async def ainvoke(self, args: dict) -> str:
            received_args.update(args)
            return "resultado"

    last_message = AIMessage(
        content="",
        tool_calls=[
            {"id": "tc-1", "name": "search_knowledge", "args": {"query": "becas de posgrado"}},
        ],
    )
    state = {"messages": [last_message]}

    node = ToolNode(llm=object(), knowledge_tool=FakeKnowledgeTool())

    await node(state=state, runtime=_runtime())  # type: ignore[arg-type]

    assert received_args == {"query": "becas de posgrado"}
