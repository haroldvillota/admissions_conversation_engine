from types import SimpleNamespace

from admissions_conversation_engine.application.setup_chat_node import SetupChatNode
from admissions_conversation_engine.domain.agent_state import ContextSchema


def _runtime() -> SimpleNamespace:
    context = ContextSchema(
        chat_id="chat-123",
        channel_id="wa",
        case="off_hours",
        user_name="Pedro",
    )
    return SimpleNamespace(context=context)


def test_setup_chat_node_returns_same_state(capsys) -> None:
    node = SetupChatNode()
    state = {"messages": []}

    result = node(state=state, config={}, runtime=_runtime())  # type: ignore[arg-type]

    assert result is state
    captured = capsys.readouterr()
    assert "chat_id:chat-123" in captured.out
