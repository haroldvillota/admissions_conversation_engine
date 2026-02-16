from types import SimpleNamespace

from admissions_conversation_engine.application.case_router_node import CaseRouterNode
from admissions_conversation_engine.domain.agent_state import ContextSchema


def _runtime_with_case(case: str) -> SimpleNamespace:
    context = ContextSchema(
        chat_id="chat-1",
        channel_id="wa",
        case=case,  # type: ignore[arg-type]
        user_name="Pedro",
    )
    return SimpleNamespace(context=context)


def test_case_router_maps_known_case_to_node() -> None:
    node = CaseRouterNode()

    result = node(state={}, runtime=_runtime_with_case("overflow"))  # type: ignore[arg-type]

    assert result == {"next_node": "overflow_node"}


def test_case_router_returns_end_for_unknown_case() -> None:
    node = CaseRouterNode()

    result = node(state={}, runtime=_runtime_with_case("unknown_case"))  # type: ignore[arg-type]

    assert result == {"next_node": "END"}
