from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from admissions_conversation_engine.application.guardrail_node import GuardrailNode
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


def test_guardrail_allows_request_when_llm_returns_allowed_true() -> None:
    node = GuardrailNode(
        llm=_llm_with_content('{"allowed": true, "reason": "OK", "safe_reply": ""}'),
        prompt="guardrail",
    )
    state = {"messages": [HumanMessage(content="hola")]}

    result = node(state=state, runtime=_runtime())  # type: ignore[arg-type]

    assert result == {"guardrail_allowed": True, "guardrail_reason": "OK"}


def test_guardrail_blocks_request_with_safe_reply() -> None:
    node = GuardrailNode(
        llm=_llm_with_content(
            '{"allowed": false, "reason": "PROHIBITED_TOPIC", "safe_reply": "No puedo ayudarte con eso."}'
        ),
        prompt="guardrail",
    )
    state = {"messages": [HumanMessage(content="dime precios")]}

    result = node(state=state, runtime=_runtime())  # type: ignore[arg-type]

    assert result["guardrail_allowed"] is False
    assert result["guardrail_reason"] == "PROHIBITED_TOPIC"
    assert result["messages"][0].content == "No puedo ayudarte con eso."


def test_guardrail_uses_fail_safe_when_llm_returns_invalid_json() -> None:
    node = GuardrailNode(
        llm=_llm_with_content("not-json"),
        prompt="guardrail",
    )
    state = {"messages": [HumanMessage(content="hola")]}

    result = node(state=state, runtime=_runtime())  # type: ignore[arg-type]

    assert result["guardrail_allowed"] is False
    assert result["guardrail_reason"] == "INJECTION"
    assert result["messages"][0].content == "I’m sorry, I can’t help with that request."
