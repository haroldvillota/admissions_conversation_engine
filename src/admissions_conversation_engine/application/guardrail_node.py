import json
from dataclasses import asdict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime
from admissions_conversation_engine.domain.agent_state import AgentState, ContextSchema

class GuardrailNode:
    def __init__(self, llm: object, prompt: str) -> None:
        self._llm = llm
        self._prompt = prompt

    def __call__(self, state: AgentState, runtime: Runtime[ContextSchema]) -> AgentState:
        
        user_message = state["messages"][-1].content if state.get("messages") else ""

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", self._prompt),
        ])

        context_dict = asdict(runtime.context)

        full_inputs = {
            **state,
            **context_dict,
            "user_message": user_message,
        }

        chain = prompt_template | self._llm
        response = chain.invoke(full_inputs)

        # Parseo de JSON
        try:
            parsed = json.loads(response.content)
        except Exception:
            # Fail-safe: si el modelo responde mal, bloqueamos
            return {
                "guardrail_allowed": False,
                "guardrail_reason": "INJECTION",
                "messages": [
                    AIMessage(content="I’m sorry, I can’t help with that request.")
                ],
            }

        allowed = parsed.get("allowed", False)
        reason = parsed.get("reason", "INJECTION")
        safe_reply = parsed.get("safe_reply", "")

        if not allowed:
            return {
                "guardrail_allowed": False,
                "guardrail_reason": reason,
                "messages": [AIMessage(content=safe_reply)],
            }

        return {
            "guardrail_allowed": True,
            "guardrail_reason": "OK",
        }
