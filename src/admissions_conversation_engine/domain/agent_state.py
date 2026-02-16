from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    
    # Language
    language: str | None
    language_confidence: float | None

    # Guardrail
    guardrail_allowed: bool
    guardrail_reason: Literal["OK", "PROHIBITED_TOPIC", "OUT_OF_SCOPE", "INJECTION"]

    # Router
    next_node: str | None


@dataclass
class ContextSchema:
    chat_id: str
    channel_id: str
    case: Literal["off_hours", "low_scoring", "overflow", "max_retries"]
    user_name: str
    