from __future__ import annotations
from dataclasses import asdict

from langchain_core.runnables import RunnableConfig

from admissions_conversation_engine.domain.agent_state import AgentState, ContextSchema
from langgraph.runtime import Runtime

class SetupChatNode:
   
    def __call__(self, state: AgentState, config: RunnableConfig, runtime: Runtime[ContextSchema]) -> AgentState:
        chat_id = runtime.context.chat_id
        print(f"chat_id:{chat_id}")
        

        return state
        

