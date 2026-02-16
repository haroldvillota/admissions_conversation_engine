from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.runtime import Runtime
from admissions_conversation_engine.domain.agent_state import AgentState, ContextSchema
from dataclasses import asdict


class SimpleLLMNode:
    def __init__(self, llm: object, prompt: str) -> None:
        self._llm = llm
        self._prompt = prompt

    def __call__(self, state: AgentState, runtime: Runtime[ContextSchema]) -> AgentState:

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", self._prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        context_dict = asdict(runtime.context)
        full_inputs = {**state, **context_dict}

        chain = prompt_template | self._llm
        response = chain.invoke(full_inputs)
        return {"messages": [response]}
        

