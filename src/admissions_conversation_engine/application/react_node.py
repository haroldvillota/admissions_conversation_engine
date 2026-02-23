from __future__ import annotations

import asyncio
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.runtime import Runtime
from admissions_conversation_engine.domain.agent_state import AgentState, ContextSchema
from dataclasses import asdict


class ReactNode:
    def __init__(self, llm: object, prompt: str, knowledge_tool: object | None = None) -> None:
        self._llm = llm
        self._prompt = prompt
        self._knowledge_tool = knowledge_tool

    async def __call__(self, state: AgentState, runtime: Runtime[ContextSchema]) -> AgentState:
        user_message = state["messages"][-1].content if state.get("messages") else ""
        retrieved_context = ""

        if self._knowledge_tool is not None and user_message:
            try:
                # ASYNC-LIMITATION: knowledge_tool.search es síncrono (no API async estándar).
                # Usamos to_thread para evitar bloquear el event loop bajo alta concurrencia.
                retrieved_context = await asyncio.to_thread(self._knowledge_tool.search, user_message)
            except Exception:
                retrieved_context = ""

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", self._prompt),
            (
                "system",
                "Contexto relevante de conocimiento interno para responder (usar solo si aplica):\n{retrieved_context}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        context_dict = asdict(runtime.context)
        full_inputs = {**state, **context_dict, "retrieved_context": retrieved_context}

        chain = prompt_template | self._llm
        response = await chain.ainvoke(full_inputs)
        return {"messages": [response]}
        
