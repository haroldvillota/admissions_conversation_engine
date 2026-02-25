from __future__ import annotations

import json
from langgraph.runtime import Runtime
from admissions_conversation_engine.domain.agent_state import AgentState, ContextSchema
from langchain_core.messages import ToolMessage


class ToolNode:
    def __init__(self, llm: object, knowledge_tool: object | None = None) -> None:
        self._llm = llm
        self._knowledge_tool = knowledge_tool

    async def __call__(self, state: AgentState, runtime: Runtime[ContextSchema]) -> AgentState:
        tools = [self._knowledge_tool]
        tools_by_name = {tool.name: tool for tool in tools}

        outputs = []
        
        for tool_call in state["messages"][-1].tool_calls:
            tool_result = await tools_by_name[tool_call["name"]].ainvoke(tool_call["args"])
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

        
