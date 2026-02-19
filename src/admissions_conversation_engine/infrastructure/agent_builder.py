from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from admissions_conversation_engine.application.react_node import (
    ReactNode,
)

from admissions_conversation_engine.application.tool_node import (
    ToolNode,
)

from admissions_conversation_engine.application.simple_llm_node import (
    SimpleLLMNode,
)

from admissions_conversation_engine.application.setup_chat_node import (
    SetupChatNode,
)

from admissions_conversation_engine.application.guardrail_node import (
    GuardrailNode,
)

from admissions_conversation_engine.application.llm_language_detector_node import (
    LlmLanguageDetectorNode,
)

from admissions_conversation_engine.application.case_router_node import (
    CaseRouterNode,
)

from admissions_conversation_engine.domain.agent_state import AgentState, ContextSchema
from admissions_conversation_engine.infrastructure.llm_factory import LLMFactory
from dataclasses import dataclass

from admissions_conversation_engine.infrastructure.config.app_config import AppConfig

from admissions_conversation_engine.domain.prompts.guardrail_prompt import render_guardrail_prompt
from admissions_conversation_engine.domain.prompts.language_detector_prompt import render_language_detector_prompt
from admissions_conversation_engine.domain.prompts.case_off_hours_prompt import render_case_off_hours_prompt
from admissions_conversation_engine.domain.prompts.case_low_scoring_prompt import render_case_low_scoring_prompt
from admissions_conversation_engine.domain.prompts.case_overflow_prompt import render_case_overflow_prompt
from admissions_conversation_engine.domain.prompts.case_max_retries_prompt import render_case_max_retries_prompt
from admissions_conversation_engine.infrastructure.rag_postgres_tool import PostgresVectorStoreTool

@dataclass(frozen=True)
class AgentBuilder:
    
    checkpointer: Any 
    app_config: AppConfig

    def build(self) -> Any:

            guardrail_llm = LLMFactory(self.app_config.llm.guardrail).build_llm()
            translator_llm = LLMFactory(self.app_config.llm.translator).build_llm()
            react_llm = LLMFactory(self.app_config.llm.react).build_llm()
            
            search_tool = PostgresVectorStoreTool(
                rag_config=self.app_config.rag
            )

            #search_tool = vector_store.make_search_tool()

            formatted_guardrail_prompt = render_guardrail_prompt(self.app_config.tenant)
            formatted_language_detector_prompt = render_language_detector_prompt(self.app_config.tenant)
            
            formatted_off_hours_prompt = render_case_off_hours_prompt(self.app_config.tenant)
            formatted_low_scoring_prompt = render_case_low_scoring_prompt(self.app_config.tenant)
            formatted_overflow_prompt = render_case_overflow_prompt(self.app_config.tenant)
            formatted_max_retries_prompt = render_case_max_retries_prompt(self.app_config.tenant)

            graph = StateGraph(AgentState, context_schema=ContextSchema)
            graph.add_node("setup", SetupChatNode())
            graph.add_node("language_detector", LlmLanguageDetectorNode(translator_llm, formatted_language_detector_prompt, self.app_config.tenant))
            graph.add_node("guardrail", GuardrailNode(guardrail_llm, formatted_guardrail_prompt))
            graph.add_node("case_router", CaseRouterNode())
            
            tool_model = react_llm.bind_tools([search_tool])
            #graph.add_node("off_hours_node", ReactNode(tool_model, formatted_off_hours_prompt, search_tool))
            graph.add_node("off_hours_node", SimpleLLMNode(tool_model, formatted_off_hours_prompt))
            graph.add_node("tools", ToolNode(tool_model, formatted_off_hours_prompt, search_tool))

            graph.add_node("low_scoring_node", SimpleLLMNode(react_llm, formatted_low_scoring_prompt))
            graph.add_node("overflow_node", SimpleLLMNode(react_llm, formatted_overflow_prompt))
            graph.add_node("max_retries_node", SimpleLLMNode(react_llm, formatted_max_retries_prompt))

            graph.add_edge(START, "setup")
            graph.add_edge("setup", "language_detector")
            graph.add_edge("language_detector", "guardrail")

            graph.add_conditional_edges(
                "guardrail",
                lambda state: "blocked" if not state.get("guardrail_allowed") else "continue",
                {
                    "blocked": END,
                    "continue": "case_router"
                }
            )

            graph.add_conditional_edges(
                "case_router",
                lambda state: state.get("next_node"),
                {
                    "off_hours_node": "off_hours_node",
                    "low_scoring_node": "low_scoring_node",
                    "overflow_node": "overflow_node",
                    "max_retries_node": "max_retries_node",
                    "END": END,
                }
            )
            def should_continue(state: AgentState):
                messages = state["messages"]
                last_message = messages[-1]
                # If there is no function call, then we finish
                if not last_message.tool_calls:
                    return "end"
                # Otherwise if there is, we continue
                else:
                    return "continue"
            
            graph.add_conditional_edges(
                # First, we define the start node. We use `agent`.
                # This means these are the edges taken after the `agent` node is called.
                "off_hours_node",
                # Next, we pass in the function that will determine which node is called next.
                should_continue,
                # Finally we pass in a mapping.
                # The keys are strings, and the values are other nodes.
                # END is a special node marking that the graph should finish.
                # What will happen is we will call `should_continue`, and then the output of that
                # will be matched against the keys in this mapping.
                # Based on which one it matches, that node will then be called.
                {
                    # If `tools`, then we call the tool node.
                    "continue": "tools",
                    # Otherwise we finish.
                    "end": END,
                },
            )

            graph.add_edge("tools", "off_hours_node")
            #graph.add_edge("off_hours_node", END)
            graph.add_edge("low_scoring_node", END)
            graph.add_edge("overflow_node", END)
            graph.add_edge("max_retries_node", END)

            return graph.compile(
                  checkpointer=self.checkpointer
            )
