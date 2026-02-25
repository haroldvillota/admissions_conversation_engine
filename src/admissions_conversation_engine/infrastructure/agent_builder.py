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

from langfuse import Langfuse

from admissions_conversation_engine.domain.agent_state import AgentState, ContextSchema
from admissions_conversation_engine.infrastructure.llm_factory import LLMFactory
from dataclasses import dataclass

from admissions_conversation_engine.infrastructure.config.app_config import AppConfig

from admissions_conversation_engine.domain.prompts.render_guardrail_prompt import (
    render_guardrail_prompt,
)
from admissions_conversation_engine.domain.prompts.guardrail_prompt import GUARDRAIL_PROMPT
from admissions_conversation_engine.domain.prompts.language_detector_prompt import (
    LANGUAGE_DETECTOR_PROMPT,
)
from admissions_conversation_engine.domain.prompts.render_language_detector_prompt import (
    render_language_detector_prompt,
)
from admissions_conversation_engine.domain.prompts.case_off_hours_prompt import OFF_HOURS_PROMPT
from admissions_conversation_engine.domain.prompts.render_case_off_hours_prompt import (
    render_case_off_hours_prompt,
)
from admissions_conversation_engine.domain.prompts.case_low_scoring_prompt import (
    LOW_SCORING_PROMPT,
)
from admissions_conversation_engine.domain.prompts.render_case_low_scoring_prompt import (
    render_case_low_scoring_prompt,
)
from admissions_conversation_engine.domain.prompts.case_overflow_prompt import (
    OVERFLOW_PROMPT,
)
from admissions_conversation_engine.domain.prompts.render_case_overflow_prompt import (
    render_case_overflow_prompt,
)
from admissions_conversation_engine.domain.prompts.case_max_retries_prompt import (
    MAX_RETRIES_PROMPT,
)
from admissions_conversation_engine.domain.prompts.render_case_max_retries_prompt import (
    render_case_max_retries_prompt,
)
from admissions_conversation_engine.infrastructure.rag_postgres_tool import PostgresVectorStoreTool

@dataclass(frozen=True)
class AgentBuilder:

    checkpointer: Any
    app_config: AppConfig
    langfuse_client: Langfuse

    def build(self) -> Any:

        guardrail_llm = LLMFactory(self.app_config.llm.guardrail).build_llm()
        translator_llm = LLMFactory(self.app_config.llm.translator).build_llm()
        llm = LLMFactory(self.app_config.llm.react).build_llm()
        
        search_tool = PostgresVectorStoreTool(
            rag_config=self.app_config.rag,
        )

        llm_with_tool = llm.bind_tools([search_tool])

        if self.langfuse_client is not None:
            guardrail_prompt = self.langfuse_client.get_prompt("guardrail")
            language_detector_prompt = self.langfuse_client.get_prompt("language_detector")
            off_hours_prompt = self.langfuse_client.get_prompt("case_off_hours")
            low_scoring_prompt = self.langfuse_client.get_prompt("case_low_scoring")
            overflow_prompt = self.langfuse_client.get_prompt("case_overflow")
            max_retries_prompt = self.langfuse_client.get_prompt("case_max_retries")
            formatted_guardrail_prompt = render_guardrail_prompt(
                guardrail_prompt.prompt, self.app_config.tenant
            )
            formatted_language_detector_prompt = render_language_detector_prompt(
                language_detector_prompt.prompt, self.app_config.tenant
            )
            formatted_off_hours_prompt = render_case_off_hours_prompt(
                off_hours_prompt.prompt, self.app_config.tenant
            )
            formatted_low_scoring_prompt = render_case_low_scoring_prompt(
                low_scoring_prompt.prompt, self.app_config.tenant
            )
            formatted_overflow_prompt = render_case_overflow_prompt(
                overflow_prompt.prompt, self.app_config.tenant
            )
            formatted_max_retries_prompt = render_case_max_retries_prompt(
                max_retries_prompt.prompt, self.app_config.tenant
            )
        else:
            formatted_guardrail_prompt = render_guardrail_prompt(
                GUARDRAIL_PROMPT, self.app_config.tenant
            )
            formatted_language_detector_prompt = render_language_detector_prompt(
                LANGUAGE_DETECTOR_PROMPT, self.app_config.tenant
            )
            formatted_off_hours_prompt = render_case_off_hours_prompt(
                OFF_HOURS_PROMPT, self.app_config.tenant
            )
            formatted_low_scoring_prompt = render_case_low_scoring_prompt(
                LOW_SCORING_PROMPT, self.app_config.tenant
            )
            formatted_overflow_prompt = render_case_overflow_prompt(
                OVERFLOW_PROMPT, self.app_config.tenant
            )
            formatted_max_retries_prompt = render_case_max_retries_prompt(
                MAX_RETRIES_PROMPT, self.app_config.tenant
            )

        graph = StateGraph(AgentState, context_schema=ContextSchema)
        
        graph.add_node("setup", SetupChatNode())
        graph.add_node(
            "language_detector",
            LlmLanguageDetectorNode(
                translator_llm,
                formatted_language_detector_prompt,
                self.app_config.tenant,
            ),
        )
        graph.add_node("guardrail", GuardrailNode(guardrail_llm, formatted_guardrail_prompt))
        graph.add_node("case_router", CaseRouterNode())

        graph.add_node("off_hours_node", SimpleLLMNode(llm_with_tool, formatted_off_hours_prompt))
        graph.add_node("low_scoring_node", SimpleLLMNode(llm_with_tool, formatted_low_scoring_prompt))
        graph.add_node("overflow_node", SimpleLLMNode(llm_with_tool, formatted_overflow_prompt))
        graph.add_node("max_retries_node", SimpleLLMNode(llm_with_tool, formatted_max_retries_prompt))

        graph.add_node("tools", ToolNode(llm_with_tool, search_tool))

        graph.add_edge(START, "setup")
        graph.add_edge("setup", "language_detector")
        graph.add_edge("language_detector", "guardrail")

        graph.add_conditional_edges(
            "guardrail",
            lambda state: "blocked" if not state.get("guardrail_allowed") else "continue",
            {
                "blocked": END,
                "continue": "case_router",
            },
        )

        graph.add_conditional_edges(
            "case_router",
            lambda state: state.get("case_node"),
            {
                "off_hours_node": "off_hours_node",
                "low_scoring_node": "low_scoring_node",
                "overflow_node": "overflow_node",
                "max_retries_node": "max_retries_node",
                "END": END,
            },
        )

        graph.add_conditional_edges(
            "off_hours_node",
            AgentBuilder._should_continue_to_tools,
            {
                "continue": "tools",
                "end": END,
            },
        )

        graph.add_conditional_edges(
            "low_scoring_node",
            AgentBuilder._should_continue_to_tools,
            {
                "continue": "tools",
                "end": END,
            },
        )

        graph.add_conditional_edges(
            "overflow_node",
            AgentBuilder._should_continue_to_tools,
            {
                "continue": "tools",
                "end": END,
            },
        )

        graph.add_conditional_edges(
            "max_retries_node",
            AgentBuilder._should_continue_to_tools,
            {
                "continue": "tools",
                "end": END,
            },
        )

        graph.add_conditional_edges(
            "tools",
            lambda state: state.get("case_node"),
            {
                "off_hours_node": "off_hours_node",
                "low_scoring_node": "low_scoring_node",
                "overflow_node": "overflow_node",
                "max_retries_node": "max_retries_node",
            }
        )

        return graph.compile(
            checkpointer=self.checkpointer,
        )

    @staticmethod
    def _should_continue_to_tools(state: AgentState) -> str:
        messages = state["messages"]
        last_message = messages[-1]
        if not last_message.tool_calls:
            return "end"
        else:
            return "continue"
