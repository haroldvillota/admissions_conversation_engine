from __future__ import annotations

from typing import Any
import logging

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

from admissions_conversation_engine.application.fasttext_language_detector_node import (
    FasttextLanguageDetectorNode,
)

from admissions_conversation_engine.application.case_router_node import (
    CaseRouterNode,
)

from langfuse import Langfuse

from admissions_conversation_engine.domain.agent_state import AgentState, ContextSchema
from admissions_conversation_engine.infrastructure.llm_factory import LLMFactory
from dataclasses import dataclass

from admissions_conversation_engine.infrastructure.config.app_config import AppConfig
from admissions_conversation_engine.infrastructure.prompt_provider import PromptProvider
from admissions_conversation_engine.infrastructure.rag_postgres_tool import PostgresVectorStoreTool

@dataclass(frozen=True)
class AgentBuilder:

    checkpointer: Any
    app_config: AppConfig
    langfuse_client: Langfuse

    def build(self) -> Any:
        logger = logging.getLogger(__name__)

        guardrail_factory = LLMFactory(self.app_config.llm.guardrail)
        guardrail_factory.build().health_check()
        logger.info("Guardrail llm ready!")
        guardrail_llm = guardrail_factory.get_llm()

        react_factory = LLMFactory(self.app_config.llm.react)
        react_factory.build().health_check()
        logger.info("React llm ready!")
        react_llm = react_factory.get_llm()

        search_tool = PostgresVectorStoreTool(
            rag_config=self.app_config.rag,
        )
        search_tool.probe_connection()
        logger.info("Vector Store ready!")

        llm_with_tool = react_llm.bind_tools([search_tool])

        prompts = PromptProvider(self.langfuse_client, self.app_config.tenant).get_formatted_prompts()

        graph = StateGraph(AgentState, context_schema=ContextSchema)
        
        graph.add_node("setup", SetupChatNode())

        ld_config = self.app_config.language_detector
        if ld_config.method == "llm":
            language_detector_factory = LLMFactory(self.app_config.llm.translator)
            logger.info("Language detector llm ready!")
            language_detector_factory.build().health_check()
            translator_llm = language_detector_factory.get_llm()
            language_detector = LlmLanguageDetectorNode(
                translator_llm,
                prompts.language_detector,
                self.app_config.tenant,
            )
        else:
            logger.info("Language detector local ready!")
            language_detector = FasttextLanguageDetectorNode(
                self.app_config.tenant,
                model_path=ld_config.fasttext_model_path,
            )

        graph.add_node("language_detector", language_detector)
        graph.add_node("guardrail", GuardrailNode(guardrail_llm, prompts.guardrail))
        graph.add_node("case_router", CaseRouterNode())

        graph.add_node("off_hours_node", SimpleLLMNode(llm_with_tool, prompts.off_hours))
        graph.add_node("low_scoring_node", SimpleLLMNode(llm_with_tool, prompts.low_scoring))
        graph.add_node("overflow_node", SimpleLLMNode(llm_with_tool, prompts.overflow))
        graph.add_node("max_retries_node", SimpleLLMNode(llm_with_tool, prompts.max_retries))

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
