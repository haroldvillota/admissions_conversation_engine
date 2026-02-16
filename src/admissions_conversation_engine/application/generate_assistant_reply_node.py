from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from admissions_conversation_engine.domain.state import (
    ConversationState,
    ConversationStateData,
)


class GenerateAssistantReplyNode:
    def __init__(self, llm: object, prompt: str) -> None:
        self._llm = llm
        self._prompt = prompt

    def __call__(self, state: ConversationStateData) -> ConversationStateData:
        conversation_state = ConversationState.from_graph_state(state)
        if not conversation_state.last_user_input.strip():
            return state
        
        #system_prompt = SystemMessage(content=self._prompt)
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", self._prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])
                
        #messages = [system_prompt] + [self._to_langchain_message(message) for message in state["messages"]]
        #response = self._llm.invoke(messages)
        chain = prompt_template | self._llm
        response = chain.invoke(state)
        updated_state = conversation_state.with_appended_message(
            "assistant",
            response.content,
        )
        return updated_state.to_graph_state()

    @staticmethod
    def _to_langchain_message(message: dict[str, str]) -> HumanMessage | AIMessage | SystemMessage:
        role = message["role"]
        content = message["content"]
        if role == "user":
            return HumanMessage(content=content)
        if role == "assistant":
            return AIMessage(content=content)
        if role == "system":
            return SystemMessage(content=content)
        return HumanMessage(content=content)
