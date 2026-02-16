from __future__ import annotations

from admissions_conversation_engine.domain.state import (
    ConversationState,
    ConversationStateData,
)


class CaptureUserInputNode:
    def __call__(self, state: ConversationStateData) -> ConversationStateData:
        conversation_state = ConversationState.from_graph_state(state)
        user_input = conversation_state.last_user_input.strip()
        if not user_input:
            return state
        updated_state = conversation_state.with_appended_message("user", user_input)
        return updated_state.to_graph_state()
