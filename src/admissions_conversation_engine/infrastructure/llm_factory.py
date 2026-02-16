from __future__ import annotations

from langchain.chat_models import BaseChatModel, init_chat_model

from admissions_conversation_engine.infrastructure.config.app_config import LlmProfileConfig

class LLMFactory:
    def __init__(self, config: LlmProfileConfig) -> None:
        self._config : LlmProfileConfig = config

    def build_llm(self) -> BaseChatModel:
        
        return init_chat_model(
            self._config.model,                      
            api_key = self._config.api_key
        )
