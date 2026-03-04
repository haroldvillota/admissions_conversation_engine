from __future__ import annotations

from langchain.chat_models import BaseChatModel, init_chat_model

from admissions_conversation_engine.infrastructure.config.app_config import LlmProfileConfig

class LLMFactory:
    def __init__(self, config: LlmProfileConfig) -> None:
        self._config: LlmProfileConfig = config
        self._llm = None

    def build(self) -> LLMFactory:
        if not self._config.api_key:
            raise RuntimeError(
                f"API key no configurada para el modelo LLM '{self._config.model}'"
            )

        self._llm = init_chat_model(
            self._config.model,
            api_key=self._config.api_key,
        )

        return self

    def health_check(self) -> None:
        if self._llm is None:
            raise RuntimeError("LLM no construido. Ejecuta build() primero.")

        try:
            self._llm.invoke("ping")
        except Exception as exc:
            raise RuntimeError(
                f"API del modelo LLM '{self._config.model}' no disponible: {exc}"
            ) from exc

    def get_llm(self) -> BaseChatModel:
        if self._llm is None:
            raise RuntimeError("LLM no construido. Ejecuta build() primero.")
        return self._llm