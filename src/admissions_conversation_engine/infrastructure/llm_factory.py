from __future__ import annotations

import logging

import openai
from langchain.chat_models import BaseChatModel, init_chat_model

from admissions_conversation_engine.infrastructure.config.app_config import LlmProfileConfig

logger = logging.getLogger(__name__)


class LLMFactory:
    def __init__(self, config: LlmProfileConfig) -> None:
        self._config: LlmProfileConfig = config

    def probe_connection(self) -> None:
        """Verifica que la API del LLM sea alcanzable con la clave configurada.

        Lanza RuntimeError con un mensaje descriptivo si la autenticación o
        la conectividad fallan, para que los entrypoints puedan detener la
        aplicación de forma temprana.
        """
        if not self._config.api_key:
            raise RuntimeError(
                f"API key no configurada para el modelo LLM '{self._config.model}'"
            )
        try:
            openai.OpenAI(api_key=self._config.api_key).models.list(limit=1)
        except Exception as exc:
            raise RuntimeError(
                f"API del modelo LLM '{self._config.model}' no disponible: {exc}"
            ) from exc

    def build_llm(self) -> BaseChatModel:
        return init_chat_model(
            self._config.model,
            api_key=self._config.api_key,
        )
