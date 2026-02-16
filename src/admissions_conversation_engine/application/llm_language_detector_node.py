import json
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langgraph.runtime import Runtime
from admissions_conversation_engine.domain.agent_state import AgentState, ContextSchema
from admissions_conversation_engine.infrastructure.config.app_config import TenantConfig


class LlmLanguageDetectorNode:
    def __init__(self, llm: object, prompt: str, config: TenantConfig) -> None:
        self._llm = llm
        self._prompt = prompt
        self._config = config

    def _parse_allowed_languages(self) -> List[str]:
        """
        Convierte el string 'es,en-US,pt' en lista limpia.
        """
        if not self._config.allowed_languages:
            return []

        return [
            lang.strip()
            for lang in self._config.allowed_languages.split(",")
            if lang.strip()
        ]

    def __call__(self, state: AgentState, runtime: Runtime[ContextSchema]) -> AgentState:
        """
        Detecta el idioma usando LLM y lo guarda en el estado:
        - language
        - language_confidence

        No agrega mensajes al chat.
        """

        user_message = state["messages"][-1].content if state.get("messages") else ""

        allowed_languages = self._parse_allowed_languages()

        # Fallback seguro
        fallback_language = self._config.language_fallback
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", self._prompt),
        ])

        full_inputs = {
            "user_message": user_message
        }

        chain = prompt_template | self._llm
        response = chain.invoke(full_inputs)

        # Parseo robusto del JSON
        try:
            parsed = json.loads(response.content)
        except Exception:
            return {
                "language": fallback_language,
                "language_confidence": 0.0,
            }

        detected_language = parsed.get("language")
        confidence = parsed.get("confidence")

        # Validación idioma detectado
        if not detected_language or detected_language not in allowed_languages:
            detected_language = fallback_language

        # Validación confidence
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0

        # Clamp 0..1
        confidence = max(0.0, min(1.0, confidence))

        return {
            "language": detected_language,
            "language_confidence": confidence,
        }
