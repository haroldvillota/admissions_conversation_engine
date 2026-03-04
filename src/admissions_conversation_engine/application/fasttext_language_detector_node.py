from typing import List

from langgraph.runtime import Runtime
from pathlib import Path
from admissions_conversation_engine.domain.agent_state import AgentState, ContextSchema
from admissions_conversation_engine.domain.tenant_config import TenantConfig


class FasttextLanguageDetectorNode:
    """
    Detecta el idioma del mensaje usando un modelo fasttext local.
    No realiza llamadas a ningún LLM externo.
    """

    def __init__(self, config: TenantConfig, model_path: str) -> None:
        import fasttext

        self._config = config
        fasttext.FastText.eprint = lambda x: None  # silencia warnings de fasttext
        PROJECT_ROOT = Path(__file__).resolve().parents[3]
        self._model = fasttext.load_model(str(PROJECT_ROOT / model_path))

    def _parse_allowed_languages(self) -> List[str]:
        if not self._config.allowed_languages:
            return []
        return [
            lang.strip()
            for lang in self._config.allowed_languages.split(",")
            if lang.strip()
        ]

    def _match_language(self, fasttext_code: str, allowed_languages: List[str]) -> str | None:
        """
        Mapea el código ISO 639-1 de fasttext (p. ej. 'es') al código BCP-47
        de la lista permitida (p. ej. 'es-ES').
        """
        for lang in allowed_languages:
            base = lang.split("-")[0].lower()
            if base == fasttext_code.lower():
                return lang
        return None

    async def __call__(self, state: AgentState, runtime: Runtime[ContextSchema]) -> AgentState:
        """
        Detecta el idioma usando fasttext y lo guarda en el estado:
        - language
        - language_confidence

        No agrega mensajes al chat.
        """
        user_message = state["messages"][-1].content if state.get("messages") else ""

        allowed_languages = self._parse_allowed_languages()
        fallback_language = self._config.language_fallback

        # fasttext no acepta saltos de línea
        text = user_message.replace("\n", " ").strip() or " "

        labels, probs = self._model.predict(text, k=1)

        raw_code = labels[0].replace("__label__", "")
        confidence = max(0.0, min(1.0, float(probs[0])))

        detected_language = self._match_language(raw_code, allowed_languages)
        if not detected_language:
            detected_language = fallback_language

        return {
            "language": detected_language,
            "language_confidence": confidence,
        }
