from __future__ import annotations

from admissions_conversation_engine.domain.prompts.case_low_scoring_prompt import (
    LOW_SCORING_PROMPT,
)
from admissions_conversation_engine.domain.tenant_config import TenantConfig


def render_case_low_scoring_prompt(config: TenantConfig) -> str:
    return LOW_SCORING_PROMPT.format(
        institution=config.institution,
        allowed_topics=config.allowed_topics,
        tone=config.tone,
        allowed_languages=config.allowed_languages,
        language_fallback=config.language_fallback,
        terms_of_service=config.terms_of_service,
    )
