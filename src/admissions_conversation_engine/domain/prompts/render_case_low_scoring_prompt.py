from __future__ import annotations

from admissions_conversation_engine.domain.tenant_config import TenantConfig


def render_case_low_scoring_prompt(prompt: str, config: TenantConfig) -> str:
    return prompt.format(
        institution=config.institution,
        allowed_topics=config.allowed_topics,
        tone=config.tone,
        allowed_languages=config.allowed_languages,
        language_fallback=config.language_fallback,
        terms_of_service=config.terms_of_service,
    )
