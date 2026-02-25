from __future__ import annotations

from admissions_conversation_engine.domain.tenant_config import TenantConfig


def render_language_detector_prompt(prompt: str, config: TenantConfig) -> str:
    return prompt.format(
        allowed_languages=config.allowed_languages,
        language_fallback=config.language_fallback,
    )
