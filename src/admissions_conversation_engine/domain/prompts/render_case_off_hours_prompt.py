from __future__ import annotations

from admissions_conversation_engine.domain.tenant_config import TenantConfig


def render_case_off_hours_prompt(prompt: str, config: TenantConfig) -> str:
    return prompt.format(
        institution=config.institution,
        tone=config.tone,
    )
