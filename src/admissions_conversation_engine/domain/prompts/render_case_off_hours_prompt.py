from __future__ import annotations

from admissions_conversation_engine.domain.prompts.case_off_hours_prompt import (
    OFF_HOURS_PROMPT,
)
from admissions_conversation_engine.domain.tenant_config import TenantConfig


def render_case_off_hours_prompt(config: TenantConfig) -> str:
    return OFF_HOURS_PROMPT.format(
        institution=config.institution,
        tone=config.tone,
    )
