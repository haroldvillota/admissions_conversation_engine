from __future__ import annotations

from admissions_conversation_engine.domain.prompts.guardrail_prompt import GUARDRAIL_PROMPT
from admissions_conversation_engine.domain.tenant_config import TenantConfig


def render_guardrail_prompt(config: TenantConfig) -> str:
    return GUARDRAIL_PROMPT.format(
        allowed_topics=config.allowed_topics,
    )
