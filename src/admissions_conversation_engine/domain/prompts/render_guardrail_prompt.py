from __future__ import annotations

from admissions_conversation_engine.domain.tenant_config import TenantConfig


def render_guardrail_prompt(prompt: str, config: TenantConfig) -> str:
    return prompt.format(
        allowed_topics=config.allowed_topics,
    )
