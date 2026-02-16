from admissions_conversation_engine.domain.prompts.case_low_scoring_prompt import (
    render_case_low_scoring_prompt,
)
from admissions_conversation_engine.domain.prompts.case_max_retries_prompt import (
    render_case_max_retries_prompt,
)
from admissions_conversation_engine.domain.prompts.case_off_hours_prompt import (
    render_case_off_hours_prompt,
)
from admissions_conversation_engine.domain.prompts.case_overflow_prompt import (
    render_case_overflow_prompt,
)
from admissions_conversation_engine.domain.prompts.guardrail_prompt import (
    render_guardrail_prompt,
)
from admissions_conversation_engine.domain.prompts.language_detector_prompt import (
    render_language_detector_prompt,
)
from admissions_conversation_engine.domain.prompts.react_prompt import render_react_prompt
from admissions_conversation_engine.infrastructure.config.app_config import TenantConfig


def _tenant() -> TenantConfig:
    return TenantConfig(
        institution="Universidad Europea",
        terms_of_service="https://example.com/tos",
        allowed_topics="Admisiones,Información de carreras",
        tone="amigable",
        language_fallback="en-US",
        allowed_languages="es-ES,en-US",
    )


def test_render_language_detector_prompt_includes_language_configuration() -> None:
    prompt = render_language_detector_prompt(_tenant())

    assert "es-ES,en-US" in prompt
    assert '"en-US"' in prompt


def test_render_guardrail_prompt_includes_allowed_topics() -> None:
    prompt = render_guardrail_prompt(_tenant())

    assert "Admisiones,Información de carreras" in prompt
    assert "{user_message}" in prompt


def test_render_case_off_hours_prompt_includes_key_tenant_fields() -> None:
    prompt = render_case_off_hours_prompt(_tenant())

    assert "Universidad Europea" in prompt
    assert "Admisiones,Información de carreras" in prompt
    assert "{user_name}" in prompt


def test_render_case_low_scoring_prompt_includes_key_tenant_fields() -> None:
    prompt = render_case_low_scoring_prompt(_tenant())

    assert "Universidad Europea" in prompt
    assert "Admisiones,Información de carreras" in prompt
    assert "{user_name}" in prompt


def test_render_case_overflow_prompt_includes_key_tenant_fields() -> None:
    prompt = render_case_overflow_prompt(_tenant())

    assert "Universidad Europea" in prompt
    assert "Admisiones,Información de carreras" in prompt
    assert "{user_name}" in prompt


def test_render_case_max_retries_prompt_includes_key_tenant_fields() -> None:
    prompt = render_case_max_retries_prompt(_tenant())

    assert "Universidad Europea" in prompt
    assert "Admisiones,Información de carreras" in prompt
    assert "{user_name}" in prompt


def test_render_react_prompt_includes_language_and_terms() -> None:
    prompt = render_react_prompt(_tenant())

    assert "Universidad Europea" in prompt
    assert "https://example.com/tos" in prompt
    assert "es-ES,en-US" in prompt
    assert "en-US" in prompt
