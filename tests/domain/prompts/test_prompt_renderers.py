from admissions_conversation_engine.domain.prompts.render_case_low_scoring_prompt import (
    render_case_low_scoring_prompt,
)
from admissions_conversation_engine.domain.prompts.case_low_scoring_prompt import (
    LOW_SCORING_PROMPT,
)
from admissions_conversation_engine.domain.prompts.render_case_max_retries_prompt import (
    render_case_max_retries_prompt,
)
from admissions_conversation_engine.domain.prompts.case_max_retries_prompt import (
    MAX_RETRIES_PROMPT,
)
from admissions_conversation_engine.domain.prompts.render_case_off_hours_prompt import (
    render_case_off_hours_prompt,
)
from admissions_conversation_engine.domain.prompts.case_off_hours_prompt import (
    OFF_HOURS_PROMPT,
)
from admissions_conversation_engine.domain.prompts.render_case_overflow_prompt import (
    render_case_overflow_prompt,
)
from admissions_conversation_engine.domain.prompts.case_overflow_prompt import (
    OVERFLOW_PROMPT,
)
from admissions_conversation_engine.domain.prompts.render_guardrail_prompt import (
    render_guardrail_prompt,
)
from admissions_conversation_engine.domain.prompts.guardrail_prompt import (
    GUARDRAIL_PROMPT,
)
from admissions_conversation_engine.domain.prompts.render_language_detector_prompt import (
    render_language_detector_prompt,
)
from admissions_conversation_engine.domain.prompts.language_detector_prompt import (
    LANGUAGE_DETECTOR_PROMPT,
)
from admissions_conversation_engine.domain.tenant_config import TenantConfig


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
    # Verifica que el prompt del detector de idioma incluye la lista de idiomas permitidos y el idioma de respaldo.
    prompt = render_language_detector_prompt(LANGUAGE_DETECTOR_PROMPT, _tenant())

    assert "es-ES,en-US" in prompt
    assert '"en-US"' in prompt


def test_render_guardrail_prompt_includes_allowed_topics() -> None:
    # Verifica que el prompt del guardrail incluye los temas permitidos y mantiene el placeholder del mensaje del usuario.
    prompt = render_guardrail_prompt(GUARDRAIL_PROMPT, _tenant())

    assert "Admisiones,Información de carreras" in prompt
    assert "{user_message}" in prompt


def test_render_case_off_hours_prompt_includes_key_tenant_fields() -> None:
    # Verifica que el prompt de fuera de horario incluye la institución y mantiene el placeholder del nombre de usuario.
    prompt = render_case_off_hours_prompt(OFF_HOURS_PROMPT, _tenant())

    assert "Universidad Europea" in prompt
    assert "{user_name}" in prompt


def test_render_case_low_scoring_prompt_includes_key_tenant_fields() -> None:
    # Verifica que el prompt de baja puntuación incluye la institución, los temas permitidos y el placeholder del usuario.
    prompt = render_case_low_scoring_prompt(LOW_SCORING_PROMPT, _tenant())

    assert "Universidad Europea" in prompt
    assert "Admisiones,Información de carreras" in prompt
    assert "{user_name}" in prompt


def test_render_case_overflow_prompt_includes_key_tenant_fields() -> None:
    # Verifica que el prompt de desbordamiento incluye la institución, los temas permitidos y el placeholder del usuario.
    prompt = render_case_overflow_prompt(OVERFLOW_PROMPT, _tenant())

    assert "Universidad Europea" in prompt
    assert "Admisiones,Información de carreras" in prompt
    assert "{user_name}" in prompt


def test_render_case_max_retries_prompt_includes_key_tenant_fields() -> None:
    # Verifica que el prompt de máximo de reintentos incluye la institución, los temas permitidos y el placeholder del usuario.
    prompt = render_case_max_retries_prompt(MAX_RETRIES_PROMPT, _tenant())

    assert "Universidad Europea" in prompt
    assert "Admisiones,Información de carreras" in prompt
    assert "{user_name}" in prompt

