from admissions_conversation_engine.domain.tenant_config import TenantConfig
from admissions_conversation_engine.infrastructure.prompt_provider import (
    FormattedPrompts,
    PromptProvider,
)


def _tenant() -> TenantConfig:
    return TenantConfig(
        institution="Universidad Europea",
        terms_of_service="https://example.com/tos",
        allowed_topics="Admisiones",
        tone="empatico",
        language_fallback="en-US",
        allowed_languages="es-ES,en-US",
    )


def test_prompt_provider_uses_hardcoded_templates_when_langfuse_is_none() -> None:
    # Verifica que cuando no hay cliente Langfuse, los prompts se generan desde las plantillas hardcoded del dominio.
    provider = PromptProvider(langfuse_client=None, tenant=_tenant())

    prompts = provider.get_formatted_prompts()

    assert isinstance(prompts, FormattedPrompts)
    assert isinstance(prompts.guardrail, str)
    assert isinstance(prompts.language_detector, str)
    assert isinstance(prompts.off_hours, str)
    assert isinstance(prompts.low_scoring, str)
    assert isinstance(prompts.overflow, str)
    assert isinstance(prompts.max_retries, str)
    # El contenido del tenant debe estar presente en los prompts renderizados
    assert "Universidad Europea" in prompts.off_hours
    assert "Admisiones" in prompts.guardrail


def test_prompt_provider_fetches_all_prompts_from_langfuse_when_client_is_set() -> None:
    # Verifica que cuando hay cliente Langfuse, se solicitan los 6 prompts esperados y se usan para construir FormattedPrompts.
    fetched_names: list[str] = []

    class FakeLangfusePrompt:
        def __init__(self, name: str) -> None:
            self.prompt = f"{name}-template"

    class FakeLangfuseClient:
        def get_prompt(self, name: str) -> FakeLangfusePrompt:
            fetched_names.append(name)
            return FakeLangfusePrompt(name)

    provider = PromptProvider(langfuse_client=FakeLangfuseClient(), tenant=_tenant())  # type: ignore[arg-type]

    prompts = provider.get_formatted_prompts()

    assert isinstance(prompts, FormattedPrompts)
    assert set(fetched_names) == {
        "guardrail",
        "language_detector",
        "case_off_hours",
        "case_low_scoring",
        "case_overflow",
        "case_max_retries",
    }


def test_prompt_provider_renders_langfuse_prompt_with_tenant_context(monkeypatch) -> None:
    # Verifica que los prompts obtenidos de Langfuse se renderizan con el contexto del tenant antes de devolverse.
    class FakeLangfusePrompt:
        def __init__(self, name: str) -> None:
            self.prompt = "{institution}"  # plantilla que usa el tenant

    class FakeLangfuseClient:
        def get_prompt(self, name: str) -> FakeLangfusePrompt:
            return FakeLangfusePrompt(name)

    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.prompt_provider.render_guardrail_prompt",
        lambda template, tenant: f"guardrail:{tenant.institution}",
    )
    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.prompt_provider.render_language_detector_prompt",
        lambda template, tenant: f"detector:{tenant.institution}",
    )
    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.prompt_provider.render_case_off_hours_prompt",
        lambda template, tenant: f"off_hours:{tenant.institution}",
    )
    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.prompt_provider.render_case_low_scoring_prompt",
        lambda template, tenant: f"low_scoring:{tenant.institution}",
    )
    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.prompt_provider.render_case_overflow_prompt",
        lambda template, tenant: f"overflow:{tenant.institution}",
    )
    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.prompt_provider.render_case_max_retries_prompt",
        lambda template, tenant: f"max_retries:{tenant.institution}",
    )

    provider = PromptProvider(langfuse_client=FakeLangfuseClient(), tenant=_tenant())  # type: ignore[arg-type]

    prompts = provider.get_formatted_prompts()

    assert prompts.guardrail == "guardrail:Universidad Europea"
    assert prompts.off_hours == "off_hours:Universidad Europea"
    assert prompts.max_retries == "max_retries:Universidad Europea"
