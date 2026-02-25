from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langfuse import Langfuse

from admissions_conversation_engine.domain.tenant_config import TenantConfig
from admissions_conversation_engine.domain.prompts.guardrail_prompt import GUARDRAIL_PROMPT
from admissions_conversation_engine.domain.prompts.language_detector_prompt import LANGUAGE_DETECTOR_PROMPT
from admissions_conversation_engine.domain.prompts.case_off_hours_prompt import OFF_HOURS_PROMPT
from admissions_conversation_engine.domain.prompts.case_low_scoring_prompt import LOW_SCORING_PROMPT
from admissions_conversation_engine.domain.prompts.case_overflow_prompt import OVERFLOW_PROMPT
from admissions_conversation_engine.domain.prompts.case_max_retries_prompt import MAX_RETRIES_PROMPT
from admissions_conversation_engine.domain.prompts.render_guardrail_prompt import render_guardrail_prompt
from admissions_conversation_engine.domain.prompts.render_language_detector_prompt import render_language_detector_prompt
from admissions_conversation_engine.domain.prompts.render_case_off_hours_prompt import render_case_off_hours_prompt
from admissions_conversation_engine.domain.prompts.render_case_low_scoring_prompt import render_case_low_scoring_prompt
from admissions_conversation_engine.domain.prompts.render_case_overflow_prompt import render_case_overflow_prompt
from admissions_conversation_engine.domain.prompts.render_case_max_retries_prompt import render_case_max_retries_prompt


@dataclass(frozen=True)
class FormattedPrompts:
    guardrail: Any
    language_detector: Any
    off_hours: Any
    low_scoring: Any
    overflow: Any
    max_retries: Any


@dataclass(frozen=True)
class PromptProvider:
    langfuse_client: Langfuse | None
    tenant: TenantConfig

    def get_formatted_prompts(self) -> FormattedPrompts:
        if self.langfuse_client is not None:
            guardrail_prompt = self.langfuse_client.get_prompt("guardrail")
            language_detector_prompt = self.langfuse_client.get_prompt("language_detector")
            off_hours_prompt = self.langfuse_client.get_prompt("case_off_hours")
            low_scoring_prompt = self.langfuse_client.get_prompt("case_low_scoring")
            overflow_prompt = self.langfuse_client.get_prompt("case_overflow")
            max_retries_prompt = self.langfuse_client.get_prompt("case_max_retries")
            return FormattedPrompts(
                guardrail=render_guardrail_prompt(guardrail_prompt.prompt, self.tenant),
                language_detector=render_language_detector_prompt(language_detector_prompt.prompt, self.tenant),
                off_hours=render_case_off_hours_prompt(off_hours_prompt.prompt, self.tenant),
                low_scoring=render_case_low_scoring_prompt(low_scoring_prompt.prompt, self.tenant),
                overflow=render_case_overflow_prompt(overflow_prompt.prompt, self.tenant),
                max_retries=render_case_max_retries_prompt(max_retries_prompt.prompt, self.tenant),
            )
        else:
            return FormattedPrompts(
                guardrail=render_guardrail_prompt(GUARDRAIL_PROMPT, self.tenant),
                language_detector=render_language_detector_prompt(LANGUAGE_DETECTOR_PROMPT, self.tenant),
                off_hours=render_case_off_hours_prompt(OFF_HOURS_PROMPT, self.tenant),
                low_scoring=render_case_low_scoring_prompt(LOW_SCORING_PROMPT, self.tenant),
                overflow=render_case_overflow_prompt(OVERFLOW_PROMPT, self.tenant),
                max_retries=render_case_max_retries_prompt(MAX_RETRIES_PROMPT, self.tenant),
            )
