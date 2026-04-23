"""LLM insight classification infrastructure."""

from med_assert.infrastructure.insights.prompts import (
    PROMPT_VERSION,
    build_user_prompt,
    system_prompt,
)

__all__ = [
    "PROMPT_VERSION",
    "build_user_prompt",
    "system_prompt",
]
