"""Build LangChain chat models from :class:`InsightLlmResolution`."""

from __future__ import annotations

from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from article_miner.application.insights.llm_provider_registry import InsightLlmResolution


def build_chat_model(resolution: InsightLlmResolution, **kwargs: Any) -> BaseChatModel:
    """Instantiate the appropriate ``BaseChatModel`` for the resolved provider."""
    temperature = float(kwargs.pop("temperature", 0.0))
    if resolution.provider == "openai":
        return ChatOpenAI(
            model=resolution.model_id,
            temperature=temperature,
            **kwargs,
        )
    if resolution.provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=resolution.model_id,
            temperature=temperature,
            **kwargs,
        )
    if resolution.provider == "claude":
        return ChatAnthropic(
            model_name=resolution.model_id,
            temperature=temperature,
            **kwargs,
        )
    if resolution.provider == "ollama":
        base = str(resolution.extra.get("base_url", "http://localhost:11434"))
        return ChatOllama(
            model=resolution.model_id,
            base_url=base,
            temperature=temperature,
            **kwargs,
        )
    msg = f"Unknown provider: {resolution.provider}"
    raise ValueError(msg)


def insight_display_name(resolution: InsightLlmResolution) -> str:
    """Stable string for job results, cache keys, and logs."""
    return f"{resolution.provider}:{resolution.model_id}"
