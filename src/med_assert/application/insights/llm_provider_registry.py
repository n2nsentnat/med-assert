"""Registry of insight LLM providers (maps env + CLI to LangChain chat model ids)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from collections.abc import Mapping

# Defaults when env vars are unset (match previous CLI behavior).
_DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
_DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
_DEFAULT_CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
_DEFAULT_OLLAMA_MODEL = "gemma3:4b"
_DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


@dataclass(frozen=True)
class InsightLlmResolution:
    """Canonical provider + model id for LangChain chat models."""

    provider: str
    model_id: str
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class InsightLlmProviderStrategy(Protocol):
    """Strategy: map process env to provider + model id + kwargs."""

    def resolve(self, env: Mapping[str, str]) -> InsightLlmResolution: ...


class _OpenAiStrategy:
    def resolve(self, env: Mapping[str, str]) -> InsightLlmResolution:
        model_id = env.get("INSIGHT_MODEL_OPENAI", _DEFAULT_OPENAI_MODEL)
        return InsightLlmResolution(provider="openai", model_id=model_id)


class _GeminiStrategy:
    def resolve(self, env: Mapping[str, str]) -> InsightLlmResolution:
        model_id = env.get("INSIGHT_MODEL_GEMINI", _DEFAULT_GEMINI_MODEL)
        if model_id.startswith("gemini/"):
            model_id = model_id.split("/", 1)[1]
        return InsightLlmResolution(provider="gemini", model_id=model_id)


class _ClaudeStrategy:
    def resolve(self, env: Mapping[str, str]) -> InsightLlmResolution:
        model_id = env.get("INSIGHT_MODEL_CLAUDE", _DEFAULT_CLAUDE_MODEL)
        if model_id.startswith("anthropic/"):
            model_id = model_id.split("/", 1)[1]
        return InsightLlmResolution(provider="claude", model_id=model_id)


class _OllamaStrategy:
    def resolve(self, env: Mapping[str, str]) -> InsightLlmResolution:
        raw = env.get("OLLAMA_MODEL", _DEFAULT_OLLAMA_MODEL)
        model_id = raw.replace("ollama/", "") if raw.startswith("ollama/") else raw
        base = env.get("OLLAMA_BASE_URL", _DEFAULT_OLLAMA_BASE_URL)
        return InsightLlmResolution(
            provider="ollama",
            model_id=model_id,
            extra={"base_url": base},
        )


_INSIGHT_LLM_STRATEGIES: dict[str, InsightLlmProviderStrategy] = {
    "openai": _OpenAiStrategy(),
    "gemini": _GeminiStrategy(),
    "claude": _ClaudeStrategy(),
    "ollama": _OllamaStrategy(),
}

_PROVIDER_API_KEY_ENV: dict[str, str | None] = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    "ollama": None,
}


def normalize_insight_provider(name: str) -> str:
    """Return canonical provider key: lowercase strip; ``anthropic`` → ``claude``."""
    key = name.strip().lower()
    if key == "anthropic":
        return "claude"
    return key


def registered_insight_providers() -> tuple[str, ...]:
    """Sorted canonical provider ids."""
    return tuple(sorted(_INSIGHT_LLM_STRATEGIES.keys()))


def resolve_insight_llm_provider(
    name: str,
    env: Mapping[str, str],
) -> InsightLlmResolution:
    """Resolve provider label. Raises ``KeyError`` if unknown."""
    key = normalize_insight_provider(name)
    strategy = _INSIGHT_LLM_STRATEGIES[key]
    return strategy.resolve(env)


def resolve_explicit_model_id(model: str) -> InsightLlmResolution:
    """Infer provider from a legacy / explicit model string (no ``--llm`` shortcut)."""
    s = model.strip()
    if not s:
        return InsightLlmResolution(
            provider="openai",
            model_id=_DEFAULT_OPENAI_MODEL,
        )
    lower = s.lower()
    if lower.startswith("gemini/") or lower.startswith("google/"):
        mid = s.split("/", 1)[1] if "/" in s else s
        return InsightLlmResolution(provider="gemini", model_id=mid)
    if lower.startswith("anthropic/"):
        return InsightLlmResolution(provider="claude", model_id=s.split("/", 1)[1])
    if lower.startswith("ollama/"):
        return InsightLlmResolution(
            provider="ollama",
            model_id=s.split("/", 1)[1],
            extra={"base_url": _DEFAULT_OLLAMA_BASE_URL},
        )
    if lower.startswith("claude-"):
        return InsightLlmResolution(provider="claude", model_id=s)
    return InsightLlmResolution(provider="openai", model_id=s)


def expected_api_key_env_name(canonical_provider: str) -> str | None:
    """Env var that should be set for API access, or ``None`` for local Ollama."""
    return _PROVIDER_API_KEY_ENV.get(normalize_insight_provider(canonical_provider))


def register_insight_llm_strategy(
    canonical_name: str,
    strategy: InsightLlmProviderStrategy,
) -> None:
    """Register or replace a strategy (mainly for tests / extension)."""
    key = canonical_name.strip().lower()
    _INSIGHT_LLM_STRATEGIES[key] = strategy
    if key not in _PROVIDER_API_KEY_ENV:
        _PROVIDER_API_KEY_ENV[key] = None
