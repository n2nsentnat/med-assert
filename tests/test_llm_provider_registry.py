"""Tests for insight LLM provider strategy registry."""

from __future__ import annotations

import pytest

from article_miner.application.insights.llm_provider_registry import (
    expected_api_key_env_name,
    normalize_insight_provider,
    registered_insight_providers,
    resolve_insight_llm_provider,
)


def test_normalize_anthropic_alias() -> None:
    assert normalize_insight_provider("Anthropic ") == "claude"


def test_resolve_openai_uses_env_and_default() -> None:
    r = resolve_insight_llm_provider("openai", {})
    assert r.model_id == "gpt-4o-mini"
    assert r.extra == {}

    r2 = resolve_insight_llm_provider("openai", {"INSIGHT_MODEL_OPENAI": "gpt-4o"})
    assert r2.model_id == "gpt-4o"


def test_resolve_ollama_prefix_and_base() -> None:
    r = resolve_insight_llm_provider("ollama", {})
    assert r.model_id == "gemma3:4b"
    assert r.extra["base_url"] == "http://localhost:11434"

    r2 = resolve_insight_llm_provider(
        "ollama",
        {"OLLAMA_MODEL": "mistral:7b", "OLLAMA_BASE_URL": "http://127.0.0.1:11434"},
    )
    assert r2.model_id == "mistral:7b"
    assert r2.extra["base_url"] == "http://127.0.0.1:11434"


def test_resolve_gemini_claude_defaults() -> None:
    g = resolve_insight_llm_provider("gemini", {})
    assert "gemini" in g.model_id.lower()

    c = resolve_insight_llm_provider("claude", {})
    assert c.model_id.startswith("claude")


def test_resolve_unknown_raises() -> None:
    with pytest.raises(KeyError):
        resolve_insight_llm_provider("unknown-vendor", {})


def test_expected_api_key_env_name() -> None:
    assert expected_api_key_env_name("openai") == "OPENAI_API_KEY"
    assert expected_api_key_env_name("ollama") is None


def test_registered_sorted() -> None:
    assert "openai" in registered_insight_providers()
    assert list(registered_insight_providers()) == sorted(
        registered_insight_providers()
    )
