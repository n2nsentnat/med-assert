"""Resilient HTTP client behavior (mocked transport)."""

from unittest.mock import patch

import httpx
import pytest

from article_miner.domain.errors import NcbiTransportError
from article_miner.infrastructure.collect.config import NcbiClientConfig
from article_miner.infrastructure.collect.rate_limiter import RateLimiter
from article_miner.infrastructure.collect.resilient_http import (
    ResilientHttpClient,
    _redact_params,
)


def test_redact_params_masks_api_key_and_truncates_term() -> None:
    long_term = "x" * 200
    out = _redact_params(
        {"db": "pubmed", "api_key": "secret", "term": long_term, "retmode": "json"}
    )
    assert out["api_key"] == "***"
    assert out["db"] == "pubmed"
    assert out["retmode"] == "json"
    assert len(out["term"]) < len(long_term)
    assert str(out["term"]).endswith("...")


def test_retries_then_succeeds() -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(503, text="unavailable")
        return httpx.Response(200, text="ok")

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, timeout=5.0)
    config = NcbiClientConfig(max_retries=5, base_backoff_seconds=0.01)
    http = ResilientHttpClient(config, RateLimiter(1000.0), client=client)
    try:
        assert http.get_text("https://example.test/x", {"a": "1"}) == "ok"
    finally:
        http.close()
    assert attempts["n"] == 3


def test_backoff_caps_at_max_backoff_seconds() -> None:
    config = NcbiClientConfig(
        base_backoff_seconds=1000.0,
        max_backoff_seconds=2.0,
    )
    http = ResilientHttpClient(config, RateLimiter(1000.0), client=httpx.Client())
    try:
        with patch("article_miner.infrastructure.collect.resilient_http.time.sleep") as mock_sleep:
            http._backoff(0)
        mock_sleep.assert_called_once()
        assert mock_sleep.call_args[0][0] == 2.0
    finally:
        http.close()


def test_get_text_omitted_params() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="ok")

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, timeout=5.0)
    config = NcbiClientConfig()
    http = ResilientHttpClient(config, RateLimiter(1000.0), client=client)
    try:
        assert http.get_text("https://example.test/x") == "ok"
    finally:
        http.close()


def test_exhausts_retries_on_503() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="no")

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, timeout=5.0)
    config = NcbiClientConfig(max_retries=2, base_backoff_seconds=0.01)
    http = ResilientHttpClient(config, RateLimiter(1000.0), client=client)
    try:
        with pytest.raises(NcbiTransportError):
            http.get_text("https://example.test/x", {})
    finally:
        http.close()


def test_raises_on_400_without_retry() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="bad request")

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, timeout=5.0)
    config = NcbiClientConfig(max_retries=2)
    http = ResilientHttpClient(config, RateLimiter(1000.0), client=client)
    try:
        with pytest.raises(NcbiTransportError):
            http.get_text("https://example.test/x", {})
    finally:
        http.close()
