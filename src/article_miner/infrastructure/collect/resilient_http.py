"""HTTP GET with rate limiting, retries, and structured errors."""

from __future__ import annotations

import logging
import random
import time
from typing import Any

import httpx

from article_miner.domain.errors import NcbiRateLimitError, NcbiTransportError
from article_miner.infrastructure.collect.config import NcbiClientConfig
from article_miner.infrastructure.collect.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


def _redact_params(params: dict[str, Any]) -> dict[str, Any]:
    """Omit secrets and trim very long values for safe logging."""
    redacted: dict[str, Any] = {}
    for key, value in params.items():
        if key == "api_key":
            redacted[key] = "***" if value else value
        elif key == "term" and isinstance(value, str) and len(value) > 120:
            redacted[key] = f"{value[:120]}..."
        else:
            redacted[key] = value
    return redacted


class ResilientHttpClient:
    """Thin wrapper: rate limit + exponential backoff retries."""

    def __init__(
        self,
        config: NcbiClientConfig,
        rate_limiter: RateLimiter,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self._config = config
        self._rate = rate_limiter
        self._client = client or httpx.Client(timeout=config.timeout_seconds)

    def close(self) -> None:
        self._client.close()

    def get_text(self, url: str, params: dict[str, Any] | None = None) -> str:
        """GET and return response body as text (raises domain errors on failure)."""
        query = params if params is not None else {}
        last_exc: Exception | None = None
        for attempt in range(self._config.max_retries + 1):
            self._rate.acquire()
            try:
                response = self._client.get(url, params=query)
            except httpx.RequestError as exc:
                last_exc = exc
                if attempt >= self._config.max_retries:
                    logger.warning(
                        "Request failed after %s attempts: url=%s params=%s error=%s",
                        self._config.max_retries + 1,
                        url,
                        _redact_params(query),
                        exc,
                    )
                    raise NcbiTransportError(f"HTTP request failed after retries: {exc}") from exc
                logger.debug(
                    "Retry %s/%s after RequestError %s: %s",
                    attempt + 1,
                    self._config.max_retries + 1,
                    type(exc).__name__,
                    exc,
                )
                self._backoff(attempt)
                continue

            if response.status_code == 429:
                if attempt >= self._config.max_retries:
                    logger.warning(
                        "HTTP 429 from NCBI after %s attempts; url=%s params=%s",
                        self._config.max_retries + 1,
                        url,
                        _redact_params(query),
                    )
                    raise NcbiRateLimitError("NCBI returned HTTP 429 (rate limited).")
                logger.debug(
                    "Retry %s/%s after HTTP 429 (Retry-After=%r)",
                    attempt + 1,
                    self._config.max_retries + 1,
                    response.headers.get("Retry-After"),
                )
                self._backoff(attempt, extra=response.headers.get("Retry-After"))
                continue

            if response.status_code in (500, 502, 503, 504):
                if attempt >= self._config.max_retries:
                    logger.warning(
                        "HTTP %s from NCBI after %s attempts: url=%s params=%s body_prefix=%r",
                        response.status_code,
                        self._config.max_retries + 1,
                        url,
                        _redact_params(query),
                        response.text[:200],
                    )
                    raise NcbiTransportError(
                        f"NCBI server error HTTP {response.status_code}: {response.text[:500]}"
                    )
                logger.debug(
                    "Retry %s/%s after HTTP %s",
                    attempt + 1,
                    self._config.max_retries + 1,
                    response.status_code,
                )
                self._backoff(attempt)
                continue

            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                logger.warning(
                    "HTTP %s (no retry): url=%s params=%s body_prefix=%r",
                    response.status_code,
                    url,
                    _redact_params(query),
                    response.text[:200],
                )
                raise NcbiTransportError(
                    f"NCBI HTTP {response.status_code}: {response.text[:500]}"
                ) from exc

            return response.text

        logger.warning(
            "Exhausted retry loop without success: url=%s params=%s last_exc=%r",
            url,
            _redact_params(query),
            last_exc,
        )
        raise NcbiTransportError(f"HTTP request failed: {last_exc!r}")

    def _backoff(self, attempt: int, *, extra: str | None = None) -> None:
        base = self._config.base_backoff_seconds * (2**attempt)
        jitter = random.uniform(0, 0.25 * base)
        delay = base + jitter
        if extra:
            try:
                delay = max(delay, float(extra))
            except ValueError:
                pass
        cap = self._config.max_backoff_seconds
        if cap is not None and delay > cap:
            logger.debug(
                "Backoff: capping sleep from %.3fs to %.3fs (max_backoff_seconds)",
                delay,
                cap,
            )
            delay = cap
        logger.debug(
            "Backoff: sleeping %.3fs before retry (attempt %s/%s, base_backoff=%ss)",
            delay,
            attempt + 1,
            self._config.max_retries + 1,
            self._config.base_backoff_seconds,
        )
        time.sleep(delay)

