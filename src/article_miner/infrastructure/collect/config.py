"""NCBI E-utilities configuration (infrastructure)."""

from __future__ import annotations

from dataclasses import dataclass

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# ESearch returns at most 10,000 IDs per request (NCBI documentation).
ESEARCH_PAGE_MAX = 10_000
# Recommended batch size for efetch id lists (stay well under server limits).
EFETCH_ID_BATCH_SIZE = 200

# NCBI: ~3 req/s without API key, ~10 req/s with key.
REQUESTS_PER_SECOND_NO_KEY = 3.0
REQUESTS_PER_SECOND_WITH_KEY = 10.0


@dataclass(frozen=True, slots=True)
class NcbiClientConfig:
    """Runtime settings for HTTP client + etiquette parameters."""

    api_key: str | None = None
    tool: str = "article_miner"
    email: str | None = None
    timeout_seconds: float = 60.0
    max_retries: int = 5
    base_backoff_seconds: float = 0.5
    #: Upper bound on a single backoff sleep (after jitter / Retry-After). ``None`` = no cap.
    max_backoff_seconds: float | None = 120.0

    @property
    def requests_per_second(self) -> float:
        return REQUESTS_PER_SECOND_WITH_KEY if self.api_key else REQUESTS_PER_SECOND_NO_KEY

