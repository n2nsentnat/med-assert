"""Optional live tests against NCBI E-utilities (network, not run by default)."""

from __future__ import annotations

import os

import pytest

from article_miner.application.collect.service import CollectArticlesService
from article_miner.infrastructure.collect.config import NcbiClientConfig
from article_miner.infrastructure.collect.pubmed_gateway import EntrezPubMedGateway
from article_miner.infrastructure.collect.rate_limiter import RateLimiter
from article_miner.infrastructure.collect.resilient_http import ResilientHttpClient


def _live_enabled() -> bool:
    return os.environ.get("ARTICLE_MINER_LIVE_NCBI", "").strip() == "1"


skip_if_no_live = pytest.mark.skipif(
    not _live_enabled(),
    reason="Set ARTICLE_MINER_LIVE_NCBI=1 to run live NCBI tests (requires network).",
)


@skip_if_no_live
@pytest.mark.live_ncbi
@pytest.mark.parametrize(
    "query",
    [
        "diabetes mellitus[tiab]",
        "hypertension[tiab]",
        "COVID-19[tiab]",
        "machine learning[tiab]",
        "randomized controlled trial[pt]",
    ],
)
def test_live_collect_articles_end_to_end(query: str) -> None:
    """Same code path as the CLI: search + batched efetch, real PubMed."""
    config = NcbiClientConfig()
    limiter = RateLimiter(config.requests_per_second)
    http = ResilientHttpClient(config, limiter)
    try:
        gateway = EntrezPubMedGateway(http, config)
        service = CollectArticlesService(gateway)
        result = service.run(query=query, requested_count=2)
    finally:
        http.close()

    assert result.query == query
    assert result.total_match_count >= 1
    assert result.requested_count == 2
    assert result.retrieved_count == 2
    assert len(result.articles) == 2
    assert not result.warnings
    for article in result.articles:
        assert article.pmid.isdigit()
        assert len(article.pmid) >= 1
        assert article.title is not None or article.abstract is not None
