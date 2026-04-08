"""Use case: collect up to N articles for a query and build validated output."""

from __future__ import annotations

from article_miner.application.collect.ports import PubMedGateway
from article_miner.domain.collect.models import CollectionOutput


class CollectArticlesService:
    """Application service orchestrating search + batched fetch."""

    def __init__(self, gateway: PubMedGateway) -> None:
        self._gateway = gateway

    def run(self, query: str, requested_count: int) -> CollectionOutput:
        if requested_count < 1:
            msg = "requested_count must be at least 1"
            raise ValueError(msg)

        total_match_count, pmids = self._gateway.search_pmids(query, requested_count)
        articles, fetch_warnings = self._gateway.fetch_articles(pmids)

        return CollectionOutput(
            query=query,
            total_match_count=total_match_count,
            requested_count=requested_count,
            retrieved_count=len(articles),
            articles=articles,
            warnings=fetch_warnings,
        )

