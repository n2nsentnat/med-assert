"""Application ports used by collect use case."""

from __future__ import annotations

from typing import Protocol

from article_miner.domain.collect.models import Article


class PubMedGateway(Protocol):
    def search_pmids(self, query: str, max_results: int) -> tuple[int, list[str]]: ...

    def fetch_articles(self, pmids: list[str]) -> tuple[list[Article], list[str]]: ...

