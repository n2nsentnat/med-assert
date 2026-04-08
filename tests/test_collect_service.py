"""Application layer tests."""

from article_miner.application.collect.service import CollectArticlesService
from article_miner.domain.collect.models import Article, CollectionOutput


class FakeGateway:
    def search_pmids(self, query: str, max_results: int) -> tuple[int, list[str]]:
        return 100, ["1", "2"]

    def fetch_articles(self, pmids: list[str]) -> tuple[list[Article], list[str]]:
        return (
            [
                Article(pmid="1", title="A"),
                Article(pmid="2", title="B"),
            ],
            [],
        )


def test_collect_service_builds_output() -> None:
    svc = CollectArticlesService(FakeGateway())
    out = svc.run("q", 10)
    assert out.query == "q"
    assert out.total_match_count == 100
    assert out.requested_count == 10
    assert out.retrieved_count == 2
    assert len(out.articles) == 2
