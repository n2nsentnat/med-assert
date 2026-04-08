"""Tests for EntrezPubMedGateway (mocked HTTP)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from article_miner.domain.errors import MalformedResponseError
from article_miner.infrastructure.collect.config import ESEARCH_URL, EFETCH_URL, NcbiClientConfig
from article_miner.infrastructure.collect.pubmed_gateway import EntrezPubMedGateway


class FakeHttp:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get_text(self, url: str, params: dict[str, Any] | None = None) -> str:
        self.calls.append((url, dict(params) if params is not None else {}))
        if not self.responses:
            msg = "no more mocked responses"
            raise AssertionError(msg)
        return self.responses.pop(0)


def test_search_respects_total_and_max() -> None:
    body = json.dumps(
        {"esearchresult": {"count": "10", "idlist": ["1", "2", "3"]}}
    )
    http = FakeHttp([body])
    gw = EntrezPubMedGateway(http, NcbiClientConfig())
    total, ids = gw.search_pmids("cancer", 5)
    assert total == 10
    assert ids == ["1", "2", "3"]
    assert http.calls[0][0] == ESEARCH_URL


def test_search_paginates_beyond_one_page() -> None:
    first = json.dumps(
        {
            "esearchresult": {
                "count": "50000",
                "idlist": [str(i) for i in range(10000)],
            }
        }
    )
    second = json.dumps(
        {
            "esearchresult": {
                "count": "50000",
                "idlist": [str(i) for i in range(10000, 15000)],
            }
        }
    )
    http = FakeHttp([first, second])
    gw = EntrezPubMedGateway(http, NcbiClientConfig())
    total, ids = gw.search_pmids("q", 15000)
    assert total == 50000
    assert len(ids) == 15000
    assert ids[0] == "0"
    assert ids[-1] == "14999"
    assert http.calls[0][1]["retstart"] in (0, "0")
    assert http.calls[1][1]["retstart"] in (10000, "10000")


def test_malformed_esearch_json() -> None:
    http = FakeHttp(["{not json"])
    gw = EntrezPubMedGateway(http, NcbiClientConfig())
    with pytest.raises(MalformedResponseError):
        gw.search_pmids("x", 1)


def test_efetch_batches_and_orders(minimal_pubmed_xml: str) -> None:
    es = json.dumps({"esearchresult": {"count": "1", "idlist": ["99999999"]}})
    http = FakeHttp([es, minimal_pubmed_xml])
    gw = EntrezPubMedGateway(http, NcbiClientConfig())
    _, pmids = gw.search_pmids("x", 1)
    arts, warns = gw.fetch_articles(pmids)
    assert not warns
    assert len(arts) == 1
    assert arts[0].pmid == "99999999"
    assert http.calls[1][0] == EFETCH_URL
    assert "99999999" in http.calls[1][1]["id"]


def test_efetch_error_tag_raises() -> None:
    http = FakeHttp(['<?xml version="1.0"?><ERROR>bad</ERROR>'])
    gw = EntrezPubMedGateway(http, NcbiClientConfig())
    with pytest.raises(MalformedResponseError, match="EFetch returned ERROR"):
        gw.fetch_articles(["1"])


def test_efetch_multiple_error_tags_joined() -> None:
    xml = '<?xml version="1.0"?><ERROR>first</ERROR><ERROR>second</ERROR>'
    http = FakeHttp([xml])
    gw = EntrezPubMedGateway(http, NcbiClientConfig())
    with pytest.raises(MalformedResponseError) as exc_info:
        gw.fetch_articles(["1"])
    assert "first" in str(exc_info.value)
    assert "second" in str(exc_info.value)
