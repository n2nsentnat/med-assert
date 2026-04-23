"""Tests for PubMed XML parsing."""

import logging

import pytest

from med_assert.domain.errors import MalformedResponseError
from med_assert.infrastructure.collect.pubmed_xml import (
    _parse_pubmed_month,
    parse_pubmed_article_element,
    parse_pubmed_xml_document,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Jan", 1),
        ("DEC", 12),
        ("03", 3),
        ("12", 12),
        ("March", 3),
        ("", None),
        (None, None),
        ("Spring", None),
        ("99", None),
    ],
)
def test_parse_pubmed_month(raw: str | None, expected: int | None) -> None:
    assert _parse_pubmed_month(raw) == expected


def test_parse_minimal_document(minimal_pubmed_xml: str) -> None:
    articles = parse_pubmed_xml_document(minimal_pubmed_xml)
    assert len(articles) == 1
    a = articles[0]
    assert a.pmid == "99999999"
    assert a.title == "Example title for unit tests."
    assert "BACKGROUND: Background text." in (a.abstract or "")
    assert "Second paragraph." in (a.abstract or "")
    assert a.journal_full == "Test Journal Full"
    assert a.journal_iso == "Test J"
    assert a.publication_year == 2024
    assert a.publication_month == 1
    assert a.publication_day == 15
    assert a.doi == "10.1000/example"
    assert a.keywords == ["k1"]
    assert len(a.authors) == 1
    assert a.authors[0].last_name == "Doe"
    assert a.authors[0].fore_name == "Jane"


def test_parse_invalid_xml_raises() -> None:
    with pytest.raises(MalformedResponseError):
        parse_pubmed_xml_document("not xml")


def test_parse_skipped_article_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Unparseable PubmedArticle is skipped and logged (PMID when present in XML)."""
    xml = """<?xml version="1.0" ?>
<PubmedArticleSet>
<PubmedArticle>
  <MedlineCitation Status="MEDLINE">
    <PMID Version="1"></PMID>
  </MedlineCitation>
</PubmedArticle>
<PubmedArticle>
  <MedlineCitation>
    <PMID Version="1">22222222</PMID>
    <Article>
      <ArticleTitle>Valid second article</ArticleTitle>
      <Language>eng</Language>
      <PublicationTypeList>
        <PublicationType>Journal Article</PublicationType>
      </PublicationTypeList>
    </Article>
  </MedlineCitation>
  <PubmedData>
    <ArticleIdList>
      <ArticleId IdType="pubmed">22222222</ArticleId>
    </ArticleIdList>
  </PubmedData>
</PubmedArticle>
</PubmedArticleSet>
"""
    with caplog.at_level(logging.WARNING):
        articles = parse_pubmed_xml_document(xml)
    assert len(articles) == 1
    assert articles[0].pmid == "22222222"
    assert "Skipping PubmedArticle" in caplog.text
    assert "Missing PMID" in caplog.text
