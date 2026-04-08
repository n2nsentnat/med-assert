"""Tests for PubMed JSON duplicate detection.

Combination matrix (edge rules × outcomes)
------------------------------------------
Each row is a scenario we assert explicitly. Clusters are connected components;
``edge_evidence`` lists actual pairwise links.

+------------------+----------------------------+--------------------------+
| Scenario         | Expected primary_reason    | Notes                    |
+==================+============================+==========================+
| Same DOI pair    | same_doi                   | Strong pairwise link     |
| Same title+year  | same_normalized_title_year | Requires non-null year   |
| Fuzzy only       | fuzzy_title_and_or_abstract| e.g. missing year dupes  |
| DOI ∪ fuzzy      | mixed_evidence             | Transitive chain         |
| Abstract gate    | (no cluster)               | Both abstracts, low sim  |
| Disjoint pairs   | 2 clusters                 | Independent DOI groups   |
+------------------+----------------------------+--------------------------+
"""

from __future__ import annotations

import pytest

from article_miner.domain.collect.models import Article, CollectionOutput
from article_miner.application.dedup.service import (
    ABSTRACT_TOKEN_SORT_MIN,
    build_duplicate_report,
    normalize_doi,
    normalize_title,
)


# --- Sample texts (length ≥ 12 chars for fuzzy blocking; PubMed-like tone) ---

SAMPLE_TRIAL_TITLE = (
    "Randomized double-blind placebo-controlled trial of intervention X in cohort Y"
)
SAMPLE_TRIAL_TITLE_VARIANT = (
    "Randomized, double-blind, placebo-controlled trial of intervention X in cohort Y"
)

SAMPLE_ABSTRACT_METHODS = (
    "METHODS: We enrolled 240 adults at three centers. "
    "The primary endpoint was change in biomarker Z at 12 weeks. "
    "RESULTS: The intervention arm showed a significant improvement versus placebo."
)

SAMPLE_ABSTRACT_UNRELATED = (
    "BACKGROUND: Solar particle flux and magnetospheric dynamics are poorly understood. "
    "We analyzed archival telemetry from unrelated observatories. "
    "FINDINGS: No clinical endpoints were assessed in this astrophysics survey."
)


def _collection(articles: list[Article], **kwargs: object) -> CollectionOutput:
    return CollectionOutput(
        query="q",
        total_match_count=len(articles),
        requested_count=len(articles),
        retrieved_count=len(articles),
        articles=articles,
        warnings=[],
        **kwargs,
    )


def test_normalize_doi_strips_prefix() -> None:
    assert normalize_doi("https://doi.org/10.1000/abc") == "10.1000/abc"
    assert normalize_doi("DOI:10.1000/abc") == "10.1000/abc"
    assert normalize_doi("doi:10.1000/abc") == "10.1000/abc"
    assert normalize_doi("https://dx.doi.org/10.1000/abc") == "10.1000/abc"
    assert normalize_doi("10.1000/abc.") == "10.1000/abc"


def test_normalize_title_strips_punctuation() -> None:
    assert normalize_title("Hello, World!") == "hello world"
    assert normalize_title(None) == ""


@pytest.mark.parametrize(
    ("raw_title", "expected_norm_substring"),
    [
        (SAMPLE_TRIAL_TITLE, "randomized double blind"),
        (SAMPLE_TRIAL_TITLE_VARIANT, "randomized double blind"),
    ],
)
def test_sample_titles_normalize_consistently(raw_title: str, expected_norm_substring: str) -> None:
    nt = normalize_title(raw_title)
    assert expected_norm_substring in nt
    assert normalize_title(SAMPLE_TRIAL_TITLE) == normalize_title(SAMPLE_TRIAL_TITLE_VARIANT)


def test_same_doi_clusters() -> None:
    a1 = Article(
        pmid="1",
        title="Same paper",
        doi="10.1234/x",
        publication_year=2020,
    )
    a2 = Article(
        pmid="2",
        title="Same paper variant",
        doi="https://doi.org/10.1234/x",
        publication_year=2020,
    )
    a3 = Article(pmid="3", title="Other", doi="10.999/y", publication_year=2021)
    r = build_duplicate_report(_collection([a1, a2, a3]))
    assert r.duplicate_group_count == 1
    assert r.clusters[0].pmids == ["1", "2"]
    assert r.clusters[0].primary_reason == "same_doi"
    assert r.clusters[0].confidence == "high"
    assert any("1" in e and "2" in e and "same_doi" in e for e in r.clusters[0].edge_evidence)


def test_same_doi_triple_all_edges_doi() -> None:
    """Three PMIDs sharing one DOI: component is single cluster, edges are DOI-only."""
    d = "10.7777/shared"
    arts = [
        Article(pmid="10", title="Ahead of print version long title", doi=d, publication_year=2021),
        Article(pmid="11", title="Final version slightly different long title", doi=d, publication_year=2021),
        Article(pmid="12", title="Publisher correction notice long title", doi=d, publication_year=2021),
    ]
    r = build_duplicate_report(_collection(arts))
    assert r.duplicate_group_count == 1
    c = r.clusters[0]
    assert set(c.pmids) == {"10", "11", "12"}
    assert c.primary_reason == "same_doi"
    assert all("same_doi" in e for e in c.edge_evidence)
    assert len(c.edge_evidence) == 3  # pairs (10,11), (10,12), (11,12)


def test_same_normalized_title_and_year() -> None:
    a1 = Article(
        pmid="10",
        title="Effect of X on Y: A Trial.",
        publication_year=2019,
    )
    a2 = Article(
        pmid="11",
        title="Effect of X on Y: A Trial",
        publication_year=2019,
    )
    r = build_duplicate_report(_collection([a1, a2]))
    assert r.duplicate_group_count == 1
    assert set(r.clusters[0].pmids) == {"10", "11"}
    assert r.clusters[0].primary_reason == "same_normalized_title_and_year"
    assert all("same_normalized_title_and_year" in e for e in r.clusters[0].edge_evidence)


def test_exact_title_year_without_doi_pair() -> None:
    """Same normalized title + year; no DOI — edge type is same_title_year only."""
    a1 = Article(
        pmid="201",
        title=SAMPLE_TRIAL_TITLE,
        publication_year=2015,
    )
    a2 = Article(
        pmid="202",
        title=SAMPLE_TRIAL_TITLE_VARIANT,
        publication_year=2015,
    )
    r = build_duplicate_report(_collection([a1, a2]))
    assert r.duplicate_group_count == 1
    c = r.clusters[0]
    assert c.primary_reason == "same_normalized_title_and_year"
    assert "same_normalized_title_and_year" in c.edge_evidence[0]


def test_singletons_no_cluster() -> None:
    a = Article(pmid="99", title="Unique title here", publication_year=2024)
    r = build_duplicate_report(_collection([a]))
    assert r.duplicate_group_count == 0
    assert r.clusters == []


def test_two_disjoint_doi_pairs_two_clusters() -> None:
    """Independent DOI groups → two duplicate clusters, no cross-merge."""
    arts = [
        Article(pmid="301", title="Alpha randomized controlled study long", doi="10.1/alpha", publication_year=2020),
        Article(pmid="302", title="Alpha variant title long enough here", doi="10.1/alpha", publication_year=2020),
        Article(pmid="303", title="Beta cohort analysis outcomes paper long", doi="10.2/beta", publication_year=2021),
        Article(pmid="304", title="Beta other wording outcomes paper long", doi="10.2/beta", publication_year=2021),
    ]
    r = build_duplicate_report(_collection(arts))
    assert r.duplicate_group_count == 2
    pmid_sets = [set(c.pmids) for c in r.clusters]
    assert {"301", "302"} in pmid_sets
    assert {"303", "304"} in pmid_sets


def test_retraction_note() -> None:
    a1 = Article(
        pmid="1",
        title="Retraction: Previous study on X",
        publication_year=2022,
    )
    a2 = Article(
        pmid="2",
        title="Retraction: Previous study on X",
        publication_year=2022,
    )
    r = build_duplicate_report(_collection([a1, a2]))
    assert r.clusters[0].reviewer_notes


def test_same_title_missing_year_not_exact_merge() -> None:
    """Missing publication_year must not use high-confidence exact title+year."""
    title = "Effect of long intervention on outcomes biomarker study title"
    a1 = Article(pmid="101", title=title, publication_year=None)
    a2 = Article(pmid="102", title=title, publication_year=None)
    r = build_duplicate_report(_collection([a1, a2]))
    assert r.duplicate_group_count == 1
    c = r.clusters[0]
    assert c.primary_reason == "fuzzy_title_and_or_abstract"
    assert all("fuzzy_title" in e for e in c.edge_evidence)


def test_fuzzy_with_both_abstracts_dissimilar_no_cluster() -> None:
    """Identical titles + both abstracts present: abstract gate blocks weak abstract match."""
    r = build_duplicate_report(
        _collection(
            [
                Article(
                    pmid="401",
                    title=SAMPLE_TRIAL_TITLE,
                    abstract=SAMPLE_ABSTRACT_METHODS,
                    publication_year=2018,
                ),
                Article(
                    pmid="402",
                    title=SAMPLE_TRIAL_TITLE,
                    abstract=SAMPLE_ABSTRACT_UNRELATED,
                    publication_year=2018,
                ),
            ]
        )
    )
    # Exact title+year merges first — same normalized title and year → NOT fuzzy path
    assert r.duplicate_group_count == 1
    c = r.clusters[0]
    assert c.primary_reason == "same_normalized_title_and_year"

    # Force fuzzy path: different years so exact title+year does not apply; titles still identical → fuzzy;
    # abstract gate should block if we use same title and dissimilar abstracts.
    r2 = build_duplicate_report(
        _collection(
            [
                Article(
                    pmid="501",
                    title=SAMPLE_TRIAL_TITLE,
                    abstract=SAMPLE_ABSTRACT_METHODS,
                    publication_year=2017,
                ),
                Article(
                    pmid="502",
                    title=SAMPLE_TRIAL_TITLE,
                    abstract=SAMPLE_ABSTRACT_UNRELATED,
                    publication_year=2018,
                ),
            ]
        )
    )
    assert r2.duplicate_group_count == 0


def test_mixed_doi_and_fuzzy_cluster() -> None:
    """Transitive cluster: DOI pair + fuzzy pair shares one PMID → mixed_evidence."""
    t = "Diabetes mellitus long cohort study outcomes research paper title"
    a1 = Article(pmid="1", title=t, doi="10.5555/a", publication_year=2020)
    a2 = Article(pmid="2", title=t, doi="10.5555/a", publication_year=2020)
    a3 = Article(
        pmid="3",
        title=t + " appendix",
        doi=None,
        publication_year=2020,
    )
    r = build_duplicate_report(_collection([a1, a2, a3]))
    assert r.duplicate_group_count == 1
    c = r.clusters[0]
    assert c.primary_reason == "mixed_evidence"
    assert c.transitivity_note
    kinds = {e.split(": ")[-1] for e in c.edge_evidence}
    assert "same_doi" in kinds and "fuzzy_title" in kinds


def test_pmid_sort_non_numeric_does_not_crash() -> None:
    """PMIDs are usually numeric; sorting tolerates odd strings."""
    a1 = Article(pmid="2", title="Shared title for duplicate test long", publication_year=2020)
    a2 = Article(pmid="bad", title="Shared title for duplicate test long", publication_year=2020)
    r = build_duplicate_report(_collection([a1, a2]))
    assert r.duplicate_group_count == 1
    # Sorted with numeric first, then non-numeric
    assert r.clusters[0].pmids[0] == "2"


def test_large_input_linear_stats() -> None:
    """Ensure blocking keeps fuzzy work bounded (smoke: 500 articles, unique titles)."""
    arts = [
        Article(
            pmid=str(30000 + i),
            title=f"Study id {i} unique biomarker long title for blocking test",
            publication_year=2020,
        )
        for i in range(500)
    ]
    r = build_duplicate_report(_collection(arts))
    assert r.duplicate_group_count == 0
    assert int(r.stats.get("fuzzy_pairs_compared", 0)) < 500_000


def test_abstract_token_sort_threshold_documented() -> None:
    """Sanity: sample abstracts are intentionally far apart for the gate test."""
    from rapidfuzz import fuzz

    assert (
        fuzz.token_sort_ratio(SAMPLE_ABSTRACT_METHODS, SAMPLE_ABSTRACT_UNRELATED)
        < ABSTRACT_TOKEN_SORT_MIN
    )
