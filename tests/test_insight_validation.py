"""Tests for insight Pass 2 validation (grounding + semantics)."""

from __future__ import annotations

from article_miner.domain.collect.models import Article
from article_miner.domain.insights.models import ClinicalFieldInsight, FieldInsightBlock, LlmInsightExtraction
from article_miner.infrastructure.insights.canonical_text import build_canonical_text, span_in_haystack
from article_miner.infrastructure.insights.insight_validation import (
    grounding_checks,
    parse_extraction_json,
    run_pass2_validation,
)
from article_miner.infrastructure.insights.semantic_rules import run_semantic_rules


def _article() -> Article:
    return Article(
        pmid="1",
        title="Trial of drug X for blood pressure reduction in adults",
        abstract=(
            "METHODS: Randomized trial, n=200. RESULTS: No significant difference in primary endpoint "
            "(p=0.12). Secondary analysis showed a trend favoring X."
        ),
        publication_year=2020,
    )


def _valid_extraction() -> LlmInsightExtraction:
    return LlmInsightExtraction(
        pmid="1",
        finding_direction=FieldInsightBlock(
            value="neutral",
            confidence=0.8,
            evidence_spans=["No significant difference"],
        ),
        statistical_significance=FieldInsightBlock(
            value="not_significant",
            confidence=0.85,
            evidence_spans=["No significant difference"],
        ),
        clinical_meaningfulness=ClinicalFieldInsight(
            value="unclear",
            confidence=0.6,
            evidence_spans=["n=200"],
            reasoning_summary="",
        ),
        main_claim=FieldInsightBlock(
            value="The trial did not show a significant primary endpoint difference.",
            confidence=0.75,
            evidence_spans=["No significant difference in primary endpoint"],
        ),
        review_flags=[],
    )


def test_build_canonical_text() -> None:
    a = _article()
    c = build_canonical_text(a)
    assert "Trial of drug" in c
    assert "METHODS" in c


def test_span_in_haystack() -> None:
    a = _article()
    h = build_canonical_text(a)
    assert span_in_haystack("Randomized trial", h, fuzzy_whitespace=True)


def test_grounding_checks_valid() -> None:
    a = _article()
    ext = _valid_extraction()
    checks = grounding_checks(a, ext)
    assert all(c.all_spans_found for c in checks)


def test_semantic_flag_significant_vs_negative_evidence() -> None:
    ext = _valid_extraction()
    ext.statistical_significance.value = "significant"
    flags = run_semantic_rules(ext)
    codes = [f.code for f in flags]
    assert "sig_vs_evidence_neg" in codes


def test_run_pass2() -> None:
    a = _article()
    ext = _valid_extraction()
    val, sem, auto, _ = run_pass2_validation(a, ext, confidence_threshold=0.3)
    assert val.schema_ok
    assert all(g.all_spans_found for g in val.grounding)


def test_parse_extraction_json_roundtrip() -> None:
    ext = _valid_extraction()
    raw = ext.model_dump_json()
    parsed, err = parse_extraction_json(raw)
    assert not err
    assert parsed is not None
    assert parsed.pmid == "1"


def test_prefilter_skips_short() -> None:
    from article_miner.infrastructure.insights.prefilter import prefilter_article

    a = Article(pmid="9", title="Hi", abstract="short", publication_year=2020)
    assert prefilter_article(a) == "short_abstract"
