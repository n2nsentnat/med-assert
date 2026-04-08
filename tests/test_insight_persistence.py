"""Tests for incremental per-article persistence."""

from __future__ import annotations

import asyncio

from article_miner.application.insights.job import InsightJobConfig, run_insight_job
from article_miner.domain.collect.models import Article, CollectionOutput


def test_incremental_jsonl_persists_each_article(tmp_path) -> None:
    inc_path = tmp_path / "insights.incremental.jsonl"
    collection = CollectionOutput(
        query="q",
        total_match_count=2,
        requested_count=2,
        retrieved_count=2,
        articles=[
            Article(pmid="1", title="No abstract article", abstract=None, publication_year=2020),
            Article(pmid="2", title="Short abstract article", abstract="short", publication_year=2020),
        ],
        warnings=[],
    )
    cfg = InsightJobConfig(
        model="gpt-4o-mini",
        enable_audit=False,
        incremental_jsonl_path=inc_path,
    )
    out = asyncio.run(run_insight_job(collection, cfg))
    assert len(out.articles) == 2
    lines = inc_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        assert '"status":' in line
        assert '"raw_llm_text":' in line

