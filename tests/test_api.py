"""FastAPI smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from article_miner.application.collect.service import CollectArticlesService
from article_miner.domain.collect.models import Article, CollectionOutput
from article_miner.domain.insight import (
    InsightJobResult,
    PerArticleInsightResult,
    PerArticleStatus,
)
import article_miner.interfaces.api.http_app as api_app_module
from article_miner.interfaces.api.http_app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_collect_mocked(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(
        self: CollectArticlesService, query: str, requested_count: int
    ) -> CollectionOutput:
        return CollectionOutput(
            query=query,
            total_match_count=1,
            requested_count=requested_count,
            retrieved_count=1,
            articles=[Article(pmid="1", title="Hello")],
            warnings=[],
        )

    monkeypatch.setattr(CollectArticlesService, "run", fake_run)
    r = client.post(
        "/collect",
        json={"query": "cancer", "count": 3},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["query"] == "cancer"
    assert data["requested_count"] == 3
    assert data["articles"][0]["title"] == "Hello"


def _sample_collection_dict() -> dict:
    return {
        "query": "q",
        "total_match_count": 2,
        "requested_count": 2,
        "retrieved_count": 2,
        "articles": [
            {
                "pmid": "1",
                "title": "Same title",
                "doi": "10.1234/x",
                "publication_year": 2020,
            },
            {
                "pmid": "2",
                "title": "Same title",
                "doi": "10.1234/x",
                "publication_year": 2020,
            },
        ],
        "warnings": [],
    }


def test_dedup(client: TestClient, tmp_path: Path) -> None:
    coll = tmp_path / "collection.json"
    coll.write_text(json.dumps(_sample_collection_dict()), encoding="utf-8")
    body = {"collection_path": str(coll)}
    r = client.post("/dedup", json=body)
    assert r.status_code == 200, r.text
    rep = r.json()["report"]
    assert rep["source_article_count"] == 2
    assert r.json().get("markdown") is None

    body_md = {**body, "include_markdown": True}
    r2 = client.post("/dedup", json=body_md)
    assert r2.status_code == 200
    assert r2.json()["markdown"] is not None
    assert "Probable duplicate" in r2.json()["markdown"]


def test_dedup_collection_missing(client: TestClient, tmp_path: Path) -> None:
    missing = tmp_path / "nope.json"
    r = client.post("/dedup", json={"collection_path": str(missing)})
    assert r.status_code == 404
    assert "not found" in r.json()["detail"].lower()


def test_dedup_collection_invalid_json(client: TestClient, tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    r = client.post("/dedup", json={"collection_path": str(bad)})
    assert r.status_code == 422


def test_collect_file_mode(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_run(
        self: CollectArticlesService, query: str, requested_count: int
    ) -> CollectionOutput:
        return CollectionOutput(
            query=query,
            total_match_count=1,
            requested_count=requested_count,
            retrieved_count=1,
            articles=[Article(pmid="1", title="Hello")],
            warnings=[],
        )

    monkeypatch.setattr(CollectArticlesService, "run", fake_run)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out" / "c.json"
    r = client.post(
        "/collect",
        json={
            "query": "cancer",
            "count": 3,
            "output_format": "file",
            "output_path": str(out),
        },
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["output_format"] == "file"
    assert data["paths"]["collection_json"] == str(out.resolve())
    assert out.read_text(encoding="utf-8")[:50]


def test_dedup_file_mode(client: TestClient, tmp_path: Path) -> None:
    coll = tmp_path / "collection.json"
    coll.write_text(json.dumps(_sample_collection_dict()), encoding="utf-8")
    out = tmp_path / "d.json"
    body = {
        "collection_path": str(coll),
        "output_format": "file",
        "output_path": str(out),
        "include_markdown": True,
    }
    r = client.post("/dedup", json=body)
    assert r.status_code == 200, r.text
    assert r.json()["paths"]["report_json"] == str(out.resolve())
    md_path = out.with_suffix(".md")
    assert r.json()["paths"]["markdown"] == str(md_path.resolve())
    assert out.read_text(encoding="utf-8")
    assert "Probable duplicate" in md_path.read_text(encoding="utf-8")


def test_insights_mocked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def fake_job(*_a, **_k):
        return InsightJobResult(
            prompt_version="p",
            model="m",
            source_query="q",
            articles=[
                PerArticleInsightResult(
                    pmid="1",
                    status=PerArticleStatus.NEEDS_HUMAN_REVIEW,
                )
            ],
            stats={"needs_review": 1.0},
        )

    monkeypatch.setattr(api_app_module, "run_insight_job", fake_job)
    coll = tmp_path / "collection.json"
    coll.write_text(
        json.dumps(
            {
                "query": "q",
                "total_match_count": 1,
                "requested_count": 1,
                "retrieved_count": 1,
                "articles": [{"pmid": "1", "title": "T"}],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    with TestClient(app) as client:
        r = client.post(
            "/insights",
            json={"collection_path": str(coll), "model": "gpt-4o-mini"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["model"] == "m"
        assert r.json()["articles"][0]["pmid"] == "1"


def test_insights_collection_missing(client: TestClient, tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    r = client.post("/insights", json={"collection_path": str(missing)})
    assert r.status_code == 404
    assert "not found" in r.json()["detail"].lower()


def test_insights_file_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async def fake_job(*_a, **_k):
        return InsightJobResult(
            prompt_version="p",
            model="m",
            source_query="q",
            articles=[
                PerArticleInsightResult(
                    pmid="1",
                    status=PerArticleStatus.NEEDS_HUMAN_REVIEW,
                )
            ],
            stats={"needs_review": 1.0},
        )

    monkeypatch.setattr(api_app_module, "run_insight_job", fake_job)
    out = tmp_path / "ins.json"
    coll = tmp_path / "collection.json"
    coll.write_text(
        json.dumps(
            {
                "query": "q",
                "total_match_count": 1,
                "requested_count": 1,
                "retrieved_count": 1,
                "articles": [{"pmid": "1", "title": "T"}],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    with TestClient(app) as client:
        r = client.post(
            "/insights",
            json={
                "collection_path": str(coll),
                "model": "gpt-4o-mini",
                "output_format": "file",
                "output_path": str(out),
                "write_report_md": True,
            },
        )
    assert r.status_code == 200, r.text
    assert r.json()["output_format"] == "file"
    assert r.json()["paths"]["json"] == str(out.resolve())
    assert out.read_text(encoding="utf-8")
    md = out.with_name("insight_output_report.md")
    assert r.json()["paths"]["report_md"] == str(md.resolve())
    assert "# Insight Output Report" in md.read_text(encoding="utf-8")
