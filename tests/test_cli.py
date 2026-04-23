"""CLI smoke tests (mocked service)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from med_assert.application.collect.service import CollectArticlesService
from med_assert.interfaces.cli import app
from med_assert.domain.collect.models import Article, CollectionOutput


def test_cli_writes_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

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
    out = tmp_path / "out.json"
    runner = CliRunner()
    result = runner.invoke(app, ["my query", "-o", str(out), "-n", "5"])
    assert result.exit_code == 0, result.output
    text = out.read_text(encoding="utf-8")
    assert '"query": "my query"' in text
    assert '"title": "Hello"' in text
