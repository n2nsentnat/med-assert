"""LangGraph placeholder smoke test."""

from __future__ import annotations

import asyncio

from article_miner.application.insights.insight_langgraph import (
    compile_insight_placeholder_graph,
)


def test_placeholder_graph_runs() -> None:
    g = compile_insight_placeholder_graph()
    out = asyncio.run(g.ainvoke({"pmid": "123", "stage": ""}))
    assert out.get("stage") == "ready"
