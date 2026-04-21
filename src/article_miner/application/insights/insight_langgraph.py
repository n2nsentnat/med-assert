"""LangGraph scaffolding for future multi-step insight orchestration.

Production classification uses :class:`~article_miner.application.insight_job.InsightClassificationJob`
with LangChain chat models. This module compiles a minimal ``StateGraph`` so LangGraph stays
part of the dependency closure and can be extended (e.g. explicit extract → validate → audit nodes).
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class InsightGraphState(TypedDict, total=False):
    """Placeholder state for graph expansion."""

    pmid: str
    stage: str


async def _mark_ready(state: InsightGraphState) -> InsightGraphState:
    return {"pmid": state.get("pmid", ""), "stage": "ready"}


def compile_insight_placeholder_graph():
    """Return a tiny compiled graph (no LLM calls). Useful for wiring tests and future nodes."""
    g = StateGraph(InsightGraphState)
    g.add_node("init", _mark_ready)
    g.add_edge(START, "init")
    g.add_edge("init", END)
    return g.compile()
