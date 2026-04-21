"""HTTP request bodies for the REST API (domain models stay in ``domain``)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from article_miner.application.dedup.service import DedupReport
from article_miner.application.insights.llm_provider_registry import (
    InsightLlmResolution,
    registered_insight_providers,
    resolve_explicit_model_id,
    resolve_insight_llm_provider,
)

OutputFormat = Literal["json", "file"]
InsightFileFormat = Literal["json", "jsonl"]

DEFAULT_API_OUTPUT_SUBDIR = Path("article_miner_output")


class FileWriteResponse(BaseModel):
    """Acknowledgement when ``output_format`` is ``file`` (payload written on the server)."""

    output_format: Literal["file"] = Field(
        "file",
        description="Response carries file paths only; full results are on disk.",
    )
    paths: dict[str, str] = Field(
        ...,
        description="Logical key → absolute path on the server host.",
    )


class CollectRequest(BaseModel):
    """PubMed collection parameters."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "diabetes mellitus[tiab]",
                "count": 25,
                "api_key": None,
                "email": None,
                "tool": "article_miner",
                "output_format": "json",
                "output_path": None,
            }
        }
    )

    query: str = Field(..., min_length=1, description="PubMed search query (Entrez syntax).")
    count: int = Field(100, ge=1, description="Maximum number of articles to retrieve.")
    api_key: str | None = Field(None, description="NCBI API key (or set NCBI_API_KEY).")
    email: str | None = Field(None, description="Contact email for NCBI etiquette.")
    tool: str = Field("article_miner", min_length=1, description="Tool name sent to NCBI.")
    output_format: OutputFormat = Field(
        "json",
        description="``json`` return body; ``file`` write JSON to ``output_path`` or default.",
    )
    output_path: str | None = Field(
        None,
        description="Server path for collection JSON when ``output_format`` is ``file``.",
    )


DEDUP_OPENAPI_EXAMPLES: dict[str, dict[str, Any]] = {
    "specter_faiss": {
        "summary": "SPECTER 2 + FAISS",
        "description": (
            "`collection_path` must point to **CollectionOutput** JSON on the server (e.g. file from "
            "`POST /collect` with `output_format=file`). Optional: `include_markdown`, `output_format`, "
            "`output_path`."
        ),
        "value": {
            "collection_path": "article_miner_output/collect.json",
            "enable_specter_faiss": True,
            "specter_model": "allenai/specter2_base",
        },
    },
    "rule_based_only": {
        "summary": "Rule-based dedup only",
        "value": {
            "collection_path": "article_miner_output/collect.json",
            "enable_specter_faiss": False,
        },
    },
}


class DedupRequest(BaseModel):
    """Request body for ``POST /dedup`` — **not** the response type ``DedupApiResponse``."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": DEDUP_OPENAPI_EXAMPLES["specter_faiss"]["value"],
        }
    )

    # Field order: input path → SPECTER flags → response/file options.
    collection_path: str = Field(
        ...,
        min_length=1,
        description=(
            "Path on the **server** to a ``CollectionOutput`` JSON file (same document as "
            "``POST /collect`` returns or writes)."
        ),
    )
    enable_specter_faiss: bool = Field(
        False,
        description="After rule-based dedup, run **SPECTER 2** embeddings + **FAISS** similarity (requires `[specter]` extra on server).",
    )
    specter_model: str | None = Field(
        None,
        description="Hugging Face model id for embeddings (default ``allenai/specter2_base``).",
    )
    include_markdown: bool = Field(
        False,
        description="Include Markdown summary (in JSON response or as a sibling file).",
    )
    output_format: OutputFormat = Field(
        "json",
        description="``json`` return body; ``file`` write report (and optional .md) to disk.",
    )
    output_path: str | None = Field(
        None,
        description="Server path for dedup JSON when ``output_format`` is ``file``.",
    )


INSIGHT_OPENAPI_EXAMPLES: dict[str, dict[str, Any]] = {
    "openai_file": {
        "summary": "Collection JSON on disk (OpenAI)",
        "description": (
            "`collection_path` must point to **CollectionOutput** JSON on the server (same schema as "
            "`POST /collect`, typically the file you pass to `POST /dedup` before insights)."
        ),
        "value": {
            "collection_path": "article_miner_output/collect.json",
            "llm": "openai",
            "model": None,
            "concurrency": 4,
            "enable_audit": True,
            "confidence_threshold": 0.5,
            "cache_path": None,
            "progress": False,
            "progress_every": 1,
            "output_format": "json",
            "output_path": None,
            "insight_file_format": "json",
            "write_report_md": True,
        },
    },
}


class InsightRequest(BaseModel):
    """Request body for ``POST /insights``."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": INSIGHT_OPENAPI_EXAMPLES["openai_file"]["value"],
        }
    )

    collection_path: str = Field(
        ...,
        min_length=1,
        description=(
            "Path on the **server** to a ``CollectionOutput`` JSON file (same document as "
            "``POST /collect``; often the same path used for ``POST /dedup`` after duplicate review)."
        ),
    )
    llm: str | None = Field(
        None,
        description=(
            "Provider shortcut (model from env): "
            + ", ".join(registered_insight_providers())
        ),
    )
    model: str | None = Field(
        None,
        description="Explicit model id when ``llm`` is not set (provider inferred).",
    )
    concurrency: int = Field(8, ge=1, le=64)
    enable_audit: bool = True
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0)
    cache_path: str | None = Field(
        None, description="Optional SQLite cache path on the server filesystem."
    )
    progress: bool = Field(False, description="Enable periodic progress logs (usually off for APIs).")
    progress_every: int = Field(1, ge=1)
    output_format: OutputFormat = Field(
        "json",
        description="``json`` return body; ``file`` write results to ``output_path`` or default.",
    )
    output_path: str | None = Field(
        None,
        description="Server path for insights output when ``output_format`` is ``file``.",
    )
    insight_file_format: InsightFileFormat = Field(
        "json",
        description="Machine-readable file layout when writing (matches CLI).",
    )
    write_report_md: bool = Field(
        True,
        description="When writing files, also write Markdown report next to main output.",
    )

    def resolve_insight_resolution(self) -> InsightLlmResolution:
        """Resolve provider + model for LangChain."""
        provider = (self.llm or "").strip().lower()
        if provider:
            return resolve_insight_llm_provider(provider, os.environ)
        if self.model:
            return resolve_explicit_model_id(self.model)
        return resolve_insight_llm_provider("openai", os.environ)


class DedupApiResponse(BaseModel):
    """Structured duplicate report plus optional Markdown summary."""

    report: DedupReport
    markdown: str | None = None
