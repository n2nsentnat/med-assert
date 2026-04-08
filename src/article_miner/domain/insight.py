"""LLM medical insight classification — domain types (schema + job results).

Deprecated path: prefer ``article_miner.domain.insights.models``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class FindingDirection(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"
    UNCLEAR = "unclear"


class StatisticalSignificance(StrEnum):
    SIGNIFICANT = "significant"
    NOT_SIGNIFICANT = "not_significant"
    UNCLEAR = "unclear"


class ClinicalMeaningfulness(StrEnum):
    MEANINGFUL = "meaningful"
    NOT_MEANINGFUL = "not_meaningful"
    UNCLEAR = "unclear"


class PerArticleStatus(StrEnum):
    """Outcome of the pipeline for one PMID."""

    SUCCESS = "success"
    NEEDS_HUMAN_REVIEW = "needs_human_review"
    INVALID_OUTPUT = "invalid_output"
    API_FAILURE = "api_failure"
    SKIPPED_PREFILTER = "skipped_prefilter"


class AutoAcceptStatus(StrEnum):
    AUTO_ACCEPT = "auto_accept"
    NEEDS_HUMAN_REVIEW = "needs_human_review"


class FieldInsightBlock(BaseModel):
    """One classified field from the LLM (Pass 1 shape)."""

    value: str
    confidence: float = Field(ge=0.0, le=1.0, description="Model self-reported confidence")
    evidence_spans: list[str] = Field(default_factory=list)


class ClinicalFieldInsight(FieldInsightBlock):
    """Clinical meaningfulness may include a short model rationale (not verified as span)."""

    reasoning_summary: str | None = None


class LlmInsightExtraction(BaseModel):
    """Raw structured output from Pass 1 (before deterministic validation)."""

    pmid: str
    finding_direction: FieldInsightBlock
    statistical_significance: FieldInsightBlock
    clinical_meaningfulness: ClinicalFieldInsight
    main_claim: FieldInsightBlock
    review_flags: list[str] = Field(default_factory=list)


class GroundingCheck(BaseModel):
    """Per-field result of substring grounding."""

    field_name: str
    all_spans_found: bool
    missing_spans: list[str] = Field(default_factory=list)


class SemanticFlag(BaseModel):
    """Deterministic contradiction / sanity flag."""

    code: str
    message: str
    severity: Literal["info", "warning", "error"] = "warning"


class ValidationPassResult(BaseModel):
    """Pass 2 aggregate."""

    schema_ok: bool
    schema_errors: list[str] = Field(default_factory=list)
    grounding: list[GroundingCheck] = Field(default_factory=list)
    semantic_flags: list[SemanticFlag] = Field(default_factory=list)
    truncation_warning: bool = False


class AuditResult(BaseModel):
    """Optional Pass 3 output."""

    supported: bool
    notes: str = ""
    raw_response: str | None = None


class ArticleInsightRecord(BaseModel):
    """Validated insight + metadata for export."""

    pmid: str
    extraction: LlmInsightExtraction
    validation: ValidationPassResult
    auto_accept: AutoAcceptStatus
    audit: AuditResult | None = None
    acceptance_reasons: list[str] = Field(default_factory=list)


class PerArticleInsightResult(BaseModel):
    """Single row in a classification job."""

    pmid: str
    status: PerArticleStatus
    insight: ArticleInsightRecord | None = None
    error_message: str | None = None
    raw_llm_text: str | None = Field(None, description="Debug: last model text if parse failed")
    prefilter_note: str | None = None


class InsightJobResult(BaseModel):
    """Full job output (JSON or JSONL aggregate)."""

    prompt_version: str
    model: str
    source_query: str | None = None
    articles: list[PerArticleInsightResult] = Field(default_factory=list)
    stats: dict[str, int | float] = Field(
        default_factory=dict,
        description="Counts: success, needs_review, invalid, api_failure, skipped, est_cost_usd, etc.",
    )
