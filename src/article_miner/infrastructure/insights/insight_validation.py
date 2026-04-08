"""Pass 2: structural validation, grounding, semantic rules, auto-accept."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from article_miner.domain.collect.models import Article
from article_miner.domain.insights.models import (
    AutoAcceptStatus,
    GroundingCheck,
    LlmInsightExtraction,
    SemanticFlag,
    ValidationPassResult,
)
from article_miner.infrastructure.insights.canonical_text import build_canonical_text, span_in_haystack
from article_miner.infrastructure.insights.semantic_rules import run_semantic_rules

ALLOWED_FINDING = frozenset({"positive", "negative", "neutral", "mixed", "unclear"})
ALLOWED_SIG = frozenset({"significant", "not_significant", "unclear"})
ALLOWED_CLIN = frozenset({"meaningful", "not_meaningful", "unclear"})

_UNCLEAR = "unclear"


def _validate_enum_labels(ext: LlmInsightExtraction) -> list[str]:
    errs: list[str] = []
    fd = ext.finding_direction.value.strip().lower()
    if fd not in ALLOWED_FINDING:
        errs.append(f"finding_direction must be one of {sorted(ALLOWED_FINDING)}, got {fd!r}")
    sg = ext.statistical_significance.value.strip().lower()
    if sg not in ALLOWED_SIG:
        errs.append(f"statistical_significance must be one of {sorted(ALLOWED_SIG)}, got {sg!r}")
    cm = ext.clinical_meaningfulness.value.strip().lower()
    if cm not in ALLOWED_CLIN:
        errs.append(f"clinical_meaningfulness must be one of {sorted(ALLOWED_CLIN)}, got {cm!r}")
    if not ext.main_claim.value.strip():
        errs.append("main_claim.value must be non-empty")
    return errs


def _ground_field(
    field_name: str,
    value: str,
    spans: list[str],
    haystack: str,
    *,
    fuzzy_whitespace: bool = True,
) -> GroundingCheck:
    """Require ≥1 grounded span when value is not unclear; unclear may have zero spans."""
    v = value.strip().lower()
    missing: list[str] = []
    if v == _UNCLEAR:
        for sp in spans:
            if sp.strip() and not span_in_haystack(sp, haystack, fuzzy_whitespace=fuzzy_whitespace):
                missing.append(sp)
        return GroundingCheck(
            field_name=field_name,
            all_spans_found=len(missing) == 0,
            missing_spans=missing,
        )
    if not spans:
        return GroundingCheck(
            field_name=field_name,
            all_spans_found=False,
            missing_spans=["<no evidence span provided>"],
        )
    for sp in spans:
        if not sp.strip():
            missing.append("<empty span>")
        elif not span_in_haystack(sp, haystack, fuzzy_whitespace=fuzzy_whitespace):
            missing.append(sp)
    return GroundingCheck(
        field_name=field_name,
        all_spans_found=len(missing) == 0,
        missing_spans=missing,
    )


def grounding_checks(article: Article, ext: LlmInsightExtraction) -> list[GroundingCheck]:
    hay = build_canonical_text(article)
    return [
        _ground_field("finding_direction", ext.finding_direction.value, ext.finding_direction.evidence_spans, hay),
        _ground_field(
            "statistical_significance",
            ext.statistical_significance.value,
            ext.statistical_significance.evidence_spans,
            hay,
        ),
        _ground_field(
            "clinical_meaningfulness",
            ext.clinical_meaningfulness.value,
            ext.clinical_meaningfulness.evidence_spans,
            hay,
        ),
        _ground_field("main_claim", ext.main_claim.value, ext.main_claim.evidence_spans, hay),
    ]


def parse_extraction_json(text: str) -> tuple[LlmInsightExtraction | None, list[str]]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, [f"Invalid JSON: {exc}"]
    if not isinstance(data, dict):
        return None, ["JSON root must be an object"]
    try:
        return LlmInsightExtraction.model_validate(data), []
    except ValidationError as exc:
        return None, [str(exc)]


def run_pass2_validation(
    article: Article,
    ext: LlmInsightExtraction,
    *,
    confidence_threshold: float = 0.5,
    truncation_warning: bool = False,
) -> tuple[ValidationPassResult, list[SemanticFlag], AutoAcceptStatus, list[str]]:
    """Run full Pass 2; return validation result, semantic flags, auto_accept, reasons."""
    schema_errors = _validate_enum_labels(ext)
    schema_ok = len(schema_errors) == 0

    ground = grounding_checks(article, ext)
    ground_ok = all(g.all_spans_found for g in ground)

    sem = run_semantic_rules(ext)
    sem_has_error = any(f.severity == "error" for f in sem)

    val = ValidationPassResult(
        schema_ok=schema_ok,
        schema_errors=schema_errors,
        grounding=ground,
        semantic_flags=sem,
        truncation_warning=truncation_warning,
    )

    reasons: list[str] = []
    if not schema_ok:
        reasons.append("schema_validation_failed")
    if not ground_ok:
        reasons.append("grounding_failed")
    if sem:
        reasons.append("semantic_flags_present")
    low_conf = (
        ext.finding_direction.confidence < confidence_threshold
        or ext.statistical_significance.confidence < confidence_threshold
        or ext.clinical_meaningfulness.confidence < confidence_threshold
        or ext.main_claim.confidence < confidence_threshold
    )
    if low_conf:
        reasons.append("below_confidence_threshold")

    mixed = ext.finding_direction.value.lower() == "mixed"
    meaningful = ext.clinical_meaningfulness.value.lower() == "meaningful"
    if mixed or meaningful:
        reasons.append("audit_recommended")

    auto = AutoAcceptStatus.AUTO_ACCEPT
    if not schema_ok or not ground_ok or sem_has_error or truncation_warning:
        auto = AutoAcceptStatus.NEEDS_HUMAN_REVIEW
    elif sem or low_conf or mixed or meaningful:
        auto = AutoAcceptStatus.NEEDS_HUMAN_REVIEW
    else:
        reasons.append("auto_accept_eligible")

    return val, sem, auto, reasons


def merge_dict_for_audit(ext: LlmInsightExtraction) -> dict[str, Any]:
    return json.loads(ext.model_dump_json())
