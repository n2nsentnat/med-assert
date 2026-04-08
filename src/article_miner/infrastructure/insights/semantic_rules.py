"""Deterministic semantic flags (Layer 3) — heuristics, not medical truth."""

from __future__ import annotations

import re

from article_miner.domain.insights.models import LlmInsightExtraction, SemanticFlag

# Patterns are lowercased for matching
_SIG_NEG = re.compile(
    r"\b(no significant|not statistically significant|did not reach significance|"
    r"ns\b|p\s*>\s*0\.|non[- ]significant)\b",
    re.IGNORECASE,
)
_SIG_POS = re.compile(
    r"\b(p\s*<\s*0\.|statistically significant|significant difference|"
    r"significantly (higher|lower|reduced|improved))\b",
    re.IGNORECASE,
)
_DIR_NEG = re.compile(
    r"\b(no difference|failed to improve|did not improve|worse outcome|"
    r"increased (risk|mortality|adverse)|harm)\b",
    re.IGNORECASE,
)
_MAGNITUDE = re.compile(
    r"\b(nnt|number needed|absolute risk|relative risk|hazard ratio|odds ratio|"
    r"mean difference|md\b|effect size|clinically meaningful|patient[- ]important|"
    r"quality of life|qol|nnt\b|% reduction)\b",
    re.IGNORECASE,
)


def _joined_evidence(ext: LlmInsightExtraction) -> str:
    parts: list[str] = []
    for block in (
        ext.finding_direction,
        ext.statistical_significance,
        ext.clinical_meaningfulness,
        ext.main_claim,
    ):
        parts.extend(block.evidence_spans)
    return " ".join(parts).lower()


def run_semantic_rules(ext: LlmInsightExtraction) -> list[SemanticFlag]:
    """Return flags that may force human review; do not mutate ext."""
    flags: list[SemanticFlag] = []
    ev = _joined_evidence(ext)
    claim = ext.main_claim.value.lower()

    v = ext.statistical_significance.value.lower()
    if v == "significant" and _SIG_NEG.search(ev):
        flags.append(
            SemanticFlag(
                code="sig_vs_evidence_neg",
                message="Label significant but evidence text suggests non-significance language.",
                severity="warning",
            )
        )
    if v == "not_significant" and _SIG_POS.search(ev):
        flags.append(
            SemanticFlag(
                code="not_sig_vs_evidence_pos",
                message="Label not_significant but evidence suggests significance language.",
                severity="warning",
            )
        )

    d = ext.finding_direction.value.lower()
    if d == "positive" and (_DIR_NEG.search(ev) or _DIR_NEG.search(claim)):
        flags.append(
            SemanticFlag(
                code="positive_vs_negative_language",
                message="Finding direction positive but negative/null language present in evidence/claim.",
                severity="warning",
            )
        )

    cm = ext.clinical_meaningfulness.value.lower()
    if cm == "meaningful" and not _MAGNITUDE.search(ev) and not _MAGNITUDE.search(
        (ext.clinical_meaningfulness.reasoning_summary or "").lower()
    ):
        flags.append(
            SemanticFlag(
                code="meaningful_without_magnitude",
                message="Clinical meaningfulness is meaningful but no effect-size / patient-important language in evidence.",
                severity="warning",
            )
        )

    return flags
