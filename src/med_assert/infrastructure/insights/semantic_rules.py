"""Deterministic semantic flags (Layer 3) — heuristics, not medical truth.

Policy:
- Use high-precision rules for ``severity="error"`` only when contradiction is obvious.
- Keep broader regex heuristics as ``warning`` flags to avoid brittle hard failures.
"""

from __future__ import annotations

import re

from med_assert.domain.insights.models import LlmInsightExtraction, SemanticFlag

# Patterns are lowercased for matching
_SIG_NEG = re.compile(
    r"\b(no significant difference|not statistically significant|"
    r"did not reach statistical significance|non[- ]significant)\b|"
    r"\bp\s*>\s*0\.0?5\b",
    re.IGNORECASE,
)
_SIG_POS = re.compile(
    r"\b(p\s*<\s*0\.0?5|statistically significant|significant difference|"
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
    neg_sig = bool(_SIG_NEG.search(ev))
    # Avoid counting positive-significance substrings embedded inside negative phrases
    # such as "not statistically significant" or "no significant difference".
    ev_without_neg = _SIG_NEG.sub(" ", ev)
    pos_sig = bool(_SIG_POS.search(ev_without_neg))
    if neg_sig and pos_sig:
        flags.append(
            SemanticFlag(
                code="mixed_significance_language",
                message="Evidence contains both significance and non-significance language; treat as ambiguous.",
                severity="warning",
            )
        )
    elif v == "significant" and neg_sig:
        flags.append(
            SemanticFlag(
                code="sig_vs_evidence_neg",
                message="Label significant but evidence text suggests non-significance language.",
                severity="error",
            )
        )
    elif v == "not_significant" and pos_sig:
        flags.append(
            SemanticFlag(
                code="not_sig_vs_evidence_pos",
                message="Label not_significant but evidence suggests significance language.",
                severity="error",
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
    if (
        cm == "meaningful"
        and not _MAGNITUDE.search(ev)
        and not _MAGNITUDE.search(
            (ext.clinical_meaningfulness.reasoning_summary or "").lower()
        )
    ):
        flags.append(
            SemanticFlag(
                code="meaningful_without_magnitude",
                message="Clinical meaningfulness is meaningful but no effect-size / patient-important language in evidence.",
                severity="warning",
            )
        )

    return flags
