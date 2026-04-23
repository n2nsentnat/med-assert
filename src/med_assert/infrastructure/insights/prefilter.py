"""Cheap pre-filter before LLM (cost control)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from med_assert.domain.collect.models import Article

_MIN_ABSTRACT_LEN = 40
_SKIP_TYPES = ("editorial", "comment", "news", "correction", "letter", "erratum")


class PrefilterAction(StrEnum):
    PASS = "pass"
    MINIMAL_UNCLEAR = "minimal_unclear"
    SKIP = "skip"


@dataclass(frozen=True, slots=True)
class PrefilterDecision:
    action: PrefilterAction
    reason: str | None = None


def prefilter_article(article: Article) -> PrefilterDecision:
    """Return routing decision before LLM call.

    Policy:
    - no/short abstract => minimal deterministic unclear output (no LLM call)
    - non-primary publication type => explicit prefilter skip
    - otherwise => normal classification path
    """
    ab = (article.abstract or "").strip()
    if not ab:
        return PrefilterDecision(
            action=PrefilterAction.MINIMAL_UNCLEAR,
            reason="skipped_prefilter_no_abstract",
        )
    if len(ab) < _MIN_ABSTRACT_LEN:
        return PrefilterDecision(
            action=PrefilterAction.MINIMAL_UNCLEAR,
            reason="skipped_prefilter_short_abstract",
        )
    for pt in article.publication_types:
        pl = pt.lower()
        for sk in _SKIP_TYPES:
            if sk in pl:
                return PrefilterDecision(
                    action=PrefilterAction.SKIP,
                    reason=f"skipped_prefilter_non_primary_research:{sk}",
                )
    return PrefilterDecision(action=PrefilterAction.PASS)
