"""Canonical title+abstract string for evidence grounding (must match prompt layout).

Grounding policy: evidence spans are verified as substrings of the canonical haystack.
Optional ``fuzzy_whitespace`` collapses runs of whitespace in both span and haystack before
matching (bounded fallback for copy/paste whitespace differences only; not semantic normalization).
"""

from __future__ import annotations

import re

from med_assert.domain.collect.models import Article


def build_canonical_text(article: Article) -> str:
    """Build the haystack used to verify evidence spans.

    Uses the same logical sections as the user prompt: Title line + Abstract line.
    Normalization: strip each part, single newlines between title and abstract.
    """
    title = (article.title or "").strip()
    abstract = (article.abstract or "").strip()
    if title and abstract:
        return f"{title}\n{abstract}"
    return title or abstract


def normalize_for_match(s: str) -> str:
    """Collapse whitespace for fuzzy fallback (optional)."""
    return re.sub(r"\s+", " ", s.strip())


def span_in_haystack(
    span: str, haystack: str, *, fuzzy_whitespace: bool = False
) -> bool:
    """Return True if span appears in haystack (exact substring)."""
    if not span:
        return False
    s = span.strip()
    if not s:
        return False
    if s in haystack:
        return True
    if fuzzy_whitespace:
        ns = normalize_for_match(s)
        nh = normalize_for_match(haystack)
        return ns in nh
    return False
