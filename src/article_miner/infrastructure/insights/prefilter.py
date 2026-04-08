"""Cheap pre-filter before LLM (cost control)."""

from __future__ import annotations

from article_miner.domain.collect.models import Article

_MIN_ABSTRACT_LEN = 40
_SKIP_TYPES = ("editorial", "comment", "news", "correction", "letter", "erratum")


def prefilter_article(article: Article) -> str | None:
    """Return a skip reason string, or None if the article should be classified normally."""
    ab = (article.abstract or "").strip()
    if not ab:
        return "missing_abstract"
    if len(ab) < _MIN_ABSTRACT_LEN:
        return "short_abstract"
    for pt in article.publication_types:
        pl = pt.lower()
        for sk in _SKIP_TYPES:
            if sk in pl:
                return f"publication_type:{sk}"
    return None
