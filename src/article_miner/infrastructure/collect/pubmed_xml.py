"""Parse PubMed efetch XML into domain Article models (lxml + XPath)."""

from __future__ import annotations

import io
import logging
from datetime import datetime
from typing import Iterable

from lxml import etree

from article_miner.domain.collect.models import Article, Author
from article_miner.domain.errors import ArticleParseError, MalformedResponseError

logger = logging.getLogger(__name__)


def _ln(name: str) -> str:
    return f"*[local-name()='{name}']"


def _fallback_pmid_for_log(article_el: etree._Element) -> str | None:
    """Best-effort PMID when full parse fails (for diagnostics only)."""
    for pmid_el in article_el.xpath(f".//{_ln('PMID')}"):
        t = _text(pmid_el)
        if t:
            return t
    return None


def _text(el: etree._Element | None) -> str | None:
    """Return direct text of ``el`` (not descendants), stripped of leading/trailing whitespace.

    Whitespace-only content becomes ``None``. Leading/trailing spaces are removed from
    all returned strings; this matches typical PubMed field cleanup but is worth
    noting if you need exact XML spacing.
    """
    if el is None:
        return None
    t = (el.text or "").strip()
    return t or None


def _element_string(el: etree._Element) -> str:
    """All text in this element (including nested), e.g. for AbstractText."""
    return "".join(el.itertext()).strip()


def _first_child(parent: etree._Element | None, local: str) -> etree._Element | None:
    """Direct child by local name; ``parent`` may be ``None`` (returns ``None``)."""
    if parent is None:
        return None
    found = parent.xpath(f"./{_ln(local)}")
    return found[0] if found else None


def _find_text(parent: etree._Element | None, local: str) -> str | None:
    """Text of a direct child; ``parent`` may be ``None`` (returns ``None``)."""
    return _text(_first_child(parent, local))


def _join_labeled_child_segments(
    parent: etree._Element | None,
    child_local: str,
    *,
    label_attr: str = "Label",
    joiner: str = "\n\n",
) -> str | None:
    """Build text from direct children (e.g. ``AbstractText``) with optional ``Label`` per segment.

    If there are no matching children, returns the parent's own text (``_text``), matching
    PubMed's occasional single-block abstract without ``AbstractText`` children.
    """
    if parent is None:
        return None
    parts: list[str] = []
    for child in parent.xpath(f"./{_ln(child_local)}"):
        label = child.get(label_attr)
        piece = _element_string(child)
        if label:
            parts.append(f"{label}: {piece}".strip())
        else:
            parts.append(piece.strip())
    if parts:
        return joiner.join(p for p in parts if p)
    return _text(parent)


def _collect_abstract(abstract_el: etree._Element | None) -> str | None:
    return _join_labeled_child_segments(abstract_el, "AbstractText")


def _parse_pubmed_month(raw: str | None) -> int | None:
    """Map PubMed ``Month`` text (e.g. ``Jan``, ``March``, ``03``) to 1–12."""
    if raw is None:
        return None
    s = raw.strip()
    if not s:
        return None
    if s.isdigit():
        n = int(s)
        return n if 1 <= n <= 12 else None
    try:
        return datetime.strptime(s[:3].title(), "%b").month
    except ValueError:
        pass
    try:
        return datetime.strptime(s.title(), "%B").month
    except ValueError:
        return None


def _parse_date_container(
    pub_date: etree._Element | None,
) -> tuple[int | None, int | None, int | None]:
    if pub_date is None:
        return None, None, None
    year_el = _first_child(pub_date, "Year")
    month_el = _first_child(pub_date, "Month")
    day_el = _first_child(pub_date, "Day")
    year: int | None = None
    day: int | None = None
    if year_el is not None and (year_el.text or "").strip():
        try:
            year = int(year_el.text.strip())  # type: ignore[union-attr]
        except ValueError:
            year = None
    month = _parse_pubmed_month(_text(month_el))
    if day_el is not None and (day_el.text or "").strip():
        try:
            day = int(day_el.text.strip())  # type: ignore[union-attr]
        except ValueError:
            day = None
    return year, month, day


def _parse_authors(author_list: etree._Element | None) -> list[Author]:
    if author_list is None:
        return []
    authors: list[Author] = []
    for author in author_list.xpath(f"./{_ln('Author')}"):
        aff = author.xpath(f"./{_ln('AffiliationInfo')}/{_ln('Affiliation')}")
        affiliation = _text(aff[0]) if aff else None
        authors.append(
            Author(
                last_name=_find_text(author, "LastName"),
                fore_name=_find_text(author, "ForeName"),
                initials=_find_text(author, "Initials"),
                affiliation=affiliation,
            )
        )
    return authors


def _parse_mesh(mesh_heading_list: etree._Element | None) -> list[str]:
    if mesh_heading_list is None:
        return []
    out: list[str] = []
    for mh in mesh_heading_list.xpath(f"./{_ln('MeshHeading')}"):
        parts: list[str] = []
        for child in mh:
            ln = etree.QName(child).localname
            if ln == "DescriptorName":
                t = _text(child)
                if t:
                    parts.append(t)
            elif ln == "QualifierName":
                t = _text(child)
                if t:
                    parts.append(t)
        if parts:
            out.append(" / ".join(parts))
    return out


def _parse_keywords(keyword_list: etree._Element | None) -> list[str]:
    if keyword_list is None:
        return []
    out: list[str] = []
    for kw in keyword_list.xpath(f"./{_ln('Keyword')}"):
        t = _text(kw)
        if t:
            out.append(t)
    return out


def _parse_publication_types(pub_type_list: etree._Element | None) -> list[str]:
    if pub_type_list is None:
        return []
    out: list[str] = []
    for pt in pub_type_list.xpath(f"./{_ln('PublicationType')}"):
        t = _text(pt)
        if t:
            out.append(t)
    return out


def _article_ids(pubmed_data: etree._Element | None) -> tuple[str | None, str | None]:
    if pubmed_data is None:
        return None, None
    doi: str | None = None
    pmc: str | None = None
    id_list = _first_child(pubmed_data, "ArticleIdList")
    if id_list is None:
        return doi, pmc
    for aid in id_list.xpath(f"./{_ln('ArticleId')}"):
        id_type = aid.get("IdType")
        val = _text(aid)
        if not val:
            continue
        if id_type == "doi":
            doi = val
        elif id_type == "pmc":
            pmc = val
    return doi, pmc


def _parse_doi(art: etree._Element | None, doi_from_pubmed_data: str | None) -> str | None:
    """Resolve DOI: prefer ``Article/ELocationID[@EIdType='doi']``, else ``PubmedData`` ArticleId."""
    if art is not None:
        found = art.xpath(f"./{_ln('ELocationID')}[@EIdType='doi']")
        if found:
            t = _text(found[0])
            if t:
                return t
    return doi_from_pubmed_data


def parse_pubmed_article_element(article_el: etree._Element) -> Article:
    """Parse a single PubmedArticle element."""
    if etree.QName(article_el).localname != "PubmedArticle":
        msg = "Expected PubmedArticle root fragment"
        raise ArticleParseError(msg)

    medline = _first_child(article_el, "MedlineCitation")
    if medline is None:
        msg = "Missing MedlineCitation"
        raise ArticleParseError(msg)

    pmid_el = _first_child(medline, "PMID")
    pmid = _text(pmid_el)
    if not pmid:
        msg = "Missing PMID"
        raise ArticleParseError(msg)

    art = _first_child(medline, "Article")
    title = _find_text(art, "ArticleTitle")
    abstract = _collect_abstract(_first_child(art, "Abstract"))
    journal = _first_child(art, "Journal")
    journal_full = _find_text(journal, "Title")
    journal_iso = _find_text(journal, "ISOAbbreviation")
    issue = _first_child(journal, "JournalIssue")
    pub_date = _first_child(issue, "PubDate")
    year, month, day = _parse_date_container(pub_date)

    pubmed_data = _first_child(article_el, "PubmedData")
    id_doi, pmc = _article_ids(pubmed_data)
    doi = _parse_doi(art, id_doi)

    lang = _find_text(art, "Language")
    authors = _parse_authors(_first_child(art, "AuthorList"))
    pub_types = _parse_publication_types(_first_child(art, "PublicationTypeList"))
    mesh = _parse_mesh(_first_child(medline, "MeshHeadingList"))
    keywords = _parse_keywords(_first_child(medline, "KeywordList"))

    return Article(
        pmid=pmid,
        title=title,
        abstract=abstract,
        journal_full=journal_full,
        journal_iso=journal_iso,
        publication_year=year,
        publication_month=month,
        publication_day=day,
        doi=doi,
        pmc_id=pmc,
        language=lang,
        publication_types=pub_types,
        mesh_terms=mesh,
        keywords=keywords,
        authors=authors,
    )


def _release_iterparse_element(elem: etree._Element) -> None:
    """Remove a processed subtree from the incremental parse tree to cap memory."""
    parent = elem.getparent()
    if parent is not None:
        parent.remove(elem)
    else:
        elem.clear()


def _parse_articles_iterparse(data: bytes) -> list[Article]:
    """Parse ``PubmedArticle`` nodes via ``iterparse`` (does not build a full DOM)."""
    articles: list[Article] = []
    stream = io.BytesIO(data)
    context = etree.iterparse(
        stream,
        events=("end",),
        huge_tree=True,
        resolve_entities=False,
    )
    try:
        for _event, elem in context:
            if etree.QName(elem).localname != "PubmedArticle":
                continue
            try:
                articles.append(parse_pubmed_article_element(elem))
            except ArticleParseError as exc:
                pmid_hint = _fallback_pmid_for_log(elem)
                logger.warning(
                    "Skipping PubmedArticle (parse error): pmid=%s detail=%s",
                    pmid_hint or "unknown",
                    exc,
                )
            finally:
                _release_iterparse_element(elem)
    except etree.XMLSyntaxError as exc:
        raise MalformedResponseError(f"Invalid XML from efetch: {exc}") from exc
    return articles


def parse_pubmed_xml_document(xml_text: str | bytes) -> list[Article]:
    """Parse a full PubmedArticleSet document into Article models.

    Uses :func:`etree.iterparse` so large efetch payloads are not kept as one full
    in-memory tree; each ``PubmedArticle`` is released after parsing.
    """
    data = xml_text.encode("utf-8") if isinstance(xml_text, str) else xml_text
    return _parse_articles_iterparse(data)


def iter_pubmed_article_elements(xml_text: str) -> Iterable[etree._Element]:
    """Yield PubmedArticle elements (loads the full document into memory).

    Prefer :func:`parse_pubmed_xml_document` for large batches; this iterator keeps
    the entire XML DOM until iteration completes.
    """
    try:
        root = etree.fromstring(
            xml_text.encode("utf-8") if isinstance(xml_text, str) else xml_text,
            parser=etree.XMLParser(resolve_entities=False, huge_tree=True),
        )
    except etree.XMLSyntaxError as exc:
        raise MalformedResponseError(f"Invalid XML from efetch: {exc}") from exc
    yield from root.xpath(f"//{_ln('PubmedArticle')}")

