"""Find probable duplicate article groups in a ``CollectionOutput`` JSON.

What counts as a duplicate here
--------------------------------
PubMed does not give a single canonical "same paper" key across preprints,
meetings, and versions. This module therefore reports probable duplicates
for human review, not automatic deletion.
"""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from med_assert.domain.collect.models import Article, CollectionOutput

FUZZY_TITLE_RATIO_MIN = 90
FUZZY_TITLE_TOKEN_SORT_MIN = 92
ABSTRACT_TOKEN_SORT_MIN = 78
MAX_BLOCK_SIZE = 280
_TITLE_PREFIX_WORDS = 4
_NY_LEN_BAND = 20
_NY_HEAD_CHARS = 40

_EDGE_SAME_DOI = "same_doi"
_EDGE_SAME_TITLE_YEAR = "same_title_year"
_EDGE_FUZZY = "fuzzy"
_EDGE_SPECTER_FAISS = "specter_faiss"

logger = logging.getLogger(__name__)


class DuplicateCluster(BaseModel):
    cluster_id: int
    pmids: list[str] = Field(description="PubMed IDs in this group (sorted)")
    primary_reason: str = Field(description="Dominant link type in the cluster")
    confidence: Literal["high", "medium"]
    detail: str = ""
    edge_evidence: list[str] = Field(default_factory=list)
    transitivity_note: str | None = None
    reviewer_notes: list[str] = Field(default_factory=list)


class DedupReport(BaseModel):
    source_article_count: int
    duplicate_group_count: int
    articles_in_some_duplicate_group: int
    methodology: str
    clusters: list[DuplicateCluster]
    stats: dict[str, int | float | str] = Field(default_factory=dict)


def format_dedup_markdown(report: DedupReport) -> str:
    lines = [
        "# Probable duplicate groups",
        "",
        f"- Source articles: **{report.source_article_count}**",
        f"- Duplicate groups (size >= 2): **{report.duplicate_group_count}**",
        f"- Articles appearing in some group: **{report.articles_in_some_duplicate_group}**",
        f"- Fuzzy pair comparisons: **{report.stats.get('fuzzy_pairs_compared', 0)}**",
        f"- SPECTER 2 + FAISS edges added: **{report.stats.get('specter_pairs_added', 0)}**",
        "",
        "## Definition (summary)",
        "",
        report.methodology,
        "",
        "## Groups",
        "",
    ]
    for c in report.clusters:
        lines.append(
            f"### Cluster {c.cluster_id} - `{c.primary_reason}` ({c.confidence})"
        )
        lines.append("")
        lines.append(c.detail)
        lines.append("")
        if c.edge_evidence:
            lines.append("**Pairwise evidence:**")
            for ev in c.edge_evidence:
                lines.append(f"- {ev}")
            lines.append("")
        if c.transitivity_note:
            lines.append(f"*Note:* {c.transitivity_note}")
            lines.append("")
        lines.append("| PMID |")
        lines.append("|------|")
        for p in c.pmids:
            lines.append(f"| {p} |")
        if c.reviewer_notes:
            lines.append("")
            lines.append("**Reviewer notes:**")
            for n in c.reviewer_notes:
                lines.append(f"- {n}")
        lines.append("")
    return "\n".join(lines)


@dataclass
class _UnionFind:
    parent: list[int]
    rank: list[int]

    @classmethod
    def new(cls, n: int) -> _UnionFind:
        return cls(list(range(n)), [0] * n)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    s = doi.strip().lower()
    for prefix in (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi:",
    ):
        if s.startswith(prefix):
            s = s[len(prefix) :]
    s = s.strip().rstrip(".,;)")
    return s or None


def normalize_title(title: str | None) -> str:
    if not title:
        return ""
    t = unicodedata.normalize("NFKD", title).lower()
    t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
    return re.sub(r"\s+", " ", t).strip()


def _title_prefix_key(norm: str) -> str:
    return " ".join(norm.split()[:_TITLE_PREFIX_WORDS]) if norm else ""


def _missing_year_block_key(nt: str, prefix: str) -> str:
    ln = len(nt)
    band = ln // _NY_LEN_BAND
    head = nt[:_NY_HEAD_CHARS] if nt else ""
    return f"ny|{prefix}|b{band}|{head}"


def _pmid_sort_key(pmid: str) -> tuple[int, int | str]:
    try:
        return (0, int(pmid))
    except ValueError:
        return (1, pmid)


def _append_edge(edges: list[tuple[int, int, str]], i: int, j: int, kind: str) -> None:
    if i == j:
        return
    if i > j:
        i, j = j, i
    edges.append((i, j, kind))


def _abstract_norm(ab: str | None) -> str:
    if not ab:
        return ""
    t = unicodedata.normalize("NFKD", ab)
    t = re.sub(r"\s+", " ", t.lower().strip())
    return t[:8000]


def _maybe_retraction_notes(a: Article) -> list[str]:
    notes: list[str] = []
    title_l = (a.title or "").lower()
    if "retract" in title_l:
        notes.append(f"PMID {a.pmid}: title mentions retraction (verify in PubMed)")
    for pt in a.publication_types:
        if "retract" in pt.lower():
            notes.append(
                f"PMID {a.pmid}: publication type includes '{pt}' (verify replacement)"
            )
            break
    return notes


def _cluster_metadata(
    indices: list[int],
    articles: list[Article],
    edges_in_cluster: list[tuple[int, int, str]],
) -> tuple[str, Literal["high", "medium"], str, list[str], str | None]:
    kinds = {k for _, _, k in edges_in_cluster}
    evidence: list[str] = []
    label_map = {
        _EDGE_SAME_DOI: "same_doi",
        _EDGE_SAME_TITLE_YEAR: "same_normalized_title_and_year",
        _EDGE_FUZZY: "fuzzy_title",
        _EDGE_SPECTER_FAISS: "specter2_embedding_cosine",
    }
    for i, j, k in sorted(
        edges_in_cluster,
        key=lambda e: (
            _pmid_sort_key(articles[e[0]].pmid),
            _pmid_sort_key(articles[e[1]].pmid),
        ),
    ):
        pair = sorted([articles[i].pmid, articles[j].pmid], key=_pmid_sort_key)
        evidence.append(f"{pair[0]}↔{pair[1]}: {label_map.get(k, k)}")

    trans_parts: list[str] = []
    if len(indices) > 2 and _EDGE_FUZZY in kinds:
        trans_parts.append(
            "Union-find can join records transitively via fuzzy links; not every pair may meet threshold directly."
        )
    if len(kinds) > 1:
        trans_parts.append("This cluster mixes link types; see pairwise evidence.")
    trans_note = " ".join(trans_parts) if trans_parts else None

    if len(kinds) > 1:
        primary = "mixed_evidence"
        conf: Literal["high", "medium"] = (
            "medium"
            if (_EDGE_FUZZY in kinds or _EDGE_SPECTER_FAISS in kinds)
            else "high"
        )
        detail = "Cluster contains more than one link type; use edge_evidence for ground truth."
    elif _EDGE_SAME_DOI in kinds:
        primary = "same_doi"
        conf = "high"
        detail = "Same normalized DOI across different PMIDs."
    elif _EDGE_SAME_TITLE_YEAR in kinds:
        primary = "same_normalized_title_and_year"
        conf = "high"
        detail = "Identical normalized title and same non-null publication year."
    elif _EDGE_SPECTER_FAISS in kinds:
        primary = "specter_embedding_similarity"
        conf = "medium"
        detail = (
            "High cosine similarity between SPECTER 2 paper embeddings "
            "(FAISS inner-product search on L2-normalized vectors)."
        )
    else:
        primary = "fuzzy_title_and_or_abstract"
        conf = "medium"
        detail = "Similar title (and abstract when present) via fuzzy matching within blocks."

    return primary, conf, detail, evidence, trans_note


def _apply_specter_faiss_edges(
    articles: list[Article],
    uf: _UnionFind,
    edges: list[tuple[int, int, str]],
    *,
    specter_model: str | None,
) -> int:
    from med_assert.infrastructure.dedup.specter_faiss import (
        DEFAULT_SPECTER_MODEL,
        compute_specter_embeddings,
        faiss_cosine_pairs,
    )

    mid = specter_model or DEFAULT_SPECTER_MODEL
    emb, _dim = compute_specter_embeddings(articles, model_id=mid)
    pairs = faiss_cosine_pairs(emb)
    added = 0
    for i, j, _sim in pairs:
        if uf.find(i) == uf.find(j):
            continue
        _append_edge(edges, i, j, _EDGE_SPECTER_FAISS)
        uf.union(i, j)
        added += 1
    return added


def build_duplicate_report(
    collection: CollectionOutput,
    *,
    enable_specter_faiss: bool | None = None,
    specter_model: str | None = None,
) -> DedupReport:
    articles = collection.articles
    n = len(articles)
    uf = _UnionFind.new(n)
    edges: list[tuple[int, int, str]] = []
    fuzzy_pairs_compared = 0

    doi_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, a in enumerate(articles):
        d = normalize_doi(a.doi)
        if d:
            doi_to_indices[d].append(i)
    for group in doi_to_indices.values():
        for ii in range(len(group)):
            for jj in range(ii + 1, len(group)):
                a, b = group[ii], group[jj]
                _append_edge(edges, a, b, _EDGE_SAME_DOI)
                uf.union(a, b)

    exact_key: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, a in enumerate(articles):
        nt = normalize_title(a.title)
        if nt and a.publication_year is not None:
            exact_key[(nt, a.publication_year)].append(i)
    for group in exact_key.values():
        for ii in range(len(group)):
            for jj in range(ii + 1, len(group)):
                a, b = group[ii], group[jj]
                _append_edge(edges, a, b, _EDGE_SAME_TITLE_YEAR)
                uf.union(a, b)

    block: dict[str, list[int]] = defaultdict(list)
    for i, a in enumerate(articles):
        nt = normalize_title(a.title)
        if len(nt) < 12:
            continue
        prefix = _title_prefix_key(nt)
        if not prefix:
            continue
        if a.publication_year is None:
            key = _missing_year_block_key(nt, prefix)
        else:
            key = f"{a.publication_year}|{prefix}"
        block[key].append(i)

    def split_oversized(bucket: list[int]) -> list[list[int]]:
        if len(bucket) <= MAX_BLOCK_SIZE:
            return [bucket]
        indexed = [(i, len(normalize_title(articles[i].title))) for i in bucket]
        indexed.sort(key=lambda x: x[1])
        chunks: list[list[int]] = []
        chunk: list[int] = []
        for idx, _ln in indexed:
            chunk.append(idx)
            if len(chunk) >= MAX_BLOCK_SIZE // 2:
                chunks.append(chunk)
                chunk = []
        if chunk:
            chunks.append(chunk)
        return chunks

    for bucket in block.values():
        for sub in split_oversized(bucket):
            m = len(sub)
            for ii in range(m):
                for jj in range(ii + 1, m):
                    i, j = sub[ii], sub[jj]
                    if uf.find(i) == uf.find(j):
                        continue
                    ai, aj = articles[i], articles[j]
                    ti = normalize_title(ai.title)
                    tj = normalize_title(aj.title)
                    if not ti or not tj:
                        continue
                    li, lj = len(ti), len(tj)
                    if li > 20 and lj > 20 and min(li, lj) < max(li, lj) * 0.55:
                        continue
                    fuzzy_pairs_compared += 1
                    tr = fuzz.ratio(ti, tj)
                    ts = fuzz.token_sort_ratio(ti, tj)
                    if not (
                        tr >= FUZZY_TITLE_RATIO_MIN or ts >= FUZZY_TITLE_TOKEN_SORT_MIN
                    ):
                        continue
                    abi = _abstract_norm(ai.abstract)
                    abj = _abstract_norm(aj.abstract)
                    if (
                        abi
                        and abj
                        and fuzz.token_sort_ratio(abi, abj) < ABSTRACT_TOKEN_SORT_MIN
                    ):
                        continue
                    _append_edge(edges, i, j, _EDGE_FUZZY)
                    uf.union(i, j)

    if enable_specter_faiss is None:
        enable_specter_faiss = os.environ.get(
            "MED_ASSERT_SPECTER", "0"
        ).lower() in ("1", "true", "yes")

    specter_pairs_added = 0
    specter_model_used = ""
    if enable_specter_faiss and n >= 2:
        try:
            from med_assert.infrastructure.dedup.specter_faiss import (
                DEFAULT_SPECTER_MODEL,
            )

            specter_model_used = specter_model or DEFAULT_SPECTER_MODEL
            specter_pairs_added = _apply_specter_faiss_edges(
                articles, uf, edges, specter_model=specter_model
            )
        except Exception as exc:
            logger.warning("SPECTER 2 / FAISS deduplication skipped: %s", exc)

    comp: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        comp[uf.find(i)].append(i)

    edges_by_root: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    for i, j, k in edges:
        edges_by_root[uf.find(i)].append((i, j, k))

    clusters: list[DuplicateCluster] = []
    cid = 0
    in_group = 0
    for root, indices in sorted(comp.items(), key=lambda x: min(x[1])):
        if len(indices) < 2:
            continue
        indices.sort(key=lambda i: _pmid_sort_key(articles[i].pmid))
        pmids = [articles[i].pmid for i in indices]
        primary, conf, detail, edge_lines, trans_note = _cluster_metadata(
            indices, articles, edges_by_root.get(root, [])
        )
        notes: list[str] = []
        for i in indices:
            notes.extend(_maybe_retraction_notes(articles[i]))
        uniq_notes = list(dict.fromkeys(notes))
        cid += 1
        in_group += len(indices)
        clusters.append(
            DuplicateCluster(
                cluster_id=cid,
                pmids=sorted(pmids, key=_pmid_sort_key),
                primary_reason=primary,
                confidence=conf,
                detail=detail,
                edge_evidence=edge_lines,
                transitivity_note=trans_note,
                reviewer_notes=uniq_notes[:20],
            )
        )

    methodology = (
        "Duplicates are probable groups for human review. "
        "High: same normalized DOI, or same normalized title + same non-null publication year "
        "(missing year never uses this rule). "
        "Medium: fuzzy title (ratio>=90 or token_sort>=92) within blocks; missing-year rows use "
        "prefix + length band + title head. "
        f"If both abstracts exist, token_sort>={ABSTRACT_TOKEN_SORT_MIN} on abstract text. "
        "Optional second layer: SPECTER 2 embeddings (sentence-transformers) with FAISS "
        "inner-product search on L2-normalized vectors (cosine), controlled by "
        "MED_ASSERT_SPECTER / --specter. "
        "Clusters list pairwise edge_evidence; transitive union-find can connect chains."
    )

    stats: dict[str, int | float | str] = {
        "fuzzy_pairs_compared": fuzzy_pairs_compared,
        "blocks_used": len(block),
        "specter_pairs_added": specter_pairs_added,
    }
    if specter_model_used:
        stats["specter_model"] = specter_model_used

    return DedupReport(
        source_article_count=n,
        duplicate_group_count=len(clusters),
        articles_in_some_duplicate_group=in_group,
        methodology=methodology,
        clusters=clusters,
        stats=stats,
    )


def load_collection(path: str) -> CollectionOutput:
    return CollectionOutput.model_validate_json(Path(path).read_text(encoding="utf-8"))
