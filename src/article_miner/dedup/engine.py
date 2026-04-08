"""Find probable duplicate article groups in a ``CollectionOutput`` JSON.

What counts as a duplicate here
--------------------------------
PubMed does not give a single canonical “same paper” key across preprints,
meetings, and versions. This module therefore reports **probable** duplicates
for human review, not automatic deletion.

**Edge rules (pairwise links, not per-cluster “uniform” confidence)**

These describe when **two** records are connected by an edge. A **cluster** is a
**connected component** in the union of those edges, so it can **mix** rule
types (e.g. one DOI edge plus one fuzzy edge in a 3-item cluster). **Do not**
read a cluster label as “everything here is high confidence”—use
``edge_evidence`` and ``transitivity_note`` on each cluster.

1. **Same DOI (strong pairwise link)** — After normalizing ``https://doi.org/`` (and
   related) prefixes and case, two records with the same non-empty DOI are
   linked as the same publication object (even if PMIDs differ).

2. **Same normalized title + same non-null publication year (strong pairwise
   link)** — Title is lowercased, punctuation stripped, whitespace collapsed.
   Same string + same **non-null** ``publication_year``. Records with **missing
   year** never use this rule (they may still link via fuzzy below).

3. **Fuzzy title + optional abstract check (weaker pairwise link)** — Within
   **blocks** (see below), ``rapidfuzz`` ``ratio`` on normalized titles ≥ 90, or
   ``token_sort_ratio`` ≥ 92. If **both** abstracts exist, we also require
   ``token_sort_ratio`` ≥ 78 on abstracts. If either abstract is missing, we rely
   on title similarity only.

**Where we draw the line**
- We **do not** merge on PMID alone (different PMIDs are expected for true
  duplicates in PubMed).
- We **do not** claim retracted vs replacement automatically; we **flag**
  publication types / titles containing “retract” for reviewers.
- Thresholds favor **precision** over recall: some true duplicates may be
  missed; reported pairs are meant to be **reviewed**.

**Scalability (~10k articles)**
- DOI and exact (title, year) grouping is **O(n)**.
- Fuzzy comparisons are **not** all-pairs: we **block** by ``(year, first few
  title tokens)`` so only similar-sized cohorts are compared. Records with **no
  publication year** use **prefix + length band + title head** (not a hash of the
  full title) so near-duplicate titles can still be compared. Buckets larger than
  ``MAX_BLOCK`` are split by length bands.

**Transitivity**
- Union-find merges **transitively** (A–B and B–C ⇒ one cluster). That is why a
  cluster can mix **edge rule types**: it is the union of pairwise links, not a
  single rule applied to every pair. For **fuzzy** edges, transitively linked
  pairs may **not** pass the fuzzy threshold **directly**; ``edge_evidence`` lists
  which pairs were actually linked and how.

**Blocking caveat**
- Blocking uses the **first few title words**, so titles that share a generic
  lead-in (“A case report of…”) may be **missed** (precision-first tradeoff).

**Output**
- Clusters with PMIDs, primary reason, **edge evidence**, optional transitivity
  notes, reviewer flags. A short ``methodology`` string is included for audit.
"""

from __future__ import annotations

import re
from pathlib import Path
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from article_miner.domain.article import Article, CollectionOutput

# --- Thresholds (tune for precision vs recall) ---
FUZZY_TITLE_RATIO_MIN = 90
FUZZY_TITLE_TOKEN_SORT_MIN = 92
ABSTRACT_TOKEN_SORT_MIN = 78
MAX_BLOCK_SIZE = 280
_TITLE_PREFIX_WORDS = 4
# Missing-year blocks: prefix + length band + title head (not full-title hash)
_NY_LEN_BAND = 20
_NY_HEAD_CHARS = 40

_EDGE_SAME_DOI = "same_doi"
_EDGE_SAME_TITLE_YEAR = "same_title_year"
_EDGE_FUZZY = "fuzzy"


class DuplicateCluster(BaseModel):
    """One connected component of duplicate candidates."""

    cluster_id: int
    pmids: list[str] = Field(description="PubMed IDs in this group (sorted)")
    primary_reason: str = Field(
        description="Dominant link type in the cluster (for reviewer orientation)"
    )
    confidence: Literal["high", "medium"]
    detail: str = Field(
        default="",
        description="Short explanation of why these were grouped",
    )
    edge_evidence: list[str] = Field(
        default_factory=list,
        description="Pairwise links actually used (not every pair may appear)",
    )
    transitivity_note: str | None = Field(
        None,
        description="When transitive fuzzy or mixed edge types affect interpretation",
    )
    reviewer_notes: list[str] = Field(
        default_factory=list,
        description="Flags such as possible retraction (not definitive)",
    )


class DedupReport(BaseModel):
    """Full report for JSON export."""

    source_article_count: int
    duplicate_group_count: int
    articles_in_some_duplicate_group: int
    methodology: str = Field(description="How duplicates were defined (for audit)")
    clusters: list[DuplicateCluster]
    stats: dict[str, int | float] = Field(
        default_factory=dict,
        description="Diagnostics (e.g. fuzzy pairs compared)",
    )


def format_dedup_markdown(report: DedupReport) -> str:
    """Human-readable Markdown summary for duplicate clusters (reviewers)."""
    lines = [
        "# Probable duplicate groups",
        "",
        f"- Source articles: **{report.source_article_count}**",
        f"- Duplicate groups (size ≥ 2): **{report.duplicate_group_count}**",
        f"- Articles appearing in some group: **{report.articles_in_some_duplicate_group}**",
        f"- Fuzzy pair comparisons: **{report.stats.get('fuzzy_pairs_compared', 0)}**",
        "",
        "## Definition (summary)",
        "",
        report.methodology,
        "",
        "## Groups",
        "",
    ]
    for c in report.clusters:
        lines.append(f"### Cluster {c.cluster_id} — `{c.primary_reason}` ({c.confidence})")
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
    """Lowercase DOI, strip common URL prefixes and trailing junk. ``None`` if missing."""
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
    s = s.strip()
    s = s.rstrip(".,;)")
    return s or None


def normalize_title(title: str | None) -> str:
    """Lowercase, strip punctuation, collapse whitespace (NFKD)."""
    if not title:
        return ""
    t = unicodedata.normalize("NFKD", title)
    t = t.lower()
    t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _title_prefix_key(norm: str) -> str:
    if not norm:
        return ""
    parts = norm.split()[:_TITLE_PREFIX_WORDS]
    return " ".join(parts)


def _missing_year_block_key(nt: str, prefix: str) -> str:
    """Block key when ``publication_year`` is missing: prefix + length band + title head."""
    ln = len(nt)
    band = ln // _NY_LEN_BAND
    head = nt[:_NY_HEAD_CHARS] if nt else ""
    return f"ny|{prefix}|b{band}|{head}"


def _pmid_sort_key(pmid: str) -> tuple[int, int | str]:
    """Sort PMIDs numerically when possible; otherwise lexicographic."""
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
                f"PMID {a.pmid}: publication type includes “{pt}” (verify replacement)"
            )
            break
    return notes


def _cluster_metadata(
    indices: list[int],
    articles: list[Article],
    edges_in_cluster: list[tuple[int, int, str]],
) -> tuple[str, Literal["high", "medium"], str, list[str], str | None]:
    """Labels, detail text, formatted edge lines, and transitivity note from recorded edges."""
    kinds = {k for _, _, k in edges_in_cluster}
    evidence: list[str] = []
    label_map = {
        _EDGE_SAME_DOI: "same_doi",
        _EDGE_SAME_TITLE_YEAR: "same_normalized_title_and_year",
        _EDGE_FUZZY: "fuzzy_title",
    }
    for i, j, k in sorted(
        edges_in_cluster,
        key=lambda e: (_pmid_sort_key(articles[e[0]].pmid), _pmid_sort_key(articles[e[1]].pmid)),
    ):
        pair = sorted([articles[i].pmid, articles[j].pmid], key=_pmid_sort_key)
        evidence.append(f"{pair[0]}↔{pair[1]}: {label_map.get(k, k)}")

    trans_parts: list[str] = []
    if len(indices) > 2 and _EDGE_FUZZY in kinds:
        trans_parts.append(
            "Union-find can join records transitively via fuzzy links; not every pair may meet the fuzzy threshold directly."
        )
    if len(kinds) > 1:
        trans_parts.append(
            "This cluster mixes link types; see pairwise evidence—do not assume all PMIDs share the same relationship."
        )
    trans_note = " ".join(trans_parts) if trans_parts else None

    if len(kinds) > 1:
        primary = "mixed_evidence"
        conf: Literal["high", "medium"] = "medium" if _EDGE_FUZZY in kinds else "high"
        detail = "Cluster contains more than one link type; use edge_evidence for ground truth."
    elif _EDGE_SAME_DOI in kinds:
        primary = "same_doi"
        conf = "high"
        detail = "Same normalized DOI across different PMIDs."
    elif _EDGE_SAME_TITLE_YEAR in kinds:
        primary = "same_normalized_title_and_year"
        conf = "high"
        detail = "Identical normalized title and same non-null publication year."
    else:
        primary = "fuzzy_title_and_or_abstract"
        conf = "medium"
        detail = "Similar title (and abstract when present) via fuzzy matching within blocks."

    return primary, conf, detail, evidence, trans_note


def build_duplicate_report(collection: CollectionOutput) -> DedupReport:
    """Cluster articles using DOI, exact (title, year), then blocked fuzzy pairs."""
    articles = collection.articles
    n = len(articles)
    uf = _UnionFind.new(n)
    edges: list[tuple[int, int, str]] = []
    fuzzy_pairs_compared = 0

    # 1) Same DOI
    doi_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, a in enumerate(articles):
        d = normalize_doi(a.doi)
        if d:
            doi_to_indices[d].append(i)
    for group in doi_to_indices.values():
        if len(group) < 2:
            continue
        for ii in range(len(group)):
            for jj in range(ii + 1, len(group)):
                a, b = group[ii], group[jj]
                _append_edge(edges, a, b, _EDGE_SAME_DOI)
                uf.union(a, b)

    # 2) Exact (normalized title, year) — require non-empty title and non-null year
    exact_key: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, a in enumerate(articles):
        nt = normalize_title(a.title)
        if not nt or a.publication_year is None:
            continue
        exact_key[(nt, a.publication_year)].append(i)
    for group in exact_key.values():
        if len(group) < 2:
            continue
        for ii in range(len(group)):
            for jj in range(ii + 1, len(group)):
                a, b = group[ii], group[jj]
                _append_edge(edges, a, b, _EDGE_SAME_TITLE_YEAR)
                uf.union(a, b)

    # 3) Fuzzy blocking
    block: dict[str, list[int]] = defaultdict(list)
    for i, a in enumerate(articles):
        nt = normalize_title(a.title)
        if len(nt) < 12:
            continue
        y = a.publication_year
        prefix = _title_prefix_key(nt)
        if not prefix:
            continue
        if y is None:
            key = _missing_year_block_key(nt, prefix)
        else:
            key = f"{y}|{prefix}"
        block[key].append(i)

    def split_oversized(bucket: list[int]) -> list[list[int]]:
        if len(bucket) <= MAX_BLOCK_SIZE:
            return [bucket]
        # Split by length quantile bands to keep comparisons local
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

    for _key, bucket in block.items():
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
                    # Length guard: wildly different lengths rarely duplicates
                    li, lj = len(ti), len(tj)
                    if li > 20 and lj > 20:
                        shorter, longer = min(li, lj), max(li, lj)
                        if shorter < longer * 0.55:
                            continue

                    fuzzy_pairs_compared += 1
                    tr = fuzz.ratio(ti, tj)
                    ts = fuzz.token_sort_ratio(ti, tj)
                    title_ok = tr >= FUZZY_TITLE_RATIO_MIN or ts >= FUZZY_TITLE_TOKEN_SORT_MIN
                    if not title_ok:
                        continue

                    abi = _abstract_norm(ai.abstract)
                    abj = _abstract_norm(aj.abstract)
                    if abi and abj:
                        if fuzz.token_sort_ratio(abi, abj) < ABSTRACT_TOKEN_SORT_MIN:
                            continue

                    _append_edge(edges, i, j, _EDGE_FUZZY)
                    uf.union(i, j)

    # Collect components of size >= 2
    comp: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        comp[uf.find(i)].append(i)

    # Index edges by union-find root (O(|E|)); avoids scanning all edges per cluster.
    edges_by_root: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    for i, j, k in edges:
        root = uf.find(i)
        edges_by_root[root].append((i, j, k))

    clusters: list[DuplicateCluster] = []
    cid = 0
    in_group = 0

    for root, indices in sorted(comp.items(), key=lambda x: min(x[1])):
        if len(indices) < 2:
            continue
        indices.sort(key=lambda i: _pmid_sort_key(articles[i].pmid))
        pmids = [articles[i].pmid for i in indices]
        edges_in_cluster = edges_by_root.get(root, [])
        primary, conf, detail, edge_lines, trans_note = _cluster_metadata(
            indices, articles, edges_in_cluster
        )
        notes: list[str] = []
        for i in indices:
            notes.extend(_maybe_retraction_notes(articles[i]))
        # Dedupe note strings
        seen: set[str] = set()
        uniq_notes = []
        for note in notes:
            if note not in seen:
                seen.add(note)
                uniq_notes.append(note)

        cid += 1
        in_group += len(indices)
        clusters.append(
            DuplicateCluster(
                cluster_id=cid,
                pmids=sorted(pmids, key=lambda p: _pmid_sort_key(p)),
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
        "Medium: fuzzy title (ratio≥90 or token_sort≥92) within blocks; missing-year rows use "
        "prefix + length band + title head (not a full-title hash). "
        f"If both abstracts exist, token_sort≥{ABSTRACT_TOKEN_SORT_MIN} on abstract text. "
        "Clusters list pairwise edge_evidence; transitive union-find can connect fuzzy-only chains. "
        "Retractions are flagged heuristically, not merged automatically with replacements."
    )

    return DedupReport(
        source_article_count=n,
        duplicate_group_count=len(clusters),
        articles_in_some_duplicate_group=in_group,
        methodology=methodology,
        clusters=clusters,
        stats={
            "fuzzy_pairs_compared": fuzzy_pairs_compared,
            "blocks_used": len(block),
        },
    )


def load_collection(path: str) -> CollectionOutput:
    """Load JSON written by ``collect-pubmed``."""
    return CollectionOutput.model_validate_json(Path(path).read_text(encoding="utf-8"))
