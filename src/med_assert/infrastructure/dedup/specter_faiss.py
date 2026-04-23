"""SPECTER 2 embeddings + FAISS cosine similarity for scientific duplicate edges."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import numpy as np

from med_assert.domain.collect.models import Article

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Default: Allen AI SPECTER2 base encoder (sentence-transformers compatible).
DEFAULT_SPECTER_MODEL = os.environ.get(
    "MED_ASSERT_SPECTER_MODEL", "allenai/specter2_base"
)

# Cosine similarity threshold (embeddings are L2-normalized; inner product = cosine).
DEFAULT_SPECTER_SIM_THRESHOLD = float(os.environ.get("MED_ASSERT_SPECTER_THRESHOLD", "0.92"))

# Neighbors to inspect per article (CPU FAISS flat index).
DEFAULT_SPECTER_FAISS_K = int(os.environ.get("MED_ASSERT_SPECTER_K", "32"))


def specter_document_text(article: Article) -> str:
    """Concatenate title and abstract (SPECTER-style single field)."""
    title = (article.title or "").strip()
    abstract = (article.abstract or "").strip()
    if abstract:
        return f"{title} {abstract}".strip()
    return title or abstract or "empty"


def compute_specter_embeddings(
    articles: list[Article],
    *,
    model_id: str = DEFAULT_SPECTER_MODEL,
    batch_size: int = 16,
) -> tuple[np.ndarray, int]:
    """Return L2-normalized float32 matrix ``(n, dim)`` and embedding dimension."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        msg = (
            "SPECTER 2 dedup requires optional ML dependencies: "
            "pip install 'med-assert[specter]'"
        )
        raise ImportError(msg) from exc

    texts = [specter_document_text(a) for a in articles]
    model = SentenceTransformer(model_id, trust_remote_code=True)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    arr = np.asarray(emb, dtype=np.float32)
    if arr.ndim != 2:
        msg = f"Expected 2D embeddings, got shape {arr.shape}"
        raise ValueError(msg)
    return arr, arr.shape[1]


def faiss_cosine_pairs(
    embeddings: np.ndarray,
    *,
    threshold: float = DEFAULT_SPECTER_SIM_THRESHOLD,
    k_neighbors: int = DEFAULT_SPECTER_FAISS_K,
) -> list[tuple[int, int, float]]:
    """Return unique pairs (i, j) with i < j and cosine similarity >= threshold."""
    import faiss

    n, d = embeddings.shape
    if n < 2:
        return []
    k_neighbors = max(2, min(k_neighbors, n))
    index = faiss.IndexFlatIP(d)
    index.add(embeddings.astype(np.float32, copy=False))
    sims, indices = index.search(embeddings.astype(np.float32, copy=False), k_neighbors)
    pairs: list[tuple[int, int, float]] = []
    seen: set[tuple[int, int]] = set()
    for i in range(n):
        for j_idx in range(k_neighbors):
            j = int(indices[i, j_idx])
            if j <= i:
                continue
            s = float(sims[i, j_idx])
            if s < threshold:
                continue
            key = (i, j)
            if key in seen:
                continue
            seen.add(key)
            pairs.append((i, j, s))
    pairs.sort(key=lambda t: (-t[2], t[0], t[1]))
    return pairs
