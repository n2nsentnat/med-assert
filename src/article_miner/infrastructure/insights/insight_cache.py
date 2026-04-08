"""Optional SQLite cache for LLM extractions (keyed by PMID + input hash + prompt + model)."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from article_miner.domain.collect.models import Article
from article_miner.infrastructure.insights.canonical_text import build_canonical_text
from article_miner.infrastructure.insights.prompts import PROMPT_VERSION


def cache_key(article: Article, model: str) -> str:
    body = f"{article.pmid}|{PROMPT_VERSION}|{model}|{build_canonical_text(article)}"
    return hashlib.sha256(body.encode()).hexdigest()


class InsightCache:
    def __init__(self, path: Path | None) -> None:
        self._path = path
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(path), check_same_thread=False)
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT NOT NULL)"
            )
            self._conn.commit()
        else:
            self._conn = None

    def get(self, key: str) -> str | None:
        if not self._conn:
            return None
        row = self._conn.execute("SELECT v FROM cache WHERE k = ?", (key,)).fetchone()
        return row[0] if row else None

    def set(self, key: str, value: str) -> None:
        if not self._conn:
            return
        self._conn.execute(
            "INSERT OR REPLACE INTO cache (k, v) VALUES (?, ?)",
            (key, value),
        )
        self._conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
