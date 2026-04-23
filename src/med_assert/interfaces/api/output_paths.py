"""Resolve default and explicit server paths for API file output."""

from __future__ import annotations

from pathlib import Path

from med_assert.interfaces.api.schemas import (
    DEFAULT_API_OUTPUT_SUBDIR,
    InsightFileFormat,
)


def _base_dir() -> Path:
    return Path.cwd()


def resolve_collect_path(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    return (_base_dir() / DEFAULT_API_OUTPUT_SUBDIR / "collect.json").resolve()


def resolve_dedup_path(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    return (_base_dir() / DEFAULT_API_OUTPUT_SUBDIR / "dedup.json").resolve()


def resolve_insight_path(explicit: str | None, fmt: InsightFileFormat) -> Path:
    if explicit:
        p = Path(explicit).expanduser()
        suf = p.suffix.lower()
        if suf not in (".json", ".jsonl"):
            p = p.with_suffix(".jsonl" if fmt == "jsonl" else ".json")
        return p.resolve()
    name = "insights.jsonl" if fmt == "jsonl" else "insights.json"
    return (_base_dir() / DEFAULT_API_OUTPUT_SUBDIR / name).resolve()


def is_jsonl_path(path: Path) -> bool:
    return path.suffix.lower() == ".jsonl"
