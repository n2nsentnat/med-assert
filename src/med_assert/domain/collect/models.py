"""Structured article and collection output models (domain + JSON schema)."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class Author(BaseModel):
    """Single author entry with optional affiliation."""

    last_name: str | None = None
    fore_name: str | None = None
    initials: str | None = None
    affiliation: str | None = None


class Article(BaseModel):
    """Flat, JSON-friendly PubMed article record."""

    pmid: str = Field(..., description="PubMed ID")
    title: str | None = None
    abstract: str | None = None
    journal_full: str | None = Field(None, description="Journal full title")
    journal_iso: str | None = Field(None, description="ISO journal abbreviation")
    publication_year: int | None = None
    publication_month: int | None = Field(
        None,
        description="Calendar month 1-12 when parseable (from PubMed Year/Month/Day)",
        ge=1,
        le=12,
    )
    publication_day: int | None = None
    doi: str | None = None
    pmc_id: str | None = None
    language: str | None = None
    publication_types: list[str] = Field(default_factory=list)
    mesh_terms: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    authors: list[Author] = Field(default_factory=list)

    @field_validator("pmid", mode="before")
    @classmethod
    def coerce_pmid(cls, v: object) -> str:
        return str(v)


class CollectionOutput(BaseModel):
    """Top-level JSON document written for downstream consumers."""

    query: str
    total_match_count: int = Field(
        ..., description="Total hits for the query in PubMed"
    )
    requested_count: int
    retrieved_count: int = Field(
        ..., description="Articles successfully parsed and included"
    )
    articles: list[Article]
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal issues (e.g. skipped PMIDs, partial pages)",
    )
