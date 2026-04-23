"""Pydantic models for ESearch JSON (retmode=json)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ESearchInner(BaseModel):
    model_config = ConfigDict(extra="ignore")

    count: str
    idlist: list[str] = Field(default_factory=list)

    @field_validator("idlist", mode="before")
    @classmethod
    def coerce_idlist(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            return [str(x) for x in v]
        return [str(v)]


class ESearchEnvelope(BaseModel):
    model_config = ConfigDict(extra="ignore")

    esearchresult: ESearchInner
