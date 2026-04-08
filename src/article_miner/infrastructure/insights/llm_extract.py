"""Pass 1 + repair + audit via LiteLLM (async)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import litellm

from article_miner.domain.collect.models import Article
from article_miner.domain.insights.models import AuditResult
from article_miner.infrastructure.insights.prompts import (
    AUDIT_USER_TEMPLATE,
    PROMPT_VERSION,
    REPAIR_USER_TEMPLATE,
    SYSTEM_PROMPT,
    build_user_prompt,
    system_prompt,
)

logger = logging.getLogger(__name__)

# Keep LiteLLM internals quiet; rely on application-level progress logs.
litellm.suppress_debug_info = True

_AUDIT_VALUES = {"supported", "weakly_supported", "unsupported"}


@dataclass
class LlmCallStats:
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""


def _usage_tokens(response: Any) -> tuple[int, int]:
    u = getattr(response, "usage", None)
    if u is None:
        return 0, 0
    inp = getattr(u, "prompt_tokens", None) or getattr(u, "input_tokens", None) or 0
    out = getattr(u, "completion_tokens", None) or getattr(u, "output_tokens", None) or 0
    return int(inp), int(out)


async def extract_insight_json(model: str, article: Article, **kwargs: Any) -> tuple[str, LlmCallStats]:
    """Single extraction call; returns raw message content (JSON string)."""
    messages = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": build_user_prompt(article)},
    ]
    response = await litellm.acompletion(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        **kwargs,
    )
    text = response.choices[0].message.content or ""
    inp, out = _usage_tokens(response)
    return text, LlmCallStats(input_tokens=inp, output_tokens=out, model=model)


async def repair_json(model: str, broken_text: str, **kwargs: Any) -> tuple[str, LlmCallStats]:
    """One repair attempt for malformed JSON."""
    messages = [
        {"role": "system", "content": "You output only valid JSON. No markdown."},
        {"role": "user", "content": REPAIR_USER_TEMPLATE.format(text=broken_text)},
    ]
    response = await litellm.acompletion(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        **kwargs,
    )
    text = response.choices[0].message.content or ""
    inp, out = _usage_tokens(response)
    return text, LlmCallStats(input_tokens=inp, output_tokens=out, model=model)


async def audit_classification(
    model: str,
    article: Article,
    classification: dict[str, Any],
    **kwargs: Any,
) -> tuple[AuditResult, LlmCallStats]:
    """Pass 3: lightweight audit JSON."""
    user = AUDIT_USER_TEMPLATE.format(
        pmid=article.pmid,
        title=article.title or "",
        abstract=article.abstract or "",
        classification_json=json.dumps(classification, indent=2)[:12000],
    )
    messages = [
        {"role": "system", "content": "You output only valid JSON in the requested schema."},
        {"role": "user", "content": user},
    ]
    response = await litellm.acompletion(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        **kwargs,
    )
    text = response.choices[0].message.content or "{}"
    inp, out = _usage_tokens(response)
    audit = parse_audit_json(text)
    if audit is None:
        audit = AuditResult(
            supported=False,
            finding_direction="unsupported",
            statistical_significance="unsupported",
            clinical_meaningfulness="unsupported",
            main_claim="unsupported",
            notes=["audit_parse_failed"],
            raw_response=text,
        )
    return audit, LlmCallStats(
        input_tokens=inp, output_tokens=out, model=model
    )


def parse_audit_json(text: str) -> AuditResult | None:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    def _coerce_verdict(key: str) -> str:
        v = str(data.get(key, "unsupported")).strip().lower()
        return v if v in _AUDIT_VALUES else "unsupported"

    raw_notes = data.get("notes", [])
    notes: list[str]
    if isinstance(raw_notes, list):
        notes = [str(n).strip() for n in raw_notes if str(n).strip()]
    elif raw_notes is None:
        notes = []
    else:
        one = str(raw_notes).strip()
        notes = [one] if one else []

    return AuditResult(
        supported=bool(data.get("supported", False)),
        finding_direction=_coerce_verdict("finding_direction"),  # type: ignore[arg-type]
        statistical_significance=_coerce_verdict("statistical_significance"),  # type: ignore[arg-type]
        clinical_meaningfulness=_coerce_verdict("clinical_meaningfulness"),  # type: ignore[arg-type]
        main_claim=_coerce_verdict("main_claim"),  # type: ignore[arg-type]
        notes=notes,
        raw_response=text,
    )


def audit_triggers(
    *,
    low_confidence: bool,
    mixed_findings: bool,
    clinically_meaningful: bool,
    grounding_failed: bool,
    semantic_flags: bool,
) -> bool:
    return (
        low_confidence
        or mixed_findings
        or clinically_meaningful
        or grounding_failed
        or semantic_flags
    )
