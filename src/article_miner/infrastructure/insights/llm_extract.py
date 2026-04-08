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

litellm.suppress_debug_info = True


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
        {"role": "system", "content": "You reply with JSON only: {\"supported\": bool, \"notes\": string}"},
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
    try:
        data = json.loads(text)
        supported = bool(data.get("supported", False))
        notes = str(data.get("notes", ""))
    except json.JSONDecodeError:
        supported = False
        notes = "audit_parse_failed"
    return AuditResult(supported=supported, notes=notes, raw_response=text), LlmCallStats(
        input_tokens=inp, output_tokens=out, model=model
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
