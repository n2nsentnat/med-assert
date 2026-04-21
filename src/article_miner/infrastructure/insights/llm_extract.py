"""Pass 1 + repair + audit via LangChain chat models (async)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from article_miner.domain.collect.models import Article
from article_miner.domain.insights.models import AuditResult
from article_miner.infrastructure.insights.prompts import (
    AUDIT_USER_TEMPLATE,
    REPAIR_USER_TEMPLATE,
    SYSTEM_PROMPT,
    build_user_prompt,
    system_prompt,
)

logger = logging.getLogger(__name__)

_AUDIT_VALUES = {"supported", "weakly_supported", "unsupported"}


@dataclass
class LlmCallStats:
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""


def _message_content(msg: BaseMessage) -> str:
    c = msg.content
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: list[str] = []
        for block in c:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(c)


def _usage_from_message(msg: AIMessage) -> tuple[int, int]:
    um = getattr(msg, "usage_metadata", None)
    if isinstance(um, dict):
        inp = int(um.get("input_tokens", 0) or um.get("prompt_tokens", 0) or 0)
        out = int(um.get("output_tokens", 0) or um.get("completion_tokens", 0) or 0)
        if inp or out:
            return inp, out
    meta = getattr(msg, "response_metadata", None) or {}
    tu = meta.get("token_usage") or meta.get("usage") or {}
    if isinstance(tu, dict):
        inp = int(
            tu.get("prompt_tokens")
            or tu.get("input_tokens")
            or tu.get("input_token_count")
            or 0
        )
        out = int(
            tu.get("completion_tokens")
            or tu.get("output_tokens")
            or tu.get("output_token_count")
            or 0
        )
        return inp, out
    return 0, 0


def _bind_json_mode(chat_model: BaseChatModel) -> BaseChatModel:
    """Prefer provider-native JSON outputs when available."""
    if isinstance(chat_model, ChatOpenAI):
        return chat_model.bind(response_format={"type": "json_object"})  # type: ignore[return-value]
    if isinstance(chat_model, ChatGoogleGenerativeAI):
        try:
            return chat_model.bind(response_mime_type="application/json")  # type: ignore[return-value]
        except (TypeError, ValueError):
            return chat_model
    if isinstance(chat_model, ChatAnthropic):
        try:
            return chat_model.bind(response_format={"type": "json_object"})  # type: ignore[return-value]
        except (TypeError, ValueError):
            return chat_model
    return chat_model


async def extract_insight_json(
    chat_model: BaseChatModel,
    article: Article,
    *,
    display_name: str = "",
) -> tuple[str, LlmCallStats]:
    """Single extraction call; returns raw message content (JSON string)."""
    runnable = _bind_json_mode(chat_model)
    messages = [
        SystemMessage(content=system_prompt()),
        HumanMessage(content=build_user_prompt(article)),
    ]
    response = await runnable.ainvoke(messages)
    if not isinstance(response, AIMessage):
        response = AIMessage(content=_message_content(response))  # type: ignore[arg-type]
    text = _message_content(response)
    inp, out = _usage_from_message(response)
    return text, LlmCallStats(input_tokens=inp, output_tokens=out, model=display_name)


async def repair_json(
    chat_model: BaseChatModel,
    broken_text: str,
    *,
    display_name: str = "",
) -> tuple[str, LlmCallStats]:
    """One repair attempt for malformed JSON."""
    runnable = _bind_json_mode(chat_model)
    messages = [
        SystemMessage(content="You output only valid JSON. No markdown."),
        HumanMessage(content=REPAIR_USER_TEMPLATE.format(text=broken_text)),
    ]
    response = await runnable.ainvoke(messages)
    if not isinstance(response, AIMessage):
        response = AIMessage(content=_message_content(response))  # type: ignore[arg-type]
    text = _message_content(response)
    inp, out = _usage_from_message(response)
    return text, LlmCallStats(input_tokens=inp, output_tokens=out, model=display_name)


async def audit_classification(
    chat_model: BaseChatModel,
    article: Article,
    classification: dict[str, Any],
    *,
    display_name: str = "",
) -> tuple[AuditResult, LlmCallStats]:
    """Pass 3: lightweight audit JSON."""
    user = AUDIT_USER_TEMPLATE.format(
        pmid=article.pmid,
        title=article.title or "",
        abstract=article.abstract or "",
        classification_json=json.dumps(classification, indent=2)[:12000],
    )
    runnable = _bind_json_mode(chat_model)
    messages = [
        SystemMessage(content="You output only valid JSON in the requested schema."),
        HumanMessage(content=user),
    ]
    response = await runnable.ainvoke(messages)
    if not isinstance(response, AIMessage):
        response = AIMessage(content=_message_content(response))  # type: ignore[arg-type]
    text = _message_content(response) or "{}"
    inp, out = _usage_from_message(response)
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
    return audit, LlmCallStats(input_tokens=inp, output_tokens=out, model=display_name)


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
