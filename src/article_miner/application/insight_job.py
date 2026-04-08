"""Async insight classification job (Pass 1 → 2 → optional 3)."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from article_miner.domain.collect.models import Article, CollectionOutput
from article_miner.domain.insights.models import (
    ArticleInsightRecord,
    AuditResult,
    AutoAcceptStatus,
    InsightJobResult,
    LlmInsightExtraction,
    PerArticleInsightResult,
    PerArticleStatus,
)
from article_miner.infrastructure.insights.canonical_text import build_canonical_text
from article_miner.infrastructure.insights.insight_cache import InsightCache, cache_key
from article_miner.infrastructure.insights.insight_validation import merge_dict_for_audit, parse_extraction_json, run_pass2_validation
from article_miner.infrastructure.insights.llm_extract import (
    LlmCallStats,
    audit_classification,
    audit_triggers,
    extract_insight_json,
    repair_json,
)
from article_miner.infrastructure.insights.prefilter import prefilter_article
from article_miner.infrastructure.insights.prompts import PROMPT_VERSION

logger = logging.getLogger(__name__)


@dataclass
class InsightJobConfig:
    model: str
    audit_model: str | None = None
    confidence_threshold: float = 0.5
    concurrency: int = 8
    max_retries: int = 3
    enable_audit: bool = True
    cache_path: Path | None = None
    #: If set, warn when title+abstract (canonical haystack) exceeds this length.
    max_canonical_chars: int | None = 12_000
    extra_completion_kwargs: dict[str, Any] = field(default_factory=dict)


class InsightClassificationJob:
    """Run Pass 1–3 over a collection with bounded concurrency."""

    def __init__(self, config: InsightJobConfig) -> None:
        self._config = config
        self._audit_model = config.audit_model or config.model
        self._cache = InsightCache(config.cache_path)

    async def run(self, collection: CollectionOutput) -> InsightJobResult:
        sem = asyncio.Semaphore(self._config.concurrency)
        totals: dict[str, float] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "success_trusted": 0,
            "needs_review": 0,
            "invalid_output": 0,
            "api_failure": 0,
            "skipped_prefilter": 0,
            "truncation_warning": 0,
        }

        async def _one(a: Article) -> PerArticleInsightResult:
            async with sem:
                return await self._process_article(a, totals)

        results = list(await asyncio.gather(*[_one(a) for a in collection.articles]))

        self._cache.close()
        return InsightJobResult(
            prompt_version=PROMPT_VERSION,
            model=self._config.model,
            source_query=collection.query,
            articles=results,
            stats=totals,
        )

    async def _process_article(self, article: Article, totals: dict[str, float]) -> PerArticleInsightResult:
        skip = prefilter_article(article)
        if skip:
            totals["skipped_prefilter"] += 1
            return PerArticleInsightResult(
                pmid=article.pmid,
                status=PerArticleStatus.SKIPPED_PREFILTER,
                prefilter_note=skip,
            )

        ck = cache_key(article, self._config.model)
        cached = self._cache.get(ck)
        if cached:
            ext, err = parse_extraction_json(cached)
            if ext:
                return await self._validate_and_finalize(article, ext, totals, raw_llm_text=cached)
            logger.warning("Cache parse failed for PMID %s", article.pmid)

        kw = self._config.extra_completion_kwargs
        last_err: str | None = None
        raw_text = ""
        for attempt in range(self._config.max_retries):
            try:
                raw_text, st = await extract_insight_json(self._config.model, article, **kw)
                totals["input_tokens"] += st.input_tokens
                totals["output_tokens"] += st.output_tokens
                break
            except Exception as exc:
                last_err = str(exc)
                logger.debug("extract attempt %s failed: %s", attempt + 1, exc)
                await asyncio.sleep(0.5 * (2**attempt))
        else:
            totals["api_failure"] += 1
            return PerArticleInsightResult(
                pmid=article.pmid,
                status=PerArticleStatus.API_FAILURE,
                error_message=last_err,
            )

        ext, err = parse_extraction_json(raw_text)
        if not ext:
            try:
                repaired, st = await repair_json(self._config.model, raw_text, **kw)
                totals["input_tokens"] += st.input_tokens
                totals["output_tokens"] += st.output_tokens
                ext, err = parse_extraction_json(repaired)
                raw_text = repaired
            except Exception as exc:
                last_err = str(exc)

        if not ext:
            totals["invalid_output"] += 1
            return PerArticleInsightResult(
                pmid=article.pmid,
                status=PerArticleStatus.INVALID_OUTPUT,
                error_message="; ".join(err) if err else last_err,
                raw_llm_text=raw_text[:8000],
            )

        self._cache.set(ck, raw_text)
        return await self._validate_and_finalize(article, ext, totals, raw_llm_text=raw_text)

    async def _validate_and_finalize(
        self,
        article: Article,
        ext: LlmInsightExtraction,
        totals: dict[str, float],
        *,
        raw_llm_text: str,
    ) -> PerArticleInsightResult:
        trunc = False
        limit = self._config.max_canonical_chars
        if limit is not None and len(build_canonical_text(article)) > limit:
            trunc = True
            totals["truncation_warning"] += 1
            logger.info(
                "PMID %s canonical text length exceeds max_canonical_chars=%s",
                article.pmid,
                limit,
            )

        val, sem_flags, auto, reasons = run_pass2_validation(
            article,
            ext,
            confidence_threshold=self._config.confidence_threshold,
            truncation_warning=trunc,
        )

        low_conf = any(
            [
                ext.finding_direction.confidence < self._config.confidence_threshold,
                ext.statistical_significance.confidence < self._config.confidence_threshold,
                ext.clinical_meaningfulness.confidence < self._config.confidence_threshold,
                ext.main_claim.confidence < self._config.confidence_threshold,
            ]
        )
        mixed = ext.finding_direction.value.lower() == "mixed"
        meaningful = ext.clinical_meaningfulness.value.lower() == "meaningful"
        grounding_failed = any(not g.all_spans_found for g in val.grounding)
        sem_bool = len(sem_flags) > 0

        audit_res: AuditResult | None = None
        run_audit = self._config.enable_audit and audit_triggers(
            low_confidence=low_conf,
            mixed_findings=mixed,
            clinically_meaningful=meaningful,
            grounding_failed=grounding_failed,
            semantic_flags=sem_bool,
        )
        if run_audit:
            try:
                audit_res, st = await audit_classification(
                    self._audit_model,
                    article,
                    merge_dict_for_audit(ext),
                    **self._config.extra_completion_kwargs,
                )
                totals["input_tokens"] += st.input_tokens
                totals["output_tokens"] += st.output_tokens
            except Exception as exc:
                audit_res = AuditResult(supported=False, notes=f"audit_error:{exc}")

        trusted = (
            auto == AutoAcceptStatus.AUTO_ACCEPT
            and val.schema_ok
            and all(g.all_spans_found for g in val.grounding)
            and (audit_res is None or audit_res.supported)
        )

        if trusted:
            st_final = PerArticleStatus.SUCCESS
            totals["success_trusted"] += 1
        else:
            st_final = PerArticleStatus.NEEDS_HUMAN_REVIEW
            totals["needs_review"] += 1

        record = ArticleInsightRecord(
            pmid=article.pmid,
            extraction=ext,
            validation=val,
            auto_accept=auto,
            audit=audit_res,
            acceptance_reasons=reasons,
        )
        return PerArticleInsightResult(
            pmid=article.pmid,
            status=st_final,
            insight=record,
            raw_llm_text=raw_llm_text[:4000] if st_final == PerArticleStatus.INVALID_OUTPUT else None,
        )


async def run_insight_job(
    collection: CollectionOutput,
    config: InsightJobConfig,
) -> InsightJobResult:
    job = InsightClassificationJob(config)
    return await job.run(collection)
