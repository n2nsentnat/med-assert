"""Async insight classification job (Pass 1 → 2 → optional 3)."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from langchain_core.language_models.chat_models import BaseChatModel

from article_miner.domain.collect.models import Article, CollectionOutput
from article_miner.domain.insights.models import (
    ArticleInsightRecord,
    AuditResult,
    AutoAcceptStatus,
    ClinicalFieldInsight,
    FieldInsightBlock,
    InsightJobResult,
    LlmInsightExtraction,
    PerArticleInsightResult,
    PerArticleStatus,
    ValidationPassResult,
)
from article_miner.infrastructure.insights.canonical_text import build_canonical_text
from article_miner.infrastructure.insights.insight_cache import (
    InsightCache,
    cache_key,
    input_hash,
)
from article_miner.infrastructure.insights.insight_validation import (
    VALIDATOR_VERSION,
    merge_dict_for_audit,
    parse_extraction_json,
    run_pass2_validation,
    try_local_json_repair,
)
from article_miner.infrastructure.insights.llm_extract import (
    LlmCallStats,
    audit_classification,
    audit_triggers,
    extract_insight_json,
    repair_json,
)
from article_miner.infrastructure.insights.prefilter import (
    PrefilterAction,
    prefilter_article,
)
from article_miner.infrastructure.insights.prompts import PROMPT_VERSION

logger = logging.getLogger(__name__)


@dataclass
class InsightJobConfig:
    """``model`` is a display id for JSON output; LLM calls use ``chat_model``."""

    model: str
    chat_model: BaseChatModel
    audit_chat_model: BaseChatModel | None = None
    confidence_threshold: float = 0.5
    concurrency: int = 8
    max_retries: int = 3
    enable_audit: bool = True
    cache_path: Path | None = None
    incremental_jsonl_path: Path | None = None
    progress: bool = True
    progress_every: int = 1
    #: If set, warn when title+abstract (canonical haystack) exceeds this length.
    max_canonical_chars: int | None = 12_000


class InsightClassificationJob:
    """Run Pass 1–3 over a collection with bounded concurrency."""

    def __init__(self, config: InsightJobConfig) -> None:
        self._config = config
        self._audit_chat = config.audit_chat_model or config.chat_model
        self._cache = InsightCache(config.cache_path)

    async def run(self, collection: CollectionOutput) -> InsightJobResult:
        sem = asyncio.Semaphore(self._config.concurrency)
        total_articles = len(collection.articles)
        totals: dict[str, float] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "auto_accepted": 0,
            "validated_but_flagged": 0,
            "needs_review": 0,
            "invalid_output": 0,
            "api_failure": 0,
            "skipped_prefilter": 0,
            "truncation_warning": 0,
        }

        async def _one(idx: int, a: Article) -> tuple[int, PerArticleInsightResult]:
            async with sem:
                return idx, await self._process_article(a, totals)

        pending = [
            asyncio.create_task(_one(i, a)) for i, a in enumerate(collection.articles)
        ]
        indexed_results: list[tuple[int, PerArticleInsightResult]] = []
        completed = 0
        for done in asyncio.as_completed(pending):
            idx, row = await done
            indexed_results.append((idx, row))
            self._append_incremental_row(row)
            completed += 1
            if self._config.progress and (
                completed == total_articles
                or completed % max(1, self._config.progress_every) == 0
            ):
                logger.info(
                    "insights progress: %s/%s done (auto=%s flagged=%s review=%s invalid=%s api_fail=%s skipped=%s)",
                    completed,
                    total_articles,
                    int(totals.get("auto_accepted", 0)),
                    int(totals.get("validated_but_flagged", 0)),
                    int(totals.get("needs_review", 0)),
                    int(totals.get("invalid_output", 0)),
                    int(totals.get("api_failure", 0)),
                    int(totals.get("skipped_prefilter", 0)),
                )

        indexed_results.sort(key=lambda x: x[0])
        results = [row for _, row in indexed_results]

        self._cache.close()
        return InsightJobResult(
            prompt_version=PROMPT_VERSION,
            model=self._config.model,
            source_query=collection.query,
            articles=results,
            stats=totals,
        )

    def _append_incremental_row(self, row: PerArticleInsightResult) -> None:
        path = self._config.incremental_jsonl_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(row.model_dump_json() + "\n")

    async def _process_article(
        self, article: Article, totals: dict[str, float]
    ) -> PerArticleInsightResult:
        t0 = time.perf_counter()
        logger.info("insights article start: pmid=%s", article.pmid)
        decision = prefilter_article(article)
        in_hash = input_hash(article)
        if decision.action == PrefilterAction.SKIP:
            totals["skipped_prefilter"] += 1
            row = PerArticleInsightResult(
                pmid=article.pmid,
                prompt_version=PROMPT_VERSION,
                model_name=self._config.model,
                input_hash=in_hash,
                validator_version=VALIDATOR_VERSION,
                status=PerArticleStatus.SKIPPED_PREFILTER,
                prefilter_note=decision.reason,
                raw_llm_text="",
            )
            logger.info(
                "insights article done: pmid=%s status=%s route=%s elapsed_ms=%s",
                article.pmid,
                row.status,
                decision.reason,
                int((time.perf_counter() - t0) * 1000),
            )
            return row
        if decision.action == PrefilterAction.MINIMAL_UNCLEAR:
            totals["validated_but_flagged"] += 1
            row = self._build_prefilter_minimal_unclear(
                article, decision.reason, in_hash, self._config.model
            )
            logger.info(
                "insights article done: pmid=%s status=%s route=%s elapsed_ms=%s",
                article.pmid,
                row.status,
                decision.reason,
                int((time.perf_counter() - t0) * 1000),
            )
            return row

        ck = cache_key(article, self._config.model)
        cached = self._cache.get(ck)
        if cached:
            ext, err = parse_extraction_json(cached)
            if ext:
                row = await self._validate_and_finalize(
                    article, ext, totals, raw_llm_text=cached
                )
                logger.info(
                    "insights article done: pmid=%s status=%s source=cache elapsed_ms=%s",
                    article.pmid,
                    row.status,
                    int((time.perf_counter() - t0) * 1000),
                )
                return row
            logger.warning("Cache parse failed for PMID %s", article.pmid)

        last_err: str | None = None
        raw_text = ""
        for attempt in range(self._config.max_retries):
            try:
                raw_text, st = await extract_insight_json(
                    self._config.chat_model,
                    article,
                    display_name=self._config.model,
                )
                totals["input_tokens"] += st.input_tokens
                totals["output_tokens"] += st.output_tokens
                break
            except Exception as exc:
                last_err = str(exc)
                logger.warning(
                    "insights extract retry: pmid=%s attempt=%s/%s error=%s",
                    article.pmid,
                    attempt + 1,
                    self._config.max_retries,
                    exc,
                )
                await asyncio.sleep(0.5 * (2**attempt))
        else:
            totals["api_failure"] += 1
            row = PerArticleInsightResult(
                pmid=article.pmid,
                prompt_version=PROMPT_VERSION,
                model_name=self._config.model,
                input_hash=in_hash,
                validator_version=VALIDATOR_VERSION,
                status=PerArticleStatus.API_FAILURE,
                error_message=last_err,
            )
            logger.info(
                "insights article done: pmid=%s status=%s elapsed_ms=%s",
                article.pmid,
                row.status,
                int((time.perf_counter() - t0) * 1000),
            )
            return row

        ext, err = parse_extraction_json(raw_text)
        if not ext:
            locally_repaired = try_local_json_repair(raw_text)
            if locally_repaired:
                ext, err = parse_extraction_json(locally_repaired)
                if ext:
                    raw_text = locally_repaired

        if not ext:
            try:
                repaired, st = await repair_json(
                    self._config.chat_model,
                    raw_text,
                    display_name=self._config.model,
                )
                totals["input_tokens"] += st.input_tokens
                totals["output_tokens"] += st.output_tokens
                ext, err = parse_extraction_json(repaired)
                raw_text = repaired
            except Exception as exc:
                last_err = str(exc)

        if not ext:
            totals["invalid_output"] += 1
            row = PerArticleInsightResult(
                pmid=article.pmid,
                prompt_version=PROMPT_VERSION,
                model_name=self._config.model,
                input_hash=in_hash,
                validator_version=VALIDATOR_VERSION,
                status=PerArticleStatus.INVALID_OUTPUT,
                error_message="; ".join(err) if err else last_err,
                raw_llm_text=raw_text[:8000],
            )
            logger.info(
                "insights article done: pmid=%s status=%s elapsed_ms=%s",
                article.pmid,
                row.status,
                int((time.perf_counter() - t0) * 1000),
            )
            return row

        self._cache.set(ck, raw_text)
        row = await self._validate_and_finalize(
            article, ext, totals, raw_llm_text=raw_text
        )
        logger.info(
            "insights article done: pmid=%s status=%s elapsed_ms=%s",
            article.pmid,
            row.status,
            int((time.perf_counter() - t0) * 1000),
        )
        return row

    @staticmethod
    def _build_prefilter_minimal_unclear(
        article: Article, reason: str | None, in_hash: str, model_name: str
    ) -> PerArticleInsightResult:
        reason_text = reason or "skipped_prefilter_minimal_unclear"
        extraction = LlmInsightExtraction(
            pmid=article.pmid,
            finding_direction=FieldInsightBlock(
                value="unclear", confidence=0.0, evidence_spans=[]
            ),
            statistical_significance=FieldInsightBlock(
                value="unclear", confidence=0.0, evidence_spans=[]
            ),
            clinical_meaningfulness=ClinicalFieldInsight(
                value="unclear",
                confidence=0.0,
                evidence_spans=[],
                reasoning_summary=reason_text,
            ),
            main_claim=FieldInsightBlock(
                value="Insufficient abstract text for reliable classification.",
                confidence=0.0,
                evidence_spans=[],
            ),
            review_flags=[reason_text],
        )
        record = ArticleInsightRecord(
            pmid=article.pmid,
            extraction=extraction,
            validation=ValidationPassResult(schema_ok=True, truncation_warning=False),
            auto_accept=AutoAcceptStatus.NEEDS_HUMAN_REVIEW,
            audit=None,
            acceptance_reasons=["prefilter_minimal_unclear", reason_text],
        )
        return PerArticleInsightResult(
            pmid=article.pmid,
            prompt_version=PROMPT_VERSION,
            model_name=model_name,
            input_hash=in_hash,
            validator_version=VALIDATOR_VERSION,
            status=PerArticleStatus.VALIDATED_BUT_FLAGGED,
            insight=record,
            prefilter_note=reason_text,
            raw_llm_text="",
        )

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
                ext.statistical_significance.confidence
                < self._config.confidence_threshold,
                ext.clinical_meaningfulness.confidence
                < self._config.confidence_threshold,
                ext.main_claim.confidence < self._config.confidence_threshold,
            ]
        )
        mixed = ext.finding_direction.value.lower() == "mixed"
        meaningful = ext.clinical_meaningfulness.value.lower() == "meaningful"
        grounding_failed = any(not g.all_spans_found for g in val.grounding)
        sem_bool = len(sem_flags) > 0

        audit_res: AuditResult | None = None
        run_audit = self._config.enable_audit and audit_triggers(
            low_confidence=False,
            mixed_findings=mixed,
            clinically_meaningful=meaningful,
            grounding_failed=False,
            semantic_flags=sem_bool,
        )
        if run_audit:
            try:
                audit_res, st = await audit_classification(
                    self._audit_chat,
                    article,
                    merge_dict_for_audit(ext),
                    display_name=self._config.model,
                )
                totals["input_tokens"] += st.input_tokens
                totals["output_tokens"] += st.output_tokens
            except Exception as exc:
                audit_res = AuditResult(
                    supported=False,
                    finding_direction="unsupported",
                    statistical_significance="unsupported",
                    clinical_meaningfulness="unsupported",
                    main_claim="unsupported",
                    notes=[f"audit_error:{exc}"],
                )

        trusted = (
            auto == AutoAcceptStatus.AUTO_ACCEPT
            and val.schema_ok
            and all(g.all_spans_found for g in val.grounding)
            and (audit_res is None or audit_res.supported)
        )

        validated_but_flagged = (
            not trusted
            and val.schema_ok
            and all(g.all_spans_found for g in val.grounding)
            and (audit_res is None or audit_res.supported)
        )

        if grounding_failed:
            st_final = PerArticleStatus.INVALID_OUTPUT
            totals["invalid_output"] += 1
        elif trusted:
            st_final = PerArticleStatus.AUTO_ACCEPTED
            totals["auto_accepted"] += 1
        elif validated_but_flagged:
            st_final = PerArticleStatus.VALIDATED_BUT_FLAGGED
            totals["validated_but_flagged"] += 1
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
            prompt_version=PROMPT_VERSION,
            model_name=self._config.model,
            input_hash=input_hash(article),
            validator_version=VALIDATOR_VERSION,
            status=st_final,
            insight=record,
            raw_llm_text=raw_llm_text[:8000],
        )


async def run_insight_job(
    collection: CollectionOutput,
    config: InsightJobConfig,
) -> InsightJobResult:
    job = InsightClassificationJob(config)
    return await job.run(collection)
