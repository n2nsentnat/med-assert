"""Concrete PubMed gateway: ESearch + batched EFetch."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import ValidationError

from med_assert.domain.collect.models import Article
from med_assert.domain.errors import MalformedResponseError
from med_assert.infrastructure.collect.ncbi_client_config import (
    EFETCH_ID_BATCH_SIZE,
    EFETCH_URL,
    ESEARCH_PAGE_MAX,
    ESEARCH_URL,
    NcbiClientConfig,
)
from med_assert.infrastructure.collect.esearch_models import (
    ESearchEnvelope,
    ESearchInner,
)
from med_assert.infrastructure.collect.http_port import HttpTextClient
from med_assert.infrastructure.collect.pubmed_xml import parse_pubmed_xml_document

logger = logging.getLogger(__name__)

_ERROR_TAG = re.compile(r"<ERROR\b", re.IGNORECASE)
_ERROR_BODY = re.compile(r"<ERROR[^>]*>([^<]+)</ERROR>", re.IGNORECASE | re.DOTALL)


class EntrezPubMedGateway:
    """NCBI E-utilities implementation of ``PubMedGateway``.

    Injects :class:`~med_assert.infrastructure.collect.http_port.HttpTextClient`;
    production code should use :class:`~med_assert.infrastructure.collect.resilient_http.ResilientHttpClient`
    so requests are rate-limited and retried.
    """

    def __init__(self, http: HttpTextClient, config: NcbiClientConfig) -> None:
        self._http = http
        self._config = config

    def _common_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "db": "pubmed",
            "tool": self._config.tool,
        }
        if self._config.api_key:
            params["api_key"] = self._config.api_key
        if self._config.email:
            params["email"] = self._config.email
        return params

    def _params_with(self, **extra: Any) -> dict[str, Any]:
        return {**self._common_params(), **extra}

    def search_pmids(self, query: str, max_results: int) -> tuple[int, list[str]]:
        all_ids: list[str] = []
        retstart = 0
        total_count = 0

        while len(all_ids) < max_results:
            remaining = max_results - len(all_ids)
            retmax = min(ESEARCH_PAGE_MAX, remaining)
            params = self._params_with(
                term=query,
                retstart=retstart,
                retmax=retmax,
                retmode="json",
            )
            body = self._http.get_text(ESEARCH_URL, params)
            inner = self._parse_esearch_json(body)
            total_count = int(inner.count)
            idlist = list(inner.idlist)
            cap = min(max_results, total_count)
            all_ids.extend(idlist)
            logger.debug(
                "ESearch page: retstart=%s retmax=%s returned_ids=%s total_match_count=%s accumulated=%s",
                retstart,
                retmax,
                len(idlist),
                total_count,
                len(all_ids),
            )
            if len(all_ids) >= cap:
                self._log_esearch_cap_notice(total_count, max_results)
                return total_count, all_ids[:cap]
            if not idlist:
                break
            if len(idlist) < retmax:
                break
            retstart += len(idlist)

        cap = min(max_results, total_count) if total_count else len(all_ids)
        self._log_esearch_cap_notice(total_count, max_results)
        return total_count, all_ids[:cap]

    @staticmethod
    def _log_esearch_cap_notice(total_count: int, max_results: int) -> None:
        if total_count > max_results:
            logger.info(
                "ESearch: total_match_count=%s exceeds requested max_results=%s; "
                "returning only the first %s PMIDs.",
                total_count,
                max_results,
                max_results,
            )

    def fetch_articles(self, pmids: list[str]) -> tuple[list[Article], list[str]]:
        if not pmids:
            return [], []

        by_pmid: dict[str, Article] = {}
        warnings: list[str] = []

        for i in range(0, len(pmids), EFETCH_ID_BATCH_SIZE):
            batch = pmids[i : i + EFETCH_ID_BATCH_SIZE]
            batch_num = i // EFETCH_ID_BATCH_SIZE + 1
            logger.info(
                "EFetch batch %s/%s: %s PMID(s) (slice %s:%s)",
                batch_num,
                (len(pmids) + EFETCH_ID_BATCH_SIZE - 1) // EFETCH_ID_BATCH_SIZE,
                len(batch),
                i,
                i + len(batch),
            )
            params = self._params_with(
                id=",".join(batch),
                retmode="xml",
            )
            xml_text = self._http.get_text(EFETCH_URL, params)
            self._raise_if_efetch_error(xml_text)
            parsed = parse_pubmed_xml_document(xml_text)
            for article in parsed:
                by_pmid[article.pmid] = article

        ordered: list[Article] = []
        missing_pmids: list[str] = []
        for pid in pmids:
            article = by_pmid.get(pid)
            if article is None:
                missing_pmids.append(pid)
                warnings.append(f"No parseable article returned for PMID {pid}.")
            else:
                ordered.append(article)

        if missing_pmids:
            preview = ", ".join(missing_pmids[:20])
            suffix = (
                f" ... (+{len(missing_pmids) - 20} more)"
                if len(missing_pmids) > 20
                else ""
            )
            logger.warning(
                "EFetch: no parseable article for %s PMID(s). IDs: %s%s",
                len(missing_pmids),
                preview,
                suffix,
            )

        return ordered, warnings

    def _parse_esearch_json(self, body: str) -> ESearchInner:
        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise MalformedResponseError(
                f"ESearch returned invalid JSON: {exc}"
            ) from exc
        try:
            envelope = ESearchEnvelope.model_validate(data)
        except ValidationError as exc:
            raise MalformedResponseError(
                f"ESearch JSON failed validation: {exc}"
            ) from exc
        return envelope.esearchresult

    @staticmethod
    def _raise_if_efetch_error(xml_text: str) -> None:
        if not _ERROR_TAG.search(xml_text):
            return
        parts = [p.strip() for p in _ERROR_BODY.findall(xml_text) if p.strip()]
        detail = "; ".join(parts) if parts else "unknown error"
        raise MalformedResponseError(f"EFetch returned ERROR: {detail}")
