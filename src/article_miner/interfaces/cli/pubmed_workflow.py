"""Collect PubMed JSON, then run duplicate detection (and optional insights)."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from article_miner.application.collect.service import CollectArticlesService
from article_miner.application.dedup.service import (
    build_duplicate_report,
    format_dedup_markdown,
)
from article_miner.application.insights.job import InsightJobConfig, run_insight_job
from article_miner.application.insights.llm_provider_registry import (
    expected_api_key_env_name,
    resolve_insight_llm_provider,
)
from article_miner.infrastructure.insights.chat_model_factory import (
    build_chat_model,
    insight_display_name,
)
from article_miner.application.insights.report import (
    default_insight_report_path,
    write_insight_report_md,
)
from article_miner.common.env import load_project_env
from article_miner.common.project_paths import default_project_root
from article_miner.domain.errors import ArticleMinerError, NcbiError
from article_miner.infrastructure.collect.ncbi_client_config import NcbiClientConfig
from article_miner.infrastructure.collect.pubmed_gateway import EntrezPubMedGateway
from article_miner.infrastructure.collect.rate_limiter import RateLimiter
from article_miner.infrastructure.collect.resilient_http import ResilientHttpClient


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect PubMed articles to JSON, then write duplicate report (JSON + Markdown).",
    )
    p.add_argument(
        "query",
        nargs="+",
        help="PubMed / Entrez search query (multiple words allowed)",
    )
    p.add_argument(
        "-n",
        "--count",
        type=int,
        default=100,
        help="Maximum articles to retrieve (default: 100)",
    )
    p.add_argument(
        "-d",
        "--dir",
        type=Path,
        default=None,
        help="Output directory (default: workflow_YYYYMMDD_HHMMSS under project root or cwd)",
    )
    p.add_argument(
        "--tool",
        default=os.environ.get("NCBI_TOOL", "article_miner"),
        help="Tool name for NCBI (default: article_miner or NCBI_TOOL)",
    )
    p.add_argument(
        "--with-insights",
        action="store_true",
        help="Run LLM insight classification after collect+dedup.",
    )
    p.add_argument(
        "--insight-model",
        default="gpt-4o-mini",
        help="LangChain / provider model id for insight classification (default: gpt-4o-mini).",
    )
    p.add_argument(
        "--insight-llm",
        choices=("openai", "gemini", "claude", "anthropic", "ollama"),
        default=None,
        help=(
            "Insight provider shortcut. Model comes from env (INSIGHT_MODEL_OPENAI / "
            "INSIGHT_MODEL_GEMINI / INSIGHT_MODEL_CLAUDE / OLLAMA_MODEL) or provider defaults."
        ),
    )
    p.add_argument(
        "--insight-concurrency",
        type=int,
        default=8,
        help="Concurrency for classify-insights (default: 8).",
    )
    p.add_argument(
        "--insight-no-audit",
        action="store_true",
        help="Disable optional Pass 3 audit in classify-insights.",
    )
    p.add_argument(
        "--insight-cache",
        type=Path,
        default=None,
        help="Optional SQLite cache path for classify-insights.",
    )
    p.add_argument(
        "--insight-confidence",
        type=float,
        default=0.5,
        help="Confidence threshold for auto-accept in classify-insights (default: 0.5).",
    )
    p.add_argument(
        "--insight-output",
        type=Path,
        default=None,
        help="Insight output file path (.json or .jsonl). Default: <out_dir>/insights.json",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_project_env()
    args = _parse_args(argv)
    query = " ".join(args.query).strip()
    if not query:
        print("error: QUERY is empty", file=sys.stderr)
        return 2

    root = default_project_root()
    if args.dir is None:
        out_dir = root / f"workflow_{datetime.now():%Y%m%d_%H%M%S}"
    else:
        out_dir = Path(args.dir).expanduser().resolve()

    out_dir.mkdir(parents=True, exist_ok=True)
    articles_path = out_dir / "articles.json"
    dupes_json = out_dir / "dupes.json"
    dupes_md = out_dir / "dupes.md"
    insights_path = args.insight_output or (out_dir / "insights.json")

    config = NcbiClientConfig(
        api_key=os.environ.get("NCBI_API_KEY"),
        email=os.environ.get("NCBI_EMAIL"),
        tool=args.tool,
    )
    limiter = RateLimiter(config.requests_per_second)
    http = ResilientHttpClient(config, limiter)
    try:
        gateway = EntrezPubMedGateway(http, config)
        service = CollectArticlesService(gateway)
        try:
            result = service.run(query=query, requested_count=args.count)
        except ValidationError as exc:
            print(f"Validation error: {exc}", file=sys.stderr)
            return 1
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        except (NcbiError, ArticleMinerError) as exc:
            print(f"PubMed/NCBI error: {exc}", file=sys.stderr)
            return 1

        try:
            articles_path.write_text(
                result.model_dump_json(indent=2, exclude_none=False),
                encoding="utf-8",
            )
        except OSError as exc:
            print(f"Failed to write {articles_path}: {exc}", file=sys.stderr)
            return 1

        report = build_duplicate_report(result)
        dupes_json.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        dupes_md.write_text(format_dedup_markdown(report), encoding="utf-8")

        if args.with_insights:
            if not args.insight_llm:
                print(
                    "error: --insight-llm is required when --with-insights is enabled",
                    file=sys.stderr,
                )
                return 2
            provider = (args.insight_llm or "").strip().lower()
            try:
                resolution = resolve_insight_llm_provider(provider, os.environ)
            except KeyError:
                print(f"error: unknown --insight-llm {provider!r}", file=sys.stderr)
                return 2
            chat_model = build_chat_model(resolution)
            display_name = insight_display_name(resolution)

            warn_var = expected_api_key_env_name(provider)
            if warn_var and not os.environ.get(warn_var):
                print(f"warning: {warn_var} is not set", file=sys.stderr)

            insights_path = Path(insights_path).expanduser().resolve()
            insight_config = InsightJobConfig(
                model=display_name,
                chat_model=chat_model,
                audit_chat_model=chat_model,
                confidence_threshold=args.insight_confidence,
                concurrency=max(1, args.insight_concurrency),
                enable_audit=not args.insight_no_audit,
                cache_path=args.insight_cache,
            )
            insight_result = asyncio.run(run_insight_job(result, insight_config))
            insights_path.parent.mkdir(parents=True, exist_ok=True)
            if insights_path.suffix.lower() == ".jsonl":
                with insights_path.open("w", encoding="utf-8") as f:
                    for row in insight_result.articles:
                        f.write(row.model_dump_json() + "\n")
                summary_path = insights_path.with_suffix(".summary.json")
                summary_path.write_text(
                    insight_result.model_dump_json(indent=2), encoding="utf-8"
                )
            else:
                insights_path.write_text(
                    insight_result.model_dump_json(indent=2), encoding="utf-8"
                )
            report_path = default_insight_report_path(insights_path)
            write_insight_report_md(
                result=insight_result,
                report_path=report_path,
                machine_readable_output=insights_path,
            )
    finally:
        http.close()

    print(f"Done. Output directory: {out_dir}")
    print(f"  articles: {articles_path}")
    print(f"  dupes JSON: {dupes_json}")
    print(f"  dupes Markdown: {dupes_md}")
    if args.with_insights:
        print(f"  insights: {insights_path}")
        print(f"  insights report: {default_insight_report_path(insights_path)}")
        if insights_path.suffix.lower() == ".jsonl":
            print(f"  insights summary: {insights_path.with_suffix('.summary.json')}")
    return 0


def run() -> None:
    """Console entry point (``pubmed-workflow``)."""
    raise SystemExit(main())


if __name__ == "__main__":
    run()
