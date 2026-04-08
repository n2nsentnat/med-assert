"""Collect PubMed JSON, then run duplicate detection (and optional insights)."""

from __future__ import annotations

import asyncio
import argparse
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from article_miner.application.collect.service import CollectArticlesService
from article_miner.application.insights.job import InsightJobConfig, run_insight_job
from article_miner.common.env import load_project_env
from article_miner.common.project_paths import default_project_root
from article_miner.domain.errors import ArticleMinerError, NcbiError
from article_miner.domain.insights.models import InsightJobResult
from article_miner.application.dedup.service import build_duplicate_report, format_dedup_markdown
from article_miner.infrastructure.collect.config import NcbiClientConfig
from article_miner.infrastructure.collect.pubmed_gateway import EntrezPubMedGateway
from article_miner.infrastructure.collect.rate_limiter import RateLimiter
from article_miner.infrastructure.collect.resilient_http import ResilientHttpClient


def _report_path_for_output(output: Path) -> Path:
    return output.with_name("insight_output_report.md")


def _write_insight_report_md(result: InsightJobResult, report_path: Path, output_path: Path) -> None:
    findings = Counter[str]()
    statuses = Counter[str]()
    needs_review_pmids: list[str] = []
    invalid_pmids: list[str] = []
    api_fail_pmids: list[str] = []

    for row in result.articles:
        statuses[row.status.value] += 1
        if row.insight is not None:
            findings[row.insight.extraction.finding_direction.value] += 1
        if row.status.value == "needs_human_review":
            needs_review_pmids.append(row.pmid)
        elif row.status.value == "invalid_output":
            invalid_pmids.append(row.pmid)
        elif row.status.value == "api_failure":
            api_fail_pmids.append(row.pmid)

    lines = [
        "# Insight Output Report",
        "",
        "## Summary",
        f"- Source query: `{result.source_query or '(none)'}`",
        f"- Prompt version: `{result.prompt_version}`",
        f"- Model: `{result.model}`",
        f"- Total articles: **{len(result.articles)}**",
        "",
        "## Status counts",
    ]
    for k, v in sorted(statuses.items()):
        lines.append(f"- `{k}`: **{v}**")

    lines.extend(
        [
            "",
            "## Finding direction distribution",
        ]
    )
    if findings:
        for k, v in sorted(findings.items()):
            lines.append(f"- `{k}`: **{v}**")
    else:
        lines.append("- No extracted findings (all rows skipped/failed).")

    lines.extend(
        [
            "",
            "## Review / failure locations",
            f"- Full machine-readable output: `{output_path}`",
            "- In that file, inspect rows where `status` is one of:",
            "  - `needs_human_review`",
            "  - `invalid_output`",
            "  - `api_failure`",
            "",
            "### PMIDs needing human review",
            ", ".join(needs_review_pmids[:100]) if needs_review_pmids else "(none)",
            "",
            "### PMIDs with invalid output",
            ", ".join(invalid_pmids[:100]) if invalid_pmids else "(none)",
            "",
            "### PMIDs with API failures",
            ", ".join(api_fail_pmids[:100]) if api_fail_pmids else "(none)",
        ]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        help="LiteLLM model id for insight classification (default: gpt-4o-mini).",
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
            extra_kwargs: dict[str, str] = {}
            if provider == "openai":
                insight_model = os.environ.get("INSIGHT_MODEL_OPENAI", "gpt-4o-mini")
            elif provider == "gemini":
                insight_model = os.environ.get("INSIGHT_MODEL_GEMINI", "gemini/gemini-2.0-flash")
            elif provider in ("claude", "anthropic"):
                insight_model = os.environ.get(
                    "INSIGHT_MODEL_CLAUDE", "anthropic/claude-3-5-sonnet-20241022"
                )
            elif provider == "ollama":
                insight_model = os.environ.get("OLLAMA_MODEL", "ollama/gemma3:4b")
                if not insight_model.startswith("ollama/"):
                    insight_model = f"ollama/{insight_model}"
                extra_kwargs["api_base"] = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            else:
                insight_model = args.insight_model

            if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
                print("warning: OPENAI_API_KEY is not set", file=sys.stderr)
            if provider == "gemini" and not os.environ.get("GEMINI_API_KEY"):
                print("warning: GEMINI_API_KEY is not set", file=sys.stderr)
            if provider in ("claude", "anthropic") and not os.environ.get("ANTHROPIC_API_KEY"):
                print("warning: ANTHROPIC_API_KEY is not set", file=sys.stderr)

            insights_path = Path(insights_path).expanduser().resolve()
            insight_config = InsightJobConfig(
                model=insight_model,
                confidence_threshold=args.insight_confidence,
                concurrency=max(1, args.insight_concurrency),
                enable_audit=not args.insight_no_audit,
                cache_path=args.insight_cache,
                extra_completion_kwargs=extra_kwargs,
            )
            insight_result = asyncio.run(run_insight_job(result, insight_config))
            insights_path.parent.mkdir(parents=True, exist_ok=True)
            if insights_path.suffix.lower() == ".jsonl":
                with insights_path.open("w", encoding="utf-8") as f:
                    for row in insight_result.articles:
                        f.write(row.model_dump_json() + "\n")
                summary_path = insights_path.with_suffix(".summary.json")
                summary_path.write_text(insight_result.model_dump_json(indent=2), encoding="utf-8")
            else:
                insights_path.write_text(insight_result.model_dump_json(indent=2), encoding="utf-8")
            report_path = _report_path_for_output(insights_path)
            _write_insight_report_md(
                result=insight_result,
                report_path=report_path,
                output_path=insights_path,
            )
    finally:
        http.close()

    print(f"Done. Output directory: {out_dir}")
    print(f"  articles: {articles_path}")
    print(f"  dupes JSON: {dupes_json}")
    print(f"  dupes Markdown: {dupes_md}")
    if args.with_insights:
        print(f"  insights: {insights_path}")
        print(f"  insights report: {_report_path_for_output(insights_path)}")
        if insights_path.suffix.lower() == ".jsonl":
            print(f"  insights summary: {insights_path.with_suffix('.summary.json')}")
    return 0


def run() -> None:
    """Console entry point (``pubmed-workflow``)."""
    raise SystemExit(main())


if __name__ == "__main__":
    run()
