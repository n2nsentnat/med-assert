"""CLI: classify medical insights for articles in a CollectionOutput JSON."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections import Counter
from pathlib import Path

import typer

from article_miner.application.insights.job import InsightJobConfig, run_insight_job
from article_miner.common.env import load_project_env
from article_miner.domain.collect.models import CollectionOutput
from article_miner.domain.insights.models import InsightJobResult

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _report_path_for_output(output: Path) -> Path:
    if output.suffix.lower() == ".jsonl":
        return output.with_name("insight_output_report.md")
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


def main(
    input_json: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Collection JSON from collect-pubmed.",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output path (.json or .jsonl).",
    ),
    model: str = typer.Option(
        "gpt-4o-mini",
        "--model",
        "-m",
        help="LiteLLM model id (e.g. gpt-4o-mini, anthropic/claude-3-5-sonnet-20241022, gemini/gemini-1.5-flash).",
    ),
    llm: str | None = typer.Option(
        None,
        "--llm",
        help="Provider shortcut: openai | gemini | claude | ollama (model from .env defaults).",
    ),
    concurrency: int = typer.Option(8, "--concurrency", "-c", min=1, max=64),
    no_audit: bool = typer.Option(False, "--no-audit", help="Disable Pass 3 audit calls."),
    cache: Path | None = typer.Option(
        None,
        "--cache",
        help="SQLite path for extraction cache (optional).",
    ),
    confidence: float = typer.Option(0.5, "--confidence", min=0.0, max=1.0),
    incremental_jsonl: Path | None = typer.Option(
        None,
        "--incremental-jsonl",
        help="Append per-article results as they complete (crash-resilient progress log).",
    ),
    no_progress: bool = typer.Option(
        False,
        "--no-progress",
        help="Disable periodic progress logs.",
    ),
    progress_every: int = typer.Option(
        1,
        "--progress-every",
        min=1,
        help="Print progress every N completed articles (default: 1).",
    ),
    report_md: Path | None = typer.Option(
        None,
        "--report-md",
        help="Markdown summary report path (default: insight_output_report.md next to output).",
    ),
) -> None:
    """Run evidence-grounded LLM insight classification (async, one article per request)."""
    load_project_env()
    if not no_progress:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    provider = (llm or "").strip().lower()
    selected_model = model
    extra_kwargs: dict[str, str] = {}
    if provider == "openai":
        selected_model = os.environ.get("INSIGHT_MODEL_OPENAI", "gpt-4o-mini")
    elif provider == "gemini":
        selected_model = os.environ.get("INSIGHT_MODEL_GEMINI", "gemini/gemini-2.0-flash")
    elif provider in ("claude", "anthropic"):
        selected_model = os.environ.get("INSIGHT_MODEL_CLAUDE", "anthropic/claude-3-5-sonnet-20241022")
    elif provider == "ollama":
        selected_model = os.environ.get("OLLAMA_MODEL", "ollama/gemma3:4b")
        if not selected_model.startswith("ollama/"):
            selected_model = f"ollama/{selected_model}"
        extra_kwargs["api_base"] = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    elif provider:
        raise typer.BadParameter("--llm must be one of: openai, gemini, claude, ollama")

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if provider != "ollama" and not api_key and not os.environ.get("LITELLM_LOG"):
        typer.secho(
            "Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY for your provider.",
            err=True,
            fg=typer.colors.YELLOW,
        )

    text = input_json.read_text(encoding="utf-8")
    collection = CollectionOutput.model_validate_json(text)

    config = InsightJobConfig(
        model=selected_model,
        confidence_threshold=confidence,
        concurrency=concurrency,
        enable_audit=not no_audit,
        cache_path=cache,
        incremental_jsonl_path=incremental_jsonl,
        progress=not no_progress,
        progress_every=progress_every,
        extra_completion_kwargs=extra_kwargs,
    )

    if incremental_jsonl is not None:
        incremental_jsonl.parent.mkdir(parents=True, exist_ok=True)
        incremental_jsonl.write_text("", encoding="utf-8")

    result = asyncio.run(run_insight_job(collection, config))

    out_path = output
    if out_path.suffix.lower() == ".jsonl":
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for row in result.articles:
                f.write(row.model_dump_json() + "\n")
        summary = result.model_dump_json(indent=2)
        summary_path = out_path.with_suffix(".summary.json")
        summary_path.write_text(summary, encoding="utf-8")
        typer.echo(f"Wrote JSONL to {out_path} and summary to {summary_path}")
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        typer.echo(f"Wrote {out_path}")

    rep_path = report_md or _report_path_for_output(out_path)
    _write_insight_report_md(result, rep_path, out_path)
    typer.echo(f"Wrote report {rep_path}")

    typer.echo(
        f"Stats: auto_accepted={result.stats.get('auto_accepted', 0)} "
        f"validated_flagged={result.stats.get('validated_but_flagged', 0)} "
        f"needs_review={result.stats.get('needs_review')} "
        f"invalid={result.stats.get('invalid_output')} "
        f"api_fail={result.stats.get('api_failure')} "
        f"skipped={result.stats.get('skipped_prefilter')} "
        f"trunc_warn={result.stats.get('truncation_warning', 0)} "
        f"tokens_in={int(result.stats.get('input_tokens', 0))} "
        f"tokens_out={int(result.stats.get('output_tokens', 0))}"
    )


def run() -> None:
    typer.run(main)


if __name__ == "__main__":
    run()
