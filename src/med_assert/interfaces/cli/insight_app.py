"""CLI: classify medical insights for articles in a CollectionOutput JSON."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

import typer

from med_assert.application.insights.job import InsightJobConfig, run_insight_job
from med_assert.application.insights.llm_provider_registry import (
    expected_api_key_env_name,
    registered_insight_providers,
    resolve_insight_llm_provider,
)
from med_assert.infrastructure.insights.chat_model_factory import (
    build_chat_model,
    insight_display_name,
)
from med_assert.application.insights.report import (
    default_insight_report_path,
    write_insight_report_md,
)
from med_assert.common.env import load_project_env
from med_assert.domain.collect.models import CollectionOutput

app = typer.Typer(add_completion=False, no_args_is_help=True)


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
    llm: str | None = typer.Option(
        None,
        "--llm",
        help="Provider shortcut: openai | gemini | claude | ollama (model from .env defaults).",
    ),
    concurrency: int = typer.Option(8, "--concurrency", "-c", min=1, max=64),
    no_audit: bool = typer.Option(
        False, "--no-audit", help="Disable Pass 3 audit calls."
    ),
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
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
        )
    provider = (llm or "").strip().lower()
    if provider:
        try:
            resolution = resolve_insight_llm_provider(provider, os.environ)
        except KeyError:
            allowed = ", ".join(registered_insight_providers())
            raise typer.BadParameter(f"--llm must be one of: {allowed}") from None
    else:
        resolution = resolve_insight_llm_provider("openai", os.environ)

    chat_model = build_chat_model(resolution)
    display_name = insight_display_name(resolution)

    key_var = expected_api_key_env_name(provider) if provider else None
    if key_var and not os.environ.get(key_var) and not os.environ.get("LANGCHAIN_TRACING_V2"):
        typer.secho(
            f"Set {key_var} for the selected provider.",
            err=True,
            fg=typer.colors.YELLOW,
        )

    text = input_json.read_text(encoding="utf-8")
    collection = CollectionOutput.model_validate_json(text)

    config = InsightJobConfig(
        model=display_name,
        chat_model=chat_model,
        audit_chat_model=chat_model,
        confidence_threshold=confidence,
        concurrency=concurrency,
        enable_audit=not no_audit,
        cache_path=cache,
        incremental_jsonl_path=incremental_jsonl,
        progress=not no_progress,
        progress_every=progress_every,
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

    rep_path = report_md or default_insight_report_path(out_path)
    write_insight_report_md(result, rep_path, out_path)
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
