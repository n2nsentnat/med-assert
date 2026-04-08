"""CLI: classify medical insights for articles in a CollectionOutput JSON."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import typer

from article_miner.application.insights.job import InsightJobConfig, run_insight_job
from article_miner.domain.collect.models import CollectionOutput

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
    model: str = typer.Option(
        "gpt-4o-mini",
        "--model",
        "-m",
        help="LiteLLM model id (e.g. gpt-4o-mini, anthropic/claude-3-5-sonnet-20241022, gemini/gemini-1.5-flash).",
    ),
    concurrency: int = typer.Option(8, "--concurrency", "-c", min=1, max=64),
    no_audit: bool = typer.Option(False, "--no-audit", help="Disable Pass 3 audit calls."),
    cache: Path | None = typer.Option(
        None,
        "--cache",
        help="SQLite path for extraction cache (optional).",
    ),
    confidence: float = typer.Option(0.5, "--confidence", min=0.0, max=1.0),
) -> None:
    """Run evidence-grounded LLM insight classification (async, one article per request)."""
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key and not os.environ.get("LITELLM_LOG"):
        typer.secho(
            "Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY for your provider.",
            err=True,
            fg=typer.colors.YELLOW,
        )

    text = input_json.read_text(encoding="utf-8")
    collection = CollectionOutput.model_validate_json(text)

    config = InsightJobConfig(
        model=model,
        confidence_threshold=confidence,
        concurrency=concurrency,
        enable_audit=not no_audit,
        cache_path=cache,
    )

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

    typer.echo(
        f"Stats: trusted={result.stats.get('success_trusted')} "
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
