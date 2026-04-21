"""CLI: find probable duplicate groups in a collected PubMed JSON file."""

from __future__ import annotations

from pathlib import Path

import typer
from pydantic import ValidationError

from article_miner.application.dedup.service import (
    build_duplicate_report,
    format_dedup_markdown,
    load_collection,
)


def main(
    input_json: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="JSON file produced by collect-pubmed (CollectionOutput).",
    ),
    output_json: Path | None = typer.Option(
        None,
        "--out-json",
        "-o",
        help="Write full DedupReport JSON (includes methodology string).",
    ),
    markdown: Path | None = typer.Option(
        None,
        "--markdown",
        "-m",
        help="Write human-readable Markdown summary for reviewers.",
    ),
    specter: bool = typer.Option(
        False,
        "--specter",
        help="Enable SPECTER 2 embeddings + FAISS similarity layer (CPU; downloads model).",
    ),
    specter_model: str | None = typer.Option(
        None,
        "--specter-model",
        help="Hugging Face embedding model (default allenai/specter2_base).",
    ),
) -> None:
    """Identify probable duplicate articles (same work, different PMIDs or versions)."""
    try:
        collection = load_collection(str(input_json))
    except ValidationError as exc:
        typer.secho(f"Invalid collection JSON: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    except OSError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    report = build_duplicate_report(
        collection,
        enable_specter_faiss=specter,
        specter_model=specter_model,
    )
    text = report.model_dump_json(indent=2)

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(text, encoding="utf-8")
        typer.echo(f"Wrote report JSON to {output_json}")

    if markdown is not None:
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(format_dedup_markdown(report), encoding="utf-8")
        typer.echo(f"Wrote Markdown to {markdown}")

    if output_json is None and markdown is None:
        typer.echo(text)


def run() -> None:
    typer.run(main)


if __name__ == "__main__":
    run()
