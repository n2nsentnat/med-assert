"""Typer CLI — composition root for PubMed collection (infrastructure + application)."""

from __future__ import annotations

from pathlib import Path

import typer
from pydantic import ValidationError

from med_assert.application.collect.service import CollectArticlesService
from med_assert.common.env import load_project_env
from med_assert.domain.errors import ArticleMinerError, NcbiError
from med_assert.infrastructure.collect.ncbi_client_config import NcbiClientConfig
from med_assert.infrastructure.collect.pubmed_gateway import EntrezPubMedGateway
from med_assert.infrastructure.collect.rate_limiter import RateLimiter
from med_assert.infrastructure.collect.resilient_http import ResilientHttpClient

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def collect(
    query: str = typer.Argument(..., help="PubMed search query (Entrez syntax)."),
    count: int = typer.Option(
        100,
        "--count",
        "-n",
        min=1,
        help="Maximum number of articles to retrieve.",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Path to write JSON output.",
        file_okay=True,
        dir_okay=False,
        writable=True,
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        help="NCBI API key (or set NCBI_API_KEY). Enables higher rate limits.",
        envvar="NCBI_API_KEY",
    ),
    email: str | None = typer.Option(
        None,
        "--email",
        help="Contact email (recommended by NCBI for API users).",
        envvar="NCBI_EMAIL",
    ),
    tool: str = typer.Option(
        "med_assert",
        "--tool",
        help="Identifying tool name sent to NCBI (required etiquette).",
    ),
) -> None:
    """Search PubMed and save structured article metadata as JSON."""
    load_project_env()
    config = NcbiClientConfig(api_key=api_key, email=email, tool=tool)
    limiter = RateLimiter(config.requests_per_second)
    http = ResilientHttpClient(config, limiter)
    try:
        gateway = EntrezPubMedGateway(http, config)
        service = CollectArticlesService(gateway)
        try:
            result = service.run(query=query, requested_count=count)
        except ValidationError as exc:
            typer.secho(f"Validation error: {exc}", err=True, fg=typer.colors.RED)
            raise typer.Exit(code=1) from exc
        except ValueError as exc:
            typer.secho(str(exc), err=True, fg=typer.colors.RED)
            raise typer.Exit(code=1) from exc
        except (NcbiError, ArticleMinerError) as exc:
            typer.secho(f"PubMed/NCBI error: {exc}", err=True, fg=typer.colors.RED)
            raise typer.Exit(code=1) from exc

        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                result.model_dump_json(indent=2, exclude_none=False),
                encoding="utf-8",
            )
        except OSError as exc:
            typer.secho(
                f"Failed to write {output}: {exc}", err=True, fg=typer.colors.RED
            )
            raise typer.Exit(code=1) from exc
        typer.echo(
            f"Wrote {result.retrieved_count} article(s) "
            f"({result.total_match_count} total matches) to {output}"
        )
        if result.warnings:
            for w in result.warnings[:20]:
                typer.secho(f"Warning: {w}", err=True, fg=typer.colors.YELLOW)
            if len(result.warnings) > 20:
                typer.secho(
                    f"... and {len(result.warnings) - 20} more warnings.",
                    err=True,
                    fg=typer.colors.YELLOW,
                )
    finally:
        http.close()


def run() -> None:
    """Entry point for ``collect-pubmed`` and ``python -m med_assert.cli.collect_app``."""
    app()


if __name__ == "__main__":
    run()
