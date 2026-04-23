"""Human-readable Markdown summary for an insight job (application layer)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from med_assert.domain.insights.models import InsightJobResult


def default_insight_report_path(output: Path) -> Path:
    """Path for the default report file next to the machine-readable output."""
    return output.with_name("insight_output_report.md")


def write_insight_report_md(
    result: InsightJobResult,
    report_path: Path,
    machine_readable_output: Path,
) -> None:
    """Write status counts, finding distribution, and review PMIDs to Markdown."""
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
            f"- Full machine-readable output: `{machine_readable_output}`",
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
