# Article miner (PubMed JSON collector)

Command-line tool that searches [PubMed](https://pubmed.ncbi.nlm.nih.gov/) via the [NCBI E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25500/) (`esearch` + `efetch`), applies rate limiting and retries, and writes **flat, validated JSON** suitable for downstream pipelines.

## Requirements

- Python **3.13**
- [uv](https://docs.astral.sh/uv/) for environments and dependency management
- **[lxml](https://lxml.de/)** for PubMed XML (`efetch`): XPath parsing, namespace-agnostic queries, `huge_tree` for large responses, and `resolve_entities=False` for safer parsing

## Setup

```bash
cd article_miner
uv sync --all-groups
```

## Usage

```bash
uv run python collect.py "machine learning" --count 100 --output results.json
```

Equivalent:

```bash
uv run python -m article_miner "cancer immunotherapy" --count 50 --output out.json
```

After install, the console script `collect-pubmed` is also available:

```bash
uv run collect-pubmed "diabetes" -n 20 -o diabetes.json
```

### Options

| Option | Meaning |
|--------|---------|
| `QUERY` (positional) | Entrez/PubMed query string |
| `-n`, `--count` | Maximum articles to retrieve (default: 100) |
| `-o`, `--output` | Output JSON path (required) |
| `--api-key` | NCBI API key; same as env `NCBI_API_KEY` |
| `--email` | Contact email; same as env `NCBI_EMAIL` (recommended by NCBI) |
| `--tool` | Tool name sent to NCBI (default: `article_miner`) |

### API key and rate limits

Without an API key, the client spaces requests for roughly **3 requests per second**. With a key, roughly **10 per second** (see [NCBI usage guidelines](https://www.ncbi.nlm.nih.gov/books/NBK25497/)).

Switching is a single flag or environment variable:

```bash
export NCBI_API_KEY=your_key_here
uv run python collect.py "query" -n 500 -o out.json
# or
uv run python collect.py "query" -n 500 -o out.json --api-key your_key_here
```

## Output JSON

The file is a single object:

- `query`, `total_match_count`, `requested_count`, `retrieved_count`
- `articles`: list of records with stable fields such as `pmid`, `title`, `abstract`, `authors`, `doi`, `journal_full`, `mesh_terms`, `keywords`, etc.
- `warnings`: non-fatal issues (for example PMIDs that did not return parseable XML)

All article objects are validated with **Pydantic** before serialization.

## LLM medical insight classification (evidence-grounded)

After you have a `CollectionOutput` JSON from `collect-pubmed`, you can run **Pass 1** (one article per LiteLLM request, JSON-only), **Pass 2** (deterministic Pydantic + substring grounding + heuristic contradiction flags), and **Pass 3** (optional audit call only when triggers fire: low confidence, mixed findings, clinically “meaningful”, failed grounding, or semantic flags).

- **Evidence**: the model must return verbatim **evidence spans** from the title/abstract; the system checks that each span occurs in a canonical concatenation of title + abstract.
- **Trust**: `PerArticleStatus` is `success` only when schema, grounding, auto-accept heuristics, and (if run) audit all agree; otherwise rows are `needs_human_review` (or `invalid_output` / `api_failure` / `skipped_prefilter`).
- **Providers**: set one of `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GEMINI_API_KEY` (LiteLLM). Choose a model with `--model` (e.g. `gpt-4o-mini`, `anthropic/claude-3-5-sonnet-20241022`, `gemini/gemini-1.5-flash`).
- **Caching**: optional `--cache /path/to/cache.sqlite` keys on PMID + input hash + prompt version + model.

```bash
uv run classify-insights results.json -o insights.json --model gpt-4o-mini
# JSONL per-article lines + .summary.json sidecar:
uv run classify-insights results.json -o insights.jsonl
```

**Prompt version** is `PROMPT_VERSION` in `article_miner.infrastructure.insights.prompts` (bump when instructions change to invalidate caches).

**Offline evaluation**: see [`scripts/eval_insight_metrics.py`](scripts/eval_insight_metrics.py) (optional `pandas` / `scikit-learn` for metrics). For rigorous work, build a gold set (75–150 articles) and report accuracy / macro-F1 / **precision on auto-accept** rows.

**Runtime metrics** (from `InsightJobResult.stats` in the JSON / `.summary.json` sidecar): `input_tokens` / `output_tokens` (when the provider returns usage), counts for `success_trusted`, `needs_review`, `invalid_output`, `api_failure`, `skipped_prefilter`, and `truncation_warning` (when canonical title+abstract exceeds the configured length). Derive schema pass rate, grounding pass rate, and human-review rate from per-article rows.

## Finding probable duplicates

PubMed often lists the **same work** under more than one PMID (preprint vs journal, meeting abstract vs paper, etc.). After collecting JSON, you can scan for **probable** duplicate groups (for human review, not automatic deletion):

```bash
uv run find-pubmed-dupes results.json -o dupes.json -m dupes.md
```

- **`-o` / `--out-json`**: machine-readable report (`methodology` field documents rules and thresholds).
- **`-m` / `--markdown`**: short Markdown with cluster tables and optional retraction hints.

Definitions, similarity metrics, scalability notes, and edge-case handling are documented in the report’s `methodology` string and in the module docstring of `article_miner.application.dedup.service`.

### One-shot workflow (collect + dedup)

**Python** (same behavior, in-process—no subprocess; uses the same services as `collect-pubmed` + dedup):

```bash
uv run pubmed-workflow -n 25 "diabetes mellitus[tiab]"
# or
uv run python scripts/pubmed_workflow.py -n 25 "diabetes mellitus[tiab]"
```

Writes a timestamped directory (default: under the project root when `pyproject.toml` is found) with `articles.json`, `dupes.json`, and `dupes.md`. Use `-d /path/to/dir` to choose the output folder. Set `NCBI_API_KEY` / `NCBI_EMAIL` in the environment if you use them for collection.

## Behavior notes

- **Search pagination**: `esearch` returns at most **10,000** IDs per call; larger requests use multiple pages.
- **Detail batches**: `efetch` IDs are requested in batches (default **200** per call) to stay within practical limits.
- **Errors**: Transient HTTP failures and 5xx responses use bounded retries with backoff; repeated **429** responses surface as a clear error. Malformed JSON/XML yields `MalformedResponseError`-style messages at the CLI.

## Architecture

Layers follow **Clean / Onion** style dependency direction (inward-only):

| Layer | Role |
|-------|------|
| **Domain** | `Article`, `CollectionOutput`, domain exceptions — no I/O |
| **Application** | `CollectArticlesService` + `PubMedGateway` port (protocol) |
| **Infrastructure** | NCBI `EntrezPubMedGateway`, `ResilientHttpClient`, `RateLimiter`, XML/JSON parsing |
| **CLI** | Typer composition root: wires config → HTTP → gateway → use case |

**SOLID**: single-purpose modules, gateway depends on `HttpTextClient` protocol for testing, use case depends on `PubMedGateway` protocol — not on `httpx` or URLs.

### Tool-specific module layout

- `article_miner/infrastructure/collect/`: PubMed collection adapters (HTTP client, gateway, XML parser, models, rate limiting).
- `article_miner/dedup/`: duplicate detection engine and reporting.
- `article_miner/infrastructure/insights/`: LLM prompts, extraction adapter, deterministic validation, cache/prefilter.
- `article_miner/common/`: shared cross-tool utilities (for example, project path helpers used by CLIs).

## Development

```bash
uv sync --all-groups
uv run pytest
```

### Live NCBI smoke test (optional)

Unit tests mock HTTP. To exercise the real E-utilities stack (same path as the CLI):

```bash
ARTICLE_MINER_LIVE_NCBI=1 uv run pytest tests/test_live_pubmed_collect.py -v -m live_ncbi
```

Requires network access. The live module runs the same assertion against several broad queries (e.g. diabetes, hypertension, COVID-19, machine learning, publication type). Without `ARTICLE_MINER_LIVE_NCBI=1`, those tests are skipped so CI stays offline by default.

## License

Add your license here.
