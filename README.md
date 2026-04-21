## MedAssert 
#### LLM-Powered Medical Assertion Extraction with Deterministic Trust Gates.
MedAssert is an evidence-grounded extraction engine that transforms unstructured biomedical abstracts into verified clinical assertions. Unlike standard LLM summarizers, MedAssert enforces a "grounding-first" policy, requiring verbatim evidence spans for every finding. It utilizes a three-pass validation pipeline—LLM extraction, deterministic grounding checks, and automated risk triage—to categorize research findings by directionality, statistical significance, and clinical meaningfulness.

This command-line tool searches [PubMed](https://pubmed.ncbi.nlm.nih.gov/) via the [NCBI E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25500/) (`esearch` + `efetch`), applies rate limiting and retries, writes flat, validated JSON, detects probable duplicates with **rule-based matching** and an optional **SPECTER 2 embedding + FAISS** similarity layer (see [Finding probable duplicates](#finding-probable-duplicates)), and ultimately produces structured clinical interpretations including:

    Finding Direction: (Positive, Negative, Neutral, Mixed, Unclear)

    Clinical Meaningfulness: Significance and impact assessment.

    Verbatim Grounding: Automatic verification that evidence spans exist in the source text.

    Trust Triage: Multi-pass validation to sort findings into auto_accepted vs. needs_human_review.

**Roadmap / planned work:** [TODO.md](TODO.md)
## Topics

- [Requirements](README.md#requirements)
- [Setup](README.md#setup)
- [HTTP API (FastAPI)](README_API.md)
- [Usage](README.md#usage)
- [Output JSON](#output-json)
- [LLM medical insight classification](#insights-section)
- [Definition of duplicates](#finding-probable-duplicates)
- [Behavior notes](#behavior-notes)
- [Architecture](#architecture)
- [Live NCBI smoke test (optional)](#live-ncbi-smoke-test-optional)
- [License](#license)

<a id="requirements"></a>
## Requirements

- Python **3.13**
- [uv](https://docs.astral.sh/uv/) for environments and dependency management
- **[lxml](https://lxml.de/)** for PubMed XML (`efetch`): XPath parsing, namespace-agnostic queries, `huge_tree` for large responses, and `resolve_entities=False` for safer parsing

<a id="setup"></a>
## Setup

Install dependencies:

```bash
cd article_miner
uv sync --all-groups
```

Create local environment file (CLI auto-loads `.env`):

```bash
cp .env .env.local 2>/dev/null || true
# or edit .env directly and set your keys
```

Minimum recommended variables:

- `NCBI_API_KEY` (optional but recommended for higher rate limits)
- one LLM provider for insights: `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY`
- for local inference: `OLLAMA_BASE_URL` and `OLLAMA_MODEL`

Quick sanity checks:

```bash
uv run collect-pubmed "diabetes mellitus[tiab]" -n 3 -o results.json
uv run find-pubmed-dupes results.json -o dupes.json -m dupes.md
uv run classify-insights results.json -o insights.json --llm ollama
```

<a id="usage"></a>
## Usage

### 1) End-to-end workflow (recommended)

Run the full pipeline in one command:

- collect PubMed articles
- run duplicate detection
- optionally run insight classification

```bash
uv run pubmed-workflow -n 25 "diabetes mellitus[tiab]"
```

With insights enabled:

```bash
uv run pubmed-workflow \
  -n 50 \
  --with-insights \
  --insight-llm openai \
  --insight-output ./out/insights.jsonl \
  "hypertension[tiab]"
```

#### End-to-end workflow options

| Option | Meaning |
|--------|---------|
| `QUERY` (positional) | PubMed / Entrez query string (can be multiple words) |
| `-n`, `--count` | Maximum articles to retrieve (default: 100) |
| `-d`, `--dir` | Output directory (default: timestamped `workflow_...`) |
| `--tool` | Tool name sent to NCBI (default: `article_miner`) |
| `--with-insights` | Enable insight classification after collect + dedup |
| `--insight-llm` | **Required when `--with-insights` is used**. Provider shortcut (`openai`, `gemini`, `claude`, `anthropic`, `ollama`) |
| `--insight-model` | Optional explicit model override for insights |
| `--insight-concurrency` | Concurrency for insight classification workers |
| `--insight-no-audit` | Disable optional audit pass |
| `--insight-cache` | SQLite cache path for insight extraction reuse |
| `--insight-confidence` | Secondary confidence triage threshold |
| `--insight-output` | Insight output path (`.json` or `.jsonl`) |

#### End-to-end output

Default output directory: `workflow_YYYYMMDD_HHMMSS/` (or `--dir` if provided).

- `articles.json`: collected `CollectionOutput`
- `dupes.json`: machine-readable dedup report
- `dupes.md`: reviewer-friendly dedup summary
- `insights.json` or `insights.jsonl` (only when `--with-insights`)
- `insights.summary.json` (when insight output is `.jsonl`)
- `insight_output_report.md`: human-readable insights summary (key metrics + PMIDs needing review/failure)

### 2) Individual tool usage

#### 2.1) Collect

```bash
uv run collect-pubmed "machine learning" --count 100 --output results.json
```

#### Collect options

| Option | Meaning |
|--------|---------|
| `QUERY` (positional) | Entrez/PubMed query string |
| `-n`, `--count` | Maximum articles to retrieve (default: 100) |
| `-o`, `--output` | Output JSON path (required) |
| `--api-key` | NCBI API key; same as env `NCBI_API_KEY` |
| `--email` | Contact email; same as env `NCBI_EMAIL` (recommended by NCBI) |
| `--tool` | Tool name sent to NCBI (default: `article_miner`) |

##### Collect output

- Single JSON file (`--output`) in `CollectionOutput` shape:
  - `query`, `total_match_count`, `requested_count`, `retrieved_count`
  - `articles[]`
  - `warnings[]`

#### 2.2) Dedup

```bash
uv run find-pubmed-dupes results.json -o dupes.json -m dupes.md
```

##### Dedup options

| Option | Meaning |
|--------|---------|
| `input_json` (positional) | Collection JSON produced by `collect-pubmed` |
| `-o`, `--out-json` | Output path for full dedup report JSON |
| `-m`, `--markdown` | Output path for reviewer-friendly Markdown summary |
| `--specter` | Add a **SPECTER 2** embedding + **FAISS** cosine-similarity layer after rule-based dedup (requires `uv sync --extra specter` or `pip install 'article-miner[specter]'`) |
| `--specter-model` | Hugging Face model id (default `allenai/specter2_base`) |

Environment: set `ARTICLE_MINER_SPECTER=1` to enable the same layer by default in programmatic `build_duplicate_report()` calls.

##### Dedup output

- If `--out-json` is set: full dedup report JSON
- If `--markdown` is set: human-readable duplicate cluster summary
- If both are omitted: prints dedup report JSON to stdout

#### 2.3) Insights

```bash
uv run classify-insights results.json -o insights.json --llm openai
# JSONL per-article lines + .summary.json sidecar:
uv run classify-insights results.json -o insights.jsonl --llm openai
```

JSONL + live progress file:

```bash
uv run classify-insights results.json \
  -o insights.jsonl \
  --llm ollama \
  --incremental-jsonl insights.progress.jsonl
```

##### Insights options

| Option | Meaning |
|--------|---------|
| `input_json` (positional) | Collection JSON produced by `collect-pubmed` |
| `-o`, `--output` | Output path (`.json` or `.jsonl`) |
| `-m`, `--model` | Optional explicit model override (LangChain providers) |
| `--llm` | Provider shortcut (`openai`, `gemini`, `claude`, `ollama`) |
| `-c`, `--concurrency` | Concurrent insight workers (default: 8) |
| `--no-audit` | Disable optional audit pass |
| `--cache` | SQLite cache path for extraction reuse |
| `--confidence` | Secondary confidence triage threshold |
| `--incremental-jsonl` | Append per-article rows as they complete |

##### Insights output

- `insights.json` output: full machine-readable `InsightJobResult` document
- `.jsonl` output: one `PerArticleInsightResult` per line + `.summary.json` sidecar
- `insight_output_report.md`: human-readable report with:
  - key status metrics
  - finding direction distribution
  - PMIDs for `needs_human_review`, `invalid_output`, and `api_failure`
- Optional `--incremental-jsonl`: crash-resilient append-only per-article progress log
- Optional `--report-md`: custom path for the markdown report (default is `insight_output_report.md` next to output)

#### API key and rate limits

Without an API key, the client spaces requests for roughly **3 requests per second**. With a key, roughly **10 per second** (see [NCBI usage guidelines](https://www.ncbi.nlm.nih.gov/books/NBK25497/)).

CLIs auto-load a local `.env` file (if present), so keys/secrets can be set once.

```bash
cat > .env <<'EOF'
OPENAI_API_KEY=your_openai_key
# or ANTHROPIC_API_KEY=...
# or GEMINI_API_KEY=...
NCBI_API_KEY=your_ncbi_key
NCBI_EMAIL=you@example.com
EOF
```

Switching is still available via a single flag or environment variable:

```bash
export NCBI_API_KEY=your_key_here
uv run collect-pubmed "query" -n 500 -o out.json
# or
uv run collect-pubmed "query" -n 500 -o out.json --api-key your_key_here
```

<a id="output-json"></a>
## Output JSON

The file is a single object:

- `query`, `total_match_count`, `requested_count`, `retrieved_count`
- `articles`: list of records with stable fields such as `pmid`, `title`, `abstract`, `authors`, `doi`, `journal_full`, `mesh_terms`, `keywords`, etc.
- `warnings`: non-fatal issues (for example PMIDs that did not return parseable XML)

All article objects are validated with **Pydantic** before serialization.

<a id="insights-section"></a>
## LLM medical insight classification (evidence-grounded)

This tool reads each collected article (title + abstract) and produces a structured clinical interpretation:

- finding direction (`positive`, `negative`, `neutral`, `mixed`, `unclear`)
- statistical significance
- clinical meaningfulness
- one concise main claim
- supporting evidence spans

Why use an LLM here:

- biomedical abstracts are free-text and nuanced
- the same concept can be expressed many different ways
- extracting meaning + evidence spans from natural language is hard to do with fixed rules only

Why not only deterministic code:

- regex/rules can validate and catch contradictions, but they are weak at full semantic extraction
- a purely rule-based system would miss many phrasing variants and become brittle
- this workflow uses LLM for extraction, then deterministic checks for trust

After you have a `CollectionOutput` JSON from `collect-pubmed`, the pipeline runs:

- **Pass 1**: one-article LLM extraction (JSON)
- **Pass 2**: deterministic validation (schema + grounding + semantic flags)
- **Pass 3**: optional structured audit only when risk triggers fire

- **Evidence**: the model must return verbatim **evidence spans** from the title/abstract; the system checks that each span occurs in a canonical concatenation of title + abstract.
- **Trust**: `PerArticleStatus` is split into `auto_accepted` (highest trust), `validated_but_flagged` (schema/grounding valid with caveats, including some prefilter-routed minimal outputs), and `needs_human_review` (not trustworthy enough to auto-accept), plus `invalid_output` / `api_failure` / `skipped_prefilter`.
- **Confidence policy**: model-reported confidence is treated as a **secondary triage signal** (recorded in reasons), not a primary trust gate. Primary gates are schema validity, grounding, deterministic contradiction flags, and high-risk labels (`mixed`, `meaningful`).
- **Providers**: set one of `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GEMINI_API_KEY` (insights use **LangChain** chat models). You can override the model explicitly with `--model`, or set provider defaults in `.env` via `INSIGHT_MODEL_OPENAI`, `INSIGHT_MODEL_CLAUDE`, `INSIGHT_MODEL_GEMINI` (e.g. `gpt-4o-mini`, `claude-3-5-sonnet-20241022`, `gemini-2.0-flash`).
  For local Ollama, use `--llm ollama` (or `--insight-llm ollama` in workflow) with `OLLAMA_BASE_URL` and `OLLAMA_MODEL`.
- **Caching**: optional `--cache /path/to/cache.sqlite` keys on PMID + input hash + prompt version + model.
- **Incremental persistence**: optional `--incremental-jsonl /path/file.jsonl` appends one `PerArticleInsightResult` per completed article (raw LLM text, extraction/validation/audit if present, final status), so long jobs are recoverable and auditable.
- **Lineage metadata**: each per-article result row includes `prompt_version`, `model_name`, `input_hash`, and `validator_version` for reproducibility across prompt/rule changes.
- **Prefilter routing**: prefilter is a routing decision, not a quality judgement. `no/short abstract` routes to deterministic minimal `unclear` output (`validated_but_flagged`) with explicit notes like `skipped_prefilter_no_abstract`; non-primary publication types route to `skipped_prefilter_non_primary_research:*`.

```bash
uv run classify-insights results.json -o insights.json --llm openai
# JSONL per-article lines + .summary.json sidecar:
uv run classify-insights results.json -o insights.jsonl --llm openai
```

**Prompt version** is `PROMPT_VERSION` in `article_miner.infrastructure.insights.prompts` (bump when instructions change to invalidate caches).

**Offline evaluation**: see [`scripts/eval_insight_metrics.py`](scripts/eval_insight_metrics.py) (optional `pandas` / `scikit-learn` for metrics). For rigorous work, build a gold set (75–150 articles) and report accuracy / macro-F1 / **precision on auto-accept** rows.

**Runtime metrics** (from `InsightJobResult.stats` in the JSON / `.summary.json` sidecar): `input_tokens` / `output_tokens` (when the provider returns usage), counts for `auto_accepted`, `validated_but_flagged`, `needs_review`, `invalid_output`, `api_failure`, `skipped_prefilter`, and `truncation_warning` (when canonical title+abstract exceeds the configured length). Derive schema pass rate, grounding pass rate, and human-review rate from per-article rows.

<a id="finding-probable-duplicates"></a>
## Definition of duplicates

Duplicate detection is intentionally **precision-first** and produces **probable** groups for review (not automatic deletion).

Two records are considered linked when one of these rules matches:

1. **Same normalized DOI** (highest confidence).
2. **Same normalized title + same non-null publication year**.
3. **High fuzzy title similarity** (with abstract similarity checks when both abstracts are present).
4. **Optional:** **SPECTER 2** document embeddings (sentence-transformers) and **FAISS** neighbor search on cosine similarity—enabled with `--specter` or `ARTICLE_MINER_SPECTER=1`, and requiring the `[specter]` optional install.

Then links are merged transitively into clusters (connected components). This means a cluster may contain mixed link evidence (for example DOI + fuzzy links), so reviewer interpretation should use pairwise `edge_evidence`, not only the cluster label.

What this does **not** do:

- It does not merge on PMID alone.
- It does not auto-resolve retraction/replacement logic; those are only flagged for reviewer attention.
- It does not claim perfect recall; thresholds are tuned to reduce false positives.

Run:

```bash
uv run find-pubmed-dupes results.json -o dupes.json -m dupes.md
```

<a id="behavior-notes"></a>
## Behavior notes

- **Search pagination**: `esearch` returns at most **10,000** IDs per call; larger requests use multiple pages.
- **Detail batches**: `efetch` IDs are requested in batches (default **200** per call) to stay within practical limits.
- **Errors**: Transient HTTP failures and 5xx responses use bounded retries with backoff; repeated **429** responses surface as a clear error. Malformed JSON/XML yields `MalformedResponseError`-style messages at the CLI.

<a id="architecture"></a>
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
- `article_miner/application/dedup/`: duplicate detection workflow/service (rule-based + optional SPECTER 2 / FAISS).
- `article_miner/infrastructure/dedup/`: embedding + FAISS helpers for the optional vector layer.
- `article_miner/infrastructure/insights/`: **LangChain** chat models, prompts, extraction, deterministic validation, cache/prefilter; **LangGraph** placeholder graph for future multi-step orchestration.
- `article_miner/common/`: shared cross-tool utilities (for example, project path helpers used by CLIs).

<a id="live-ncbi-smoke-test-optional"></a>
### Live NCBI smoke test (optional)

Unit tests mock HTTP. To exercise the real E-utilities stack (same path as the CLI):

```bash
ARTICLE_MINER_LIVE_NCBI=1 uv run pytest tests/test_live_pubmed_collect.py -v -m live_ncbi
```

Requires network access. The live module runs the same assertion against several broad queries (e.g. diabetes, hypertension, COVID-19, machine learning, publication type). Without `ARTICLE_MINER_LIVE_NCBI=1`, those tests are skipped so CI stays offline by default.

<a id="license"></a>
## License

Add your license here.
