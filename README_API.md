# med-assert HTTP API

FastAPI service exposing **collect** (PubMed), **dedup** (duplicate groups, including optional **SPECTER 2** embeddings with **FAISS** similarity search), and **insights** (LLM classification via LangChain). Request and response models are defined with Pydantic; interactive docs are served by FastAPI.

## Run the server

From the project root, after `uv sync`:

```bash
uv run med-assert-api
```

This starts Uvicorn on `http://127.0.0.1:8000` by default.

Alternatively, invoke Uvicorn directly (useful for extra flags):

```bash
uv run uvicorn med_assert.interfaces.api.app:app --host 0.0.0.0 --port 8000 --reload
```

(`med_assert.interfaces.api.http_app:app` is equivalent; the `app` module is a thin re-export.)

## Interactive documentation

- **Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **OpenAPI JSON:** [http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json)

## Environment

The API loads the same `.env` / project env as the CLI (`load_project_env()` on collect and insights).

| Concern | Typical variables |
|--------|-------------------|
| PubMed / NCBI | `NCBI_API_KEY`, `NCBI_EMAIL` (optional; can also be sent in the collect JSON body) |
| Insights (OpenAI) | `OPENAI_API_KEY`, optional `INSIGHT_MODEL_OPENAI` |
| Insights (Gemini) | `GEMINI_API_KEY`, optional `INSIGHT_MODEL_GEMINI` |
| Insights (Claude) | `ANTHROPIC_API_KEY`, optional `INSIGHT_MODEL_CLAUDE` |
| Insights (Ollama) | `OLLAMA_BASE_URL`, `OLLAMA_MODEL` |

## Output format and file paths

Every mutating endpoint accepts:

| Field | Values | Description |
|-------|--------|-------------|
| `output_format` | `json` (default) or `file` | `json` returns the full result in the HTTP body. `file` writes results on the **server** filesystem and returns a small JSON acknowledgement with absolute paths. |
| `output_path` | string or omitted | Used when `output_format` is `file`. Absolute or relative path for the primary output file. If omitted, a default under the server process working directory is used (see below). |

**Default paths** (when `output_format` is `file` and `output_path` is null): paths are relative to the server’s current working directory:

| Endpoint | Default file |
|----------|----------------|
| `POST /collect` | `med_assert_output/collect.json` |
| `POST /dedup` | Reads **input** from `collection_path` (see below). Writes `med_assert_output/dedup.json` by default (optional sibling `dedup.md` when `include_markdown` is true) |
| `POST /insights` | Reads **input** from `collection_path`. Writes `med_assert_output/insights.json` or `insights.jsonl` by default (see `insight_file_format`) |

**File-mode response** (`output_format: "file"`):

```json
{
  "output_format": "file",
  "paths": {
    "collection_json": "/abs/path/...",
    "report_json": "...",
    "markdown": "...",
    "json": "...",
    "jsonl": "...",
    "summary_json": "...",
    "report_md": "..."
  }
}
```

Only relevant keys are present (e.g. insights JSON mode sets `json` and optionally `report_md`; JSONL mode sets `jsonl`, `summary_json`, and optionally `report_md`).

**Security:** `output_path` is interpreted on the server host. Restrict deployment or permissions so clients cannot write outside intended directories.

## Endpoints

### `GET /health`

Liveness check.

**Response:** `{"status":"ok"}`

---

### `POST /collect`

Search PubMed and return structured article metadata as JSON (same shape as the CLI `collect-pubmed` output).

**Request body (JSON)**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | string | required | PubMed / Entrez query |
| `count` | integer | `100` | Max articles (≥ 1) |
| `api_key` | string \| null | `null` | NCBI API key |
| `email` | string \| null | `null` | Contact email for NCBI |
| `tool` | string | `"med_assert"` | Tool name sent to NCBI |
| `output_format` | string | `"json"` | `json` or `file` |
| `output_path` | string \| null | `null` | Server path for `collect.json` when `output_format` is `file` |

**Success:** `200` — either `CollectionOutput` in the body (`output_format: json`), or `FileWriteResponse` with `paths.collection_json` (`output_format: file`).

**Errors**

| Status | When |
|--------|------|
| `400` | Invalid arguments (e.g. bad count) |
| `422` | Request body validation failed |
| `500` | Failed to write output file |
| `502` | PubMed / NCBI or transport errors |

**Examples**

```bash
curl -s -X POST http://127.0.0.1:8000/collect \
  -H 'Content-Type: application/json' \
  -d '{"query":"diabetes mellitus[tiab]","count":3}'
```

Write to disk on the server (default path):

```bash
curl -s -X POST http://127.0.0.1:8000/collect \
  -H 'Content-Type: application/json' \
  -d '{"query":"diabetes mellitus[tiab]","count":3,"output_format":"file"}'
```

---

### `POST /dedup`

Build a duplicate-group report from a **collection JSON file already on the server** (same `CollectionOutput` schema as `/collect` returns or writes).

**Request body (JSON)**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `collection_path` | string | required | Path on the **server** to a `CollectionOutput` JSON file (e.g. path returned as `paths.collection_json` from `POST /collect` with `output_format=file`) |
| `include_markdown` | boolean | `false` | Include Markdown (in JSON response or as a file when using `file` mode) |
| `enable_specter_faiss` | boolean | `false` | Optional SPECTER 2 embeddings + FAISS layer (requires `pip install 'med-assert[specter]'` on the server) |
| `specter_model` | string \| null | `null` | Override HF model id (default `allenai/specter2_base`) |
| `output_format` | string | `"json"` | `json` or `file` |
| `output_path` | string \| null | `null` | Server path for dedup JSON when `output_format` is `file` |

**Success:** `200` — either `DedupApiResponse` (`report`, optional `markdown` in the body) or `FileWriteResponse` with `paths.report_json` and, when `include_markdown` is true, `paths.markdown` (sibling `.md` file next to the JSON path).

**Errors**

| Status | When |
|--------|------|
| `404` | `collection_path` does not exist or is not a file |
| `422` | File is not valid JSON or does not match `CollectionOutput` |

**Example**

```bash
curl -s -X POST http://127.0.0.1:8000/dedup \
  -H 'Content-Type: application/json' \
  -d '{"collection_path":"/path/on/server/collect.json","include_markdown":true}'
```

---

### `POST /insights`

Run the async LLM insight job on a **collection JSON file on the server** (same `CollectionOutput` schema as `/collect`; typically the same path you used for `/dedup`). Returns an `InsightJobResult` (per-article rows, aggregate `stats`, etc.).

**Request body (JSON)**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `collection_path` | string | required | Path on the **server** to a `CollectionOutput` JSON file |
| `llm` | string \| null | `null` | Provider shortcut: `openai`, `gemini`, `claude`, `ollama` (model from env / defaults) |
| `model` | string \| null | `null` | Explicit model id when `llm` is not set (provider inferred) |
| `concurrency` | integer | `8` | Parallel articles (1–64) |
| `enable_audit` | boolean | `true` | Enable optional audit pass |
| `confidence_threshold` | number | `0.5` | 0.0–1.0 |
| `cache_path` | string \| null | `null` | Optional SQLite cache path on the server |
| `progress` | boolean | `false` | Enable progress logging |
| `progress_every` | integer | `1` | Progress log frequency |
| `output_format` | string | `"json"` | `json` or `file` |
| `output_path` | string \| null | `null` | Server path for the main insights file when `output_format` is `file` |
| `insight_file_format` | string | `"json"` | `json` or `jsonl` (JSONL also writes a `.summary.json` next to the JSONL, same as CLI) |
| `write_report_md` | boolean | `true` | When writing files, also write `insight_output_report.md` beside the main output |

If the path has no `.json` / `.jsonl` suffix, an extension is added from `insight_file_format`. If both `llm` and `model` are omitted, the server uses `INSIGHT_MODEL_OPENAI` or falls back to `gpt-4o-mini`.

**Success:** `200` — either full `InsightJobResult` JSON or `FileWriteResponse` with paths for the written files.

**Errors**

| Status | When |
|--------|------|
| `400` | Unknown `llm` provider name |
| `404` | `collection_path` does not exist or is not a file |
| `422` | File is not valid JSON, does not match `CollectionOutput`, or insight job validation failed |

**Example**

```bash
curl -s -X POST http://127.0.0.1:8000/insights \
  -H 'Content-Type: application/json' \
  -d '{
    "collection_path": "/path/on/server/collect.json",
    "llm": "openai",
    "concurrency": 4
  }'
```

Long-running jobs: this request stays open until the job finishes; use a client with a sufficient timeout or run behind a task queue if you need async job IDs (not included in this API).

## Typical pipeline

1. `POST /collect` with `output_format=file` → note `paths.collection_json` on the server
2. `POST /dedup` with `{"collection_path": "<that path>"}` → review duplicate groups (or return JSON in the response body with default `output_format=json`)
3. `POST /insights` with `{"collection_path": "<same or updated collect path>"}` (and LLM env configured) → structured insights

## See also

- Main project documentation: [README.md](README.md)
