# Key Architecture Decisions

## 1) One article per LLM call (instead of multi-article batching)

**Decision:** Each LLM request handles exactly one article.

**Why:** I optimized for traceability, auditability, robustness, and failure isolation. With one-article calls, every response maps to one PMID, retries are targeted, and malformed outputs do not contaminate neighboring records. This makes each decision easier to audit end-to-end (raw response -> validation -> final status), and makes the pipeline more robust under partial failures or interruptions. It also enables per-article incremental persistence and precise status accounting (`auto_accepted`, `invalid_output`, etc.).

**Tradeoff:** Higher request overhead and potentially lower throughput than packing multiple articles into one prompt.

**Result:** I gained reliability, debuggability, and safer recovery behavior (partial job completion is meaningful), while sacrificing some raw API efficiency.

---

## 2) Reduce LLM usage and make calls cost-aware

**Decision:** LLM is used for extraction first; deterministic/local recovery is preferred before extra LLM calls.

**What I implemented:**

- Prefilter routing before LLM (`skip` or minimal deterministic `unclear` output for no/short abstracts).
- Strict parse -> local JSON repair (fence removal, first-object extraction, trailing comma cleanup) -> only then one LLM repair call.
- Optional audit call only on risk triggers (not every article).
- Incremental persistence to avoid re-paying work after interruptions.

**Tradeoff:** Lower cost/latency and better operational resilience, but potentially fewer “rescued” cases than aggressive multi-pass LLM repair.

**Result:** Cost-effective pipeline where LLM is the extractor, not the default error handler.

---

## 3) LLM vs rule-based classification

**Decision:** Use an LLM for classification, with deterministic validation as guardrails.

**Tradeoff:**

- Rule-based: precise but brittle (fails on varied scientific language)
- LLM-based: flexible but can hallucinate or overgeneralize

**Resolution:**

- I use the LLM for semantic interpretation (finding direction, significance, clinical meaning).
- This includes nuanced fields such as **Clinical meaningfulness**, which are difficult to capture with fixed rules alone.
- I require verbatim evidence spans for each field.
- I apply deterministic checks for:
  - evidence grounding (span must appear in source text)
  - contradiction detection (high-precision hard errors, softer warnings otherwise)
  - enum/schema constraints

**Result:** This hybrid design keeps outputs expressive while making them verifiable and auditable. The system prioritizes precision and auditability, which is critical for medical use cases.

- **Reliability vs cost:** deterministic guards + selective LLM usage
- **LLM vs rules:** LLM for interpretation, rules for validation and trust
- **Trust vs coverage:** precision-first acceptance (`auto_accepted`) with explicit flagged paths when certainty is lower

