"""Versioned prompts for evidence-grounded article classification."""

from __future__ import annotations

from med_assert.domain.collect.models import Article

PROMPT_VERSION = "2026.04.08.2"

SYSTEM_PROMPT = """You are an evidence-grounded biomedical article classifier.

Your task is to classify a single PubMed-style article using only the provided title, abstract, and metadata.

Rules:
1. Return JSON only.
2. Use only the allowed label values for each field.
3. Every classification field must include one or more verbatim evidence spans copied exactly from the title or abstract text as provided (character-for-character after trimming leading/trailing whitespace on each span).
4. Do not infer facts not supported by the title or abstract.
5. If the source text is insufficient for a label, use the value "unclear" (where allowed) for that field.
6. Prefer conservative labeling over guessing.
7. If findings conflict in the abstract, use "mixed" for finding_direction.
8. For clinical_meaningfulness, use "meaningful" only when the text supports practical or patient-important importance, not merely statistical significance.

Output must not include markdown fences or commentary outside JSON."""


def build_user_prompt(article: Article) -> str:
    title = article.title or ""
    abstract = article.abstract or ""
    pub_types = (
        ", ".join(article.publication_types) if article.publication_types else "(none)"
    )
    kws = ", ".join(article.keywords) if article.keywords else "(none)"
    return f"""Classify this article.

PMID: {article.pmid}
Title: {title}
Abstract: {abstract}
Publication types: {pub_types}
Keywords: {kws}

Allowed labels:
- finding_direction: positive | negative | neutral | mixed | unclear
- statistical_significance: significant | not_significant | unclear
- clinical_meaningfulness: meaningful | not_meaningful | unclear

Return JSON with this exact structure (all keys required):
{{
  "pmid": "{article.pmid}",
  "finding_direction": {{ "value": "<label>", "confidence": 0.0, "evidence_spans": ["..."] }},
  "statistical_significance": {{ "value": "<label>", "confidence": 0.0, "evidence_spans": ["..."] }},
  "clinical_meaningfulness": {{
    "value": "<label>",
    "confidence": 0.0,
    "evidence_spans": ["..."],
    "reasoning_summary": "short text or empty string"
  }},
  "main_claim": {{ "value": "one concise main claim sentence", "confidence": 0.0, "evidence_spans": ["..."] }},
  "review_flags": []
}}

Important:
- Evidence spans must be exact quotes from the Title or Abstract lines above.
- If a field is unclear, set its value to "unclear" where listed, and still provide evidence spans if any phrase supports that uncertainty.
- main_claim must be a single focused claim, not a broad summary.
- Do not output markdown."""


REPAIR_USER_TEMPLATE = """The following text was supposed to be a single JSON object matching the medical article classification schema, but it may be invalid or truncated.

Fix it: return ONLY valid JSON with the same keys as specified previously. No markdown.

Broken text:
---
{text}
---
"""


AUDIT_USER_TEMPLATE = """You audit a biomedical classification for consistency with quoted evidence.

Article PMID: {pmid}
Title: {title}
Abstract: {abstract}

Proposed JSON classification (for review):
{classification_json}

Return JSON only with this exact shape:
{{
  "supported": true or false,
  "finding_direction": "supported" | "weakly_supported" | "unsupported",
  "statistical_significance": "supported" | "weakly_supported" | "unsupported",
  "clinical_meaningfulness": "supported" | "weakly_supported" | "unsupported",
  "main_claim": "supported" | "weakly_supported" | "unsupported",
  "notes": ["short reason 1", "short reason 2"]
}}

Rules:
- "supported" means clearly justified by quoted evidence.
- "weakly_supported" means plausible but incomplete/indirect evidence.
- "unsupported" means evidence does not justify the field."""


def system_prompt() -> str:
    return SYSTEM_PROMPT
