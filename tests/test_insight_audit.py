"""Tests for structured audit parsing."""

from article_miner.infrastructure.insights.llm_extract import parse_audit_json


def test_parse_audit_json_structured_payload() -> None:
    raw = """
    {
      "supported": true,
      "finding_direction": "supported",
      "statistical_significance": "supported",
      "clinical_meaningfulness": "weakly_supported",
      "main_claim": "unsupported",
      "notes": ["Clinical meaningfulness lacks strong magnitude evidence."]
    }
    """
    out = parse_audit_json(raw)
    assert out is not None
    assert out.supported is True
    assert out.finding_direction == "supported"
    assert out.clinical_meaningfulness == "weakly_supported"
    assert out.main_claim == "unsupported"
    assert out.notes


def test_parse_audit_json_invalid_verdicts_fallback_to_unsupported() -> None:
    raw = """
    {
      "supported": false,
      "finding_direction": "maybe",
      "statistical_significance": "unknown",
      "clinical_meaningfulness": "supported",
      "main_claim": "unsupported",
      "notes": "single note"
    }
    """
    out = parse_audit_json(raw)
    assert out is not None
    assert out.finding_direction == "unsupported"
    assert out.statistical_significance == "unsupported"
    assert out.clinical_meaningfulness == "supported"
    assert out.notes == ["single note"]

