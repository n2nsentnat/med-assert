#!/usr/bin/env python3
"""Offline evaluation stub: compare gold vs predicted labels (JSONL with pmid + label fields).

Install optional deps for full metrics: ``uv sync --group dev`` and ``pip install pandas scikit-learn``.

Example (after adapting column names to your gold file):
  python scripts/eval_insight_metrics.py gold.jsonl pred.jsonl

Without pandas, prints a short merge count only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Compare gold vs prediction JSONL by pmid.")
    p.add_argument("gold", type=Path)
    p.add_argument("pred", type=Path)
    args = p.parse_args()
    gold = {str(r.get("pmid")): r for r in _load_jsonl(args.gold)}
    pred = {str(r.get("pmid")): r for r in _load_jsonl(args.pred)}
    common = set(gold) & set(pred)
    print(f"Gold rows: {len(gold)}  Pred rows: {len(pred)}  Common pmids: {len(common)}")
    try:
        import pandas as pd  # type: ignore[import-not-found]
        from sklearn.metrics import classification_report  # type: ignore[import-not-found]

        # Expect nested insight.extraction.finding_direction.value when using article_miner export
        def fd(r: dict[str, object]) -> str:
            ins = r.get("insight") or {}
            if isinstance(ins, dict):
                ex = ins.get("extraction") or {}
                if isinstance(ex, dict):
                    fd0 = ex.get("finding_direction") or {}
                    if isinstance(fd0, dict) and "value" in fd0:
                        return str(fd0["value"])
            return str(r.get("finding_direction_gold", "unknown"))

        y_true = [fd(gold[k]) for k in sorted(common)]
        y_pred = [fd(pred[k]) for k in sorted(common)]
        print(classification_report(y_true, y_pred, zero_division=0))
    except ImportError:
        print("Install pandas and scikit-learn for classification_report.")


if __name__ == "__main__":
    main()
