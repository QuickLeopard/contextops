"""Dataset loading for v0.2.

Loads golden QA datasets from JSON, JSONL, or CSV. Each row has at minimum
a query and an expected answer; context is optional.
"""

from __future__ import annotations

import csv
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class DatasetItem:
    """One row of a golden dataset."""

    query: str
    expected: str = ""
    context: str = ""
    metadata: dict | None = None


def load(path: str | Path) -> list[DatasetItem]:
    """Auto-detect format from extension and load."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")

    suffix = p.suffix.lower()
    if suffix == ".jsonl":
        items = _load_jsonl(p)
    elif suffix == ".json":
        items = _load_json(p)
    elif suffix == ".csv":
        items = _load_csv(p)
    else:
        raise ValueError(f"Unsupported format: {suffix}. Use .json, .jsonl, or .csv.")

    _validate(items, source=str(p))
    return items


def _validate(items: list[DatasetItem], *, source: str) -> None:
    """Warn (non-fatal) about dataset rows likely to skew eval results.

    Empty queries, empty expected answers, and exact-duplicate queries don't
    make a dataset unusable, but they silently distort judge scores and
    cost/latency estimates — worth surfacing rather than failing outright.
    """
    if not items:
        warnings.warn(f"Dataset '{source}' is empty — eval will score 0 rows.", stacklevel=2)
        return

    n_empty_query = sum(1 for it in items if not it.query.strip())
    if n_empty_query:
        warnings.warn(
            f"Dataset '{source}': {n_empty_query}/{len(items)} row(s) have an empty query.",
            stacklevel=2,
        )

    n_empty_expected = sum(1 for it in items if not it.expected.strip())
    if n_empty_expected:
        warnings.warn(
            f"Dataset '{source}': {n_empty_expected}/{len(items)} row(s) have no expected "
            "answer — 'completeness' scoring will be unreliable for these rows.",
            stacklevel=2,
        )

    seen: dict[str, int] = {}
    for it in items:
        if it.query.strip():
            seen[it.query] = seen.get(it.query, 0) + 1
    n_dupes = sum(1 for count in seen.values() if count > 1)
    if n_dupes:
        warnings.warn(
            f"Dataset '{source}': {n_dupes} duplicate query value(s) found — "
            "results may double-count some rows.",
            stacklevel=2,
        )


def _load_jsonl(p: Path) -> list[DatasetItem]:
    items = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(_from_dict(json.loads(line)))
    return items


def _load_json(p: Path) -> list[DatasetItem]:
    data = json.loads(p.read_text())
    if isinstance(data, list):
        return [_from_dict(d) for d in data]
    if isinstance(data, dict) and "items" in data:
        return [_from_dict(d) for d in data["items"]]
    raise ValueError("JSON must be a list of objects, or {'items': [...]}.")


def _load_csv(p: Path) -> list[DatasetItem]:
    items = []
    with p.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append(_from_dict(dict(row)))
    return items


def _from_dict(d: dict) -> DatasetItem:
    return DatasetItem(
        query=d.get("query") or d.get("question") or d.get("q") or "",
        expected=d.get("expected") or d.get("answer") or d.get("a") or "",
        context=d.get("context") or d.get("ctx") or "",
        metadata={k: v for k, v in d.items() if k not in {"query", "question", "q", "expected", "answer", "a", "context", "ctx"}},
    )


def to_jsonl(items: list[DatasetItem], path: str | Path) -> None:
    """Dump a dataset back to JSONL — handy for tests and examples."""
    p = Path(path)
    with p.open("w") as f:
        for item in items:
            d = {"query": item.query, "expected": item.expected, "context": item.context}
            if item.metadata:
                d.update(item.metadata)
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def iter_batches(items: list[DatasetItem], batch_size: int) -> Iterator[list[DatasetItem]]:
    """Yield batches of items for streaming eval."""
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]