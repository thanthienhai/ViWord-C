"""Drop distillation paragraphs that overlap an evaluation set (DATN §3.2).

Both checks run before querying the teacher. The 13-token n-gram check is the leak
check. The VietNews GUID check is only a conservative extra filter: `guid` in
`nam194/vietnews` is a per-split row index, not an article id (101 GUIDs occur in both
dev and test with different articles), so a GUID match does NOT indicate the same
article. It is kept so that the frozen paragraph snapshot (data/SHA256SUMS) stays
reproducible; it only removes ~100 unrelated paragraphs and cannot cause a leak.

Usage:
  python scripts/leak_filter.py --input data/distill/paragraphs.jsonl \
      --eval-files data/tasks/*.jsonl --output data/distill/paragraphs_clean.jsonl
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from viword.data import read_jsonl, write_jsonl
from viword.distill import LeakIndex


REQUIRED_SOURCES = {"vinli", "vietnews", "vimmrc", "belebele"}
REQUIRED_DEV_SOURCES = {"vinli", "vietnews", "vimmrc"}


def eval_source(path: str) -> str:
    """Infer the normalized dataset name from a task JSONL filename."""
    stem = Path(path).stem.lower()
    return stem.split("_", 1)[0]


def eval_split(path: str) -> str:
    """Return ``dev``/``test`` or ``all`` for a task JSONL filename."""
    stem = Path(path).stem.lower()
    return stem.split("_", 1)[1] if "_" in stem else "all"


def context_of(row: dict) -> str:
    return row.get("context") or row.get("premise") or ""


def vietnews_id(record_id: object) -> str | None:
    """Return the VietNews GUID (a per-split row index, see the module docstring)."""
    value = str(record_id or "")
    if value.startswith("news-"):
        return value.removeprefix("news-")
    for prefix in ("vietnews-dev-", "vietnews-test-", "vietnews-train-"):
        if value.startswith(prefix):
            return value.removeprefix(prefix)
    return None


def load_eval_records(paths: list[str]) -> tuple[list[dict], dict]:
    """Load evaluation contexts and enforce the required test/dev coverage."""
    if not paths:
        raise ValueError("no evaluation files were supplied")
    records, coverage = [], {}
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"missing evaluation file: {path}")
        rows = list(read_jsonl(str(path)))
        source, split = eval_source(str(path)), eval_split(str(path))
        contexts = [context_of(row) for row in rows if context_of(row).strip()]
        if not rows:
            raise ValueError(f"evaluation file is empty: {path}")
        if not contexts:
            raise ValueError(f"evaluation file has no context/premise text: {path}")
        coverage.setdefault(source, {})[split] = len(contexts)
        records.extend(rows)

    missing_sources = sorted(REQUIRED_SOURCES - set(coverage))
    if missing_sources:
        raise ValueError("missing required evaluation sources: " + ", ".join(missing_sources))
    missing_dev = sorted(source for source in REQUIRED_DEV_SOURCES
                         if "dev" not in coverage.get(source, {}))
    if missing_dev:
        raise ValueError("missing required dev files: " + ", ".join(missing_dev))
    missing_test = sorted(source for source in REQUIRED_DEV_SOURCES
                          if "test" not in coverage.get(source, {}))
    if missing_test:
        raise ValueError("missing required test files: " + ", ".join(missing_test))
    return records, coverage


def filter_rows(rows: list[dict], index: LeakIndex, eval_vietnews_ids: set[str]) -> tuple[list[dict], Counter]:
    """Apply stable-id and n-gram checks, returning clean rows and reason counts."""
    clean, dropped = [], Counter()
    for row in rows:
        reason = None
        if vietnews_id(row.get("id")) in eval_vietnews_ids:
            reason = "vietnews_guid_conservative"
        elif index.leaks(row["text"]):
            reason = "ngram"
        if reason:
            dropped[(row.get("source") or "?", reason)] += 1
        else:
            clean.append(row)
    return clean, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--eval-files", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    try:
        eval_rows, coverage = load_eval_records(args.eval_files)
        index = LeakIndex([context_of(r) for r in eval_rows])
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"leak_filter: {exc}") from exc

    rows = list(read_jsonl(args.input))
    eval_vietnews_ids = {key for r in eval_rows if (key := vietnews_id(r.get("id")))}
    clean, dropped = filter_rows(rows, index, eval_vietnews_ids)
    dropped_by_source = Counter()
    dropped_by_reason = Counter()
    for (source, reason), count in dropped.items():
        dropped_by_source[source] += count
        dropped_by_reason[reason] += count
    write_jsonl(args.output, clean)
    print(json.dumps({
        "evaluation_contexts_indexed": len(eval_rows),
        "evaluation_coverage": coverage,
        "kept": len(clean),
        "input_paragraphs": len(rows),
        "dropped_by_source": dict(dropped_by_source),
        "dropped_by_reason": dict(dropped_by_reason),
        "vietnews_eval_ids_indexed": len(eval_vietnews_ids),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
