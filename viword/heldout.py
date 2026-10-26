"""Test rows that stay unseen until the test evaluation (DATN §4.3, amendment of 2026-10-22).

During development, readers scored some test rows: smoke runs, prechecks and trials always took the first
rows of a file (`--limit`). config/test_exclusions.json lists those rows (scripts/test_exclusions.py). They
leave the primary analysis, which is then run only on rows no reader had scored before the test run; the full
test sets stay as a sensitivity analysis. The Belebele ones form `belebele_dev`, since Belebele has no dev split.

`refuse_unseen_test_rows` keeps the remaining rows unseen: a reader may score them only in the eval stage,
which sets VIWORD_TEST_EVAL=1 after its own checks. Rows are matched by content, so a renamed or truncated
copy of a test file is refused too.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .data import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
EXCLUSIONS = "config/test_exclusions.json"
ALLOW_TEST_ENV = "VIWORD_TEST_EVAL"
# file stem -> task name of every test file (Belebele has a single split, used as test)
TEST_FILES = {"belebele": "belebele", "vinli_test": "vinli", "vimmrc_test": "vimmrc",
              "vietnews_test": "vietnews", "viquad_test": "viquad", "gsm8k_test": "gsm8k",
              "oneruler_test": "oneruler"}
# Direction B (legal RAG) keeps its own test split, outside the registered exclusion list: no reader has scored
# any of its rows, so every row is unseen, and RAG rows built from it carry derived_from = "vlegal_test" (or
# "vibid_test").
LEGAL_TEST_FILES = {"vlegal_test": "vlegal", "vibid_test": "vibid"}
ALL_TEST_FILES = {**TEST_FILES, **LEGAL_TEST_FILES}


def load_exclusions(path: str | Path = EXCLUSIONS) -> dict[str, dict]:
    """{test file stem: {"task": name, "ids": set of excluded ids}}."""
    with open(path, encoding="utf-8") as f:
        sets = json.load(f)["sets"]
    return {stem: {"task": s["task"], "ids": set(s["ids"])} for stem, s in sets.items()}


def exclusions_digest(path: str | Path = EXCLUSIONS) -> str:
    """Hash of the excluded ids, not of the file bytes (git may rewrite line endings)."""
    sets = {stem: sorted(s["ids"]) for stem, s in load_exclusions(path).items()}
    return hashlib.sha256(json.dumps(sets, sort_keys=True).encode("utf-8")).hexdigest()


def excluded_ids(exclusions: dict[str, dict], data_path: str | Path) -> set[str]:
    """Excluded ids of one test file, looked up by its stem (data/tasks/vinli_test.jsonl -> vinli_test)."""
    stem = Path(data_path).name.removesuffix(".jsonl")
    if stem not in exclusions:
        raise ValueError(f"{data_path} is not a test file listed in the exclusions ({sorted(exclusions)})")
    return exclusions[stem]["ids"]


def excluded_by_task(exclusions: dict[str, dict]) -> dict[str, set[str]]:
    """Result rows carry the task, not the file: {task: excluded ids}."""
    return {s["task"]: s["ids"] for s in exclusions.values()}


def drop_excluded(rows: list[dict], exclusions: dict[str, dict]) -> list[dict]:
    """Result rows of the primary analysis: those of rows no reader scored before the test run."""
    by_task = excluded_by_task(exclusions)
    return [r for r in rows if r["id"] not in by_task.get(r["task"], ())]


def fingerprint(row: dict) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def unseen_fingerprints(tasks_dir: str | Path, exclusions: dict[str, dict]) -> set[str]:
    """Fingerprints of the test rows that no reader has scored yet."""
    unseen = set()
    for stem in ALL_TEST_FILES:
        path = Path(tasks_dir) / f"{stem}.jsonl"
        if path.exists():
            skip = exclusions.get(stem, {}).get("ids", set())
            unseen.update(fingerprint(row) for row in read_jsonl(path) if row["id"] not in skip)
    return unseen


def refuse_unseen_test_rows(rows: list[dict], tasks_dir: str | Path = ROOT / "data/tasks",
                            exclusions_path: str | Path = ROOT / EXCLUSIONS, env=None) -> None:
    """Raise SystemExit if a reader is about to score an unseen test row outside the eval stage.
    Without the exclusions file every test row counts as unseen."""
    env = os.environ if env is None else env
    if env.get(ALLOW_TEST_ENV) == "1":
        return
    exclusions = load_exclusions(exclusions_path) if Path(exclusions_path).exists() else {}
    unseen = unseen_fingerprints(tasks_dir, exclusions)
    hits = [row["id"] for row in rows if fingerprint(row) in unseen]
    # rows built from test passages (scripts/make_long_context.py) are test material too
    hits += [row["id"] for row in rows if row.get("derived_from") in ALL_TEST_FILES]
    if hits:
        raise SystemExit(f"refusing to let a reader score {len(hits)} test row(s) kept for the test run "
                         f"(e.g. {hits[0]}): only `run_pipeline.sh eval` reads them ({ALLOW_TEST_ENV}=1). "
                         "Use a dev set; for Belebele, data/tasks/belebele_dev.jsonl.")
