import json

import pytest

from scripts.leak_filter import filter_rows, load_eval_records, vietnews_id
from viword.distill import LeakIndex


def _write(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_vietnews_stable_id_normalizes_train_and_eval_prefixes():
    assert vietnews_id("news-123") == "123"
    assert vietnews_id("vietnews-test-123") == "123"
    assert vietnews_id("vietnews-dev-123") == "123"
    assert vietnews_id("wiki-123") is None


def test_filter_rows_rejects_vietnews_id_even_without_an_ngram_match():
    rows = [
        {"id": "news-123", "source": "vietnews", "text": "một đoạn hoàn toàn khác"},
        {"id": "news-456", "source": "vietnews", "text": "một đoạn hoàn toàn khác"},
    ]
    clean, dropped = filter_rows(rows, LeakIndex(["bối cảnh đánh giá khác biệt"], n=3), {"123"})
    assert [row["id"] for row in clean] == ["news-456"]
    assert dropped[("vietnews", "vietnews_guid_conservative")] == 1


def test_load_eval_records_requires_sources_dev_and_test(tmp_path):
    names = ["vinli_dev", "vinli_test", "vietnews_dev", "vietnews_test",
             "vimmrc_dev", "vimmrc_test", "belebele"]
    paths = []
    for name in names:
        path = tmp_path / f"{name}.jsonl"
        field = "premise" if name.startswith("vinli") else "context"
        _write(path, [{"id": f"{name}-1", field: "một văn bản đánh giá đủ dài"}])
        paths.append(str(path))
    rows, coverage = load_eval_records(paths)
    assert len(rows) == len(names)
    assert coverage["vinli"]["dev"] == 1
    with pytest.raises(FileNotFoundError, match="missing evaluation file"):
        load_eval_records(paths[:-1] + [str(tmp_path / "missing.jsonl")])

