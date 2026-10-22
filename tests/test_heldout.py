"""Test rows readers scored during development stay out of the primary analysis, and the other test rows
stay unseen until the eval stage (DATN §4.3, amendment of 2026-10-22)."""
import json
import re
from pathlib import Path

import pytest

from scripts.mde import count_test_clusters
from scripts.test_exclusions import EXPOSURES, check, exposed_rows, make_belebele_dev
from viword.data import write_jsonl
from viword.heldout import (ALLOW_TEST_ENV, TEST_FILES, drop_excluded, excluded_ids, load_exclusions,
                            refuse_unseen_test_rows)
from viword.segment_meta import build_metadata, verify_metadata, write_metadata

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "config/test_exclusions.json"
SCRIPTS = ROOT / "scripts"


def _rows(prefix, n, per_cluster=1):
    return [{"id": f"{prefix}-{i}", "cluster": f"{prefix}-c{i // per_cluster}", "premise": f"p {i}",
             "hypothesis": "h", "label": "neutral"} for i in range(n)]


def _exclusions(tmp_path, ids):
    path = tmp_path / "test_exclusions.json"
    path.write_text(json.dumps({"sets": {"vinli_test": {"task": "vinli", "ids": ids}}}), encoding="utf-8")
    return path


def test_committed_list_covers_every_audited_read():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["exposures"] == EXPOSURES
    assert set(config["sets"]) == set(TEST_FILES)
    assert {stem: s["rows"] for stem, s in config["sets"].items()} == exposed_rows()
    assert exposed_rows()["belebele"] == 200 and exposed_rows()["vinli_test"] == 300
    for entry in config["sets"].values():
        assert len(entry["ids"]) == len(set(entry["ids"])) == entry["rows"]


def test_committed_list_matches_the_data():
    if not all((ROOT / f"data/tasks/{stem}.jsonl").exists() for stem in TEST_FILES):
        pytest.skip("prepared test files not present")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert check(config, ROOT / "data/tasks") == []
    for stem, entry in config["sets"].items():  # every development read took the first rows of the file
        lines = (ROOT / f"data/tasks/{stem}.jsonl").read_text(encoding="utf-8").splitlines()
        assert [json.loads(line)["id"] for line in lines[:entry["rows"]]] == entry["ids"]


def test_a_reader_may_not_score_an_unseen_test_row(tmp_path):
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    rows = _rows("vinli-test", 5)
    write_jsonl(tasks / "vinli_test.jsonl", rows)
    exclusions = _exclusions(tmp_path, ["vinli-test-0", "vinli-test-1"])
    refuse_unseen_test_rows(rows[:2], tasks, exclusions, env={})  # rows readers have already scored
    with pytest.raises(SystemExit, match="vinli-test-2"):
        refuse_unseen_test_rows(rows[:3], tasks, exclusions, env={})
    copy = [dict(reversed(list(rows[4].items())))]  # same content under another file name and key order
    with pytest.raises(SystemExit):
        refuse_unseen_test_rows(copy, tasks, exclusions, env={})
    refuse_unseen_test_rows(rows, tasks, exclusions, env={ALLOW_TEST_ENV: "1"})  # the eval stage
    refuse_unseen_test_rows(_rows("vinli-dev", 3), tasks, exclusions, env={})
    with pytest.raises(SystemExit):  # without the list, every test row is unseen
        refuse_unseen_test_rows(rows[:1], tasks, tmp_path / "missing.json", env={})


def test_primary_analysis_drops_the_scored_rows_and_counts_the_remaining_clusters(tmp_path):
    exclusions = load_exclusions(_exclusions(tmp_path, ["vinli-test-0", "vinli-test-1"]))
    rows = [{"task": "vinli", "id": f"vinli-test-{i}"} for i in range(4)] + [{"task": "vimmrc", "id": "vinli-test-0"}]
    assert [r["id"] for r in drop_excluded(rows, exclusions)] == ["vinli-test-2", "vinli-test-3", "vinli-test-0"]
    test_file = tmp_path / "vinli_test.jsonl"
    write_jsonl(test_file, _rows("vinli-test", 6, per_cluster=2))
    assert count_test_clusters([f"vinli={test_file}"]) == {"vinli": 3}
    assert count_test_clusters([f"vinli={test_file}"], exclusions) == {"vinli": 2}  # cluster c0 is gone
    with pytest.raises(ValueError, match="not a test file"):
        excluded_ids(exclusions, tmp_path / "vinli_dev.jsonl")


def test_mde_report_must_use_the_committed_exclusions(tmp_path, monkeypatch):
    from scripts import mde

    exclusions = _exclusions(tmp_path, ["vinli-test-0"])
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"dev_only": True, "passed": True, "rouge_margin_check": {"margin_points": 0.5},
                                  "h1_contrast": "viword − llmlingua2vi", "reader": "r",
                                  "test_exclusions_sha256": "an older list"}), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["mde.py", "--verify", str(report), "--exclude", str(exclusions)])
    with pytest.raises(SystemExit, match="recompute"):
        mde.main()


def test_belebele_dev_is_a_verified_subset_of_belebele(tmp_path):
    tasks, segmented = tmp_path / "tasks", tmp_path / "segmented"
    tasks.mkdir()
    segmented.mkdir()
    rows = [{"id": f"b#{i}", "cluster": f"b{i // 2}", "context": f"c {i}", "question": "q",
             "choices": ["1", "2", "3", "4"], "answer": "A"} for i in range(6)]
    write_jsonl(tasks / "belebele.jsonl", rows)
    seg = segmented / "belebele.jsonl"
    write_jsonl(seg, [{"id": r["id"], "doc": []} for r in rows])
    write_metadata(seg, build_metadata(seg, segmenter="vncorenlp"))
    from viword.segment_meta import sha256_path
    config = {"sets": {"belebele": {"task": "belebele", "ids": ["b#0", "b#1", "b#2"],
                                    "file_sha256": sha256_path(tasks / "belebele.jsonl")}}}

    assert make_belebele_dev(config, tasks, segmented) == 3
    dev_seg = segmented / "belebele_dev.jsonl"
    assert [json.loads(line)["id"] for line in (tasks / "belebele_dev.jsonl").read_text(encoding="utf-8").splitlines()] \
        == ["b#0", "b#1", "b#2"]
    assert verify_metadata(dev_seg) == []
    with open(dev_seg, "a", encoding="utf-8") as f:  # a line that is not in belebele.jsonl
        f.write('{"id": "b#9", "doc": []}\n')
    meta = json.loads((segmented / "belebele_dev.jsonl.meta.json").read_text(encoding="utf-8"))
    meta["output_sha256"] = sha256_path(dev_seg)
    (segmented / "belebele_dev.jsonl.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    assert any("not in its source" in e for e in verify_metadata(dev_seg))
    make_belebele_dev(config, tasks, segmented)
    with open(seg, "a", encoding="utf-8") as f:  # the source changes after the subset was taken
        f.write('{"id": "b#6", "doc": []}\n')
    assert any("has changed" in e for e in verify_metadata(dev_seg))


def _stage(text, name):
    match = re.search(rf"^{name}\(\) \{{[^\n]*\n(.*?)^\}}", text, re.S | re.M)
    assert match, name
    return match.group(1)


def test_only_the_eval_stage_lets_a_reader_read_unseen_test_rows():
    pipeline = (SCRIPTS / "run_pipeline.sh").read_text(encoding="utf-8")
    eval_body = _stage(pipeline, "stage_eval")
    assert eval_body.count(f"{ALLOW_TEST_ENV}=1 python scripts/evaluate.py") == 2
    assert eval_body.index("require_committed_exclusions") < eval_body.index(f"{ALLOW_TEST_ENV}=1")
    others = pipeline.replace(eval_body, "")
    for name in ("ops_full_run.sh", "full_h100.sh", "smoke_h100.sh"):
        others += (SCRIPTS / name).read_text(encoding="utf-8")
    assert ALLOW_TEST_ENV not in others


def test_smoke_and_precheck_read_dev_sets_only():
    smoke = (SCRIPTS / "smoke_h100.sh").read_text(encoding="utf-8")
    smoke_eval = _stage(smoke, "step_eval")
    assert "data/tasks/belebele_dev.jsonl" in smoke_eval and "data/tasks/vinli_dev.jsonl" in smoke_eval
    assert "_test" not in smoke_eval and "data/tasks/belebele.jsonl" not in smoke_eval
    ops = (SCRIPTS / "ops_full_run.sh").read_text(encoding="utf-8")
    precheck = _stage(ops, "precheck")
    assert "data/tasks/belebele_dev.jsonl" in precheck and "data/tasks/belebele.jsonl" not in precheck
    assert "test_exclusions.py belebele-dev" in _stage(ops, "check_data")
    assert "test_exclusions.py belebele-dev" in _stage(smoke, "step_check_data")


def test_analysis_teacher_bound_and_mde_use_the_exclusions():
    pipeline = (SCRIPTS / "run_pipeline.sh").read_text(encoding="utf-8")
    analyze = _stage(pipeline, "stage_analyze")
    assert 'analyze_hypotheses --exclude "$TEST_EXCLUSIONS"' in analyze
    assert analyze.count("analyze_hypotheses") == 2  # primary, then sensitivity on every row
    assert '--exclude "$TEST_EXCLUSIONS" --limit "$TEACHER_TEST_LIMIT"' in _stage(pipeline, "stage_teacher_test")
    assert '--exclude "$TEST_EXCLUSIONS" --limit "$TEACHER_TEST_LIMIT"' in _stage(pipeline, "stage_eval")
    assert '--exclude "$TEST_EXCLUSIONS"' in _stage(pipeline, "stage_mde")
    assert '--margin-points "$MDE_MARGIN" --reader "$PRIMARY_READER" \\\n    --exclude "$TEST_EXCLUSIONS"' \
        in _stage(pipeline, "stage_eval")
