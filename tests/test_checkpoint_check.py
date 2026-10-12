import pytest

from scripts.checkpoint_check import render_report, summarize, verify


def _rows(task="vinli", values=None, n=2):
    values = values or {"truncation": 0.50, "lexprior": 0.60, "viword_s0": 0.70}
    rows = []
    for method, score in values.items():
        for i in range(n):
            rows.append({
                "task": task,
                "method": method,
                "ratio": 0.333,
                "id": f"{task}-{i}",
                "accuracy": score,
                "rougeL": score,
            })
    return rows


def test_checkpoint_check_requires_beating_truncation_and_lexprior():
    summary = summarize(_rows(), expected_per_task=2)
    assert summary["passed"]
    assert summary["baseline_methods"] == ["truncation", "lexprior"]

    failed = summarize(
        _rows(values={"truncation": 0.50, "lexprior": 0.70, "viword_s0": 0.65}),
        expected_per_task=2,
    )
    assert not failed["passed"]
    assert not failed["comparisons"][0]["beats_lexprior"]


def test_majority_of_dev_tasks_is_enough():
    # loses to truncation on VietNews (lead bias) but wins on ViNLI and ViMMRC
    rows = (_rows("vinli") + _rows("vimmrc")
            + _rows("vietnews", {"truncation": 0.20, "lexprior": 0.10, "viword_s0": 0.18}))
    summary = summarize(rows, expected_per_task=2)
    assert summary["passed"]
    rows = (_rows("vinli")
            + _rows("vimmrc", {"truncation": 0.8, "lexprior": 0.1, "viword_s0": 0.7})
            + _rows("vietnews", {"truncation": 0.20, "lexprior": 0.10, "viword_s0": 0.18}))
    assert not summarize(rows, expected_per_task=2)["passed"]


def test_llmlingua2vi_is_reported_but_does_not_gate():
    rows = _rows(values={"truncation": 0.5, "lexprior": 0.6, "viword_s0": 0.7, "llmlingua2vi_s0": 0.1})
    summary = summarize(rows, expected_per_task=2)
    assert summary["passed"]
    reference = next(c for c in summary["checkpoints"] if c["method"] == "llmlingua2vi_s0")
    assert not reference["gated"] and not reference["passed"]
    assert "reference" in render_report(summary)


def test_checkpoint_check_requires_the_full_200_example_slice():
    summary = summarize(_rows(), expected_per_task=200)
    assert not summary["passed"]
    assert summary["comparisons"][0]["n"] == 2


def test_eval_verification_blocks_a_failed_report(tmp_path):
    report = tmp_path / "checkpoint.json"
    report.write_text('{"passed": false}\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="eval is blocked"):
        verify(str(report))

