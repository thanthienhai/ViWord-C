import json

import pytest

from scripts.evaluate_probe import evaluate


def _write(tmp_path, labels):
    segmented = tmp_path / "segmented"
    segmented.mkdir(exist_ok=True)
    (segmented / "toy.jsonl").write_text(
        "\n".join([
            json.dumps({"id": "yes", "doc": [[["không", "R", "O"], ["đi", "V", "O"]]]}),
            json.dumps({"id": "fp", "doc": [[["chưa", "R", "O"], ["kể", "V", "O"]]]}),
            json.dumps({"id": "no", "doc": [[["không", "R", "O"], ["lâu", "A", "O"]]]}),
        ]) + "\n",
        encoding="utf-8",
    )
    path = tmp_path / "labels.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in labels) + "\n", encoding="utf-8")
    return str(path), str(segmented)


def row(i, gold, **extra):
    return {"id": i, "source": "toy", "cluster": i, "sheet": "blind-v1", "annotator": "test",
            "gold_negation": gold, **extra}


def test_probe_audit_reports_precision_and_recall_from_both_halves(tmp_path):
    labels, segmented = _write(tmp_path, [row("yes", True), row("fp", False), row("no", False)])
    report = evaluate(labels, segmented, min_rows=2)
    assert report["predicted_probe_targets"] == 2 and report["n_unedited"] == 1
    assert report["precision"] == 0.5
    assert report["recall"] == 1.0


def test_recall_is_not_reported_without_unedited_premises(tmp_path):
    labels, segmented = _write(tmp_path, [row("yes", True), row("fp", True)])
    report = evaluate(labels, segmented, min_rows=2)
    assert report["recall"] is None and report["f1"] is None


def test_probe_audit_needs_enough_distinct_edited_premises(tmp_path):
    labels, segmented = _write(tmp_path, [row("yes", True), row("no", False)])
    with pytest.raises(ValueError, match="at least 2 distinct premises edited"):
        evaluate(labels, segmented, min_rows=2)
    labels, segmented = _write(tmp_path, [row("yes", True), row("fp", True, cluster="yes")])
    with pytest.raises(ValueError, match="annotated twice"):
        evaluate(labels, segmented, min_rows=1)


def test_non_blind_labels_are_rejected(tmp_path):
    labels, segmented = _write(tmp_path, [{"id": "yes", "source": "toy", "annotator": "manual-v1",
                                           "gold_negation": True}])
    with pytest.raises(ValueError, match="blind sheet"):
        evaluate(labels, segmented, min_rows=1)
