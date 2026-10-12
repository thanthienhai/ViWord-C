import json

import pytest

from scripts.evaluate_protect import evaluate


def _setup(tmp_path, doc, label):
    segmented = tmp_path / "segmented"
    segmented.mkdir()
    (segmented / "toy.jsonl").write_text(json.dumps({"id": "a", "doc": doc}, ensure_ascii=False) + "\n",
                                         encoding="utf-8")
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps(label, ensure_ascii=False) + "\n", encoding="utf-8")
    return str(labels), str(segmented)


DOC = [[["không", "R", "O"], ["đi", "V", "O"]], [["Hà Nội", "Np", "B-LOC"], ["đẹp", "A", "O"]]]


def blind(gold, sentence=None, words=None):
    words = words or [[0, "không", "R", "O"], [1, "đi", "V", "O"],
                      [2, "Hà Nội", "Np", "B-LOC"], [3, "đẹp", "A", "O"]]
    return {"id": "a", "source": "toy", "sentence": sentence, "sheet": "blind-v1",
            "annotator": "test", "words": words, "gold_tiers": gold}


def test_protect_report_counts_tier_precision_recall(tmp_path):
    labels, segmented = _setup(tmp_path, DOC, blind({"0": "T1"}))
    report = evaluate(labels, segmented, min_rows=1)
    assert report["tiers"]["T1"]["tp"] == 1  # "không" found
    assert report["tiers"]["T1"]["fp"] == 1  # the detector also marks "Hà Nội"; the annotator did not
    assert report["tiers"]["T1"]["recall"] == 1.0


def test_sentence_level_rows_score_only_that_sentence(tmp_path):
    labels, segmented = _setup(tmp_path, DOC, blind({}, sentence=1, words=[[0, "Hà Nội", "Np", "B-LOC"],
                                                                            [1, "đẹp", "A", "O"]]))
    report = evaluate(labels, segmented, min_rows=1)
    assert report["n_without_p"] == 1
    assert report["tiers"]["T1"]["predicted"] == 1 and report["tiers"]["T1"]["fp"] == 1


def test_detector_prefilled_labels_are_rejected(tmp_path):
    old_format = {"id": "a", "source": "toy", "annotator": "manual-v1", "gold_tiers": {"0": "T1"}}
    labels, segmented = _setup(tmp_path, DOC, old_format)
    with pytest.raises(ValueError, match="blind sheet"):
        evaluate(labels, segmented, min_rows=1)


def test_unfilled_or_drifted_sheets_are_rejected(tmp_path):
    labels, segmented = _setup(tmp_path, DOC, blind(None))
    with pytest.raises(ValueError, match="not filled in"):
        evaluate(labels, segmented, min_rows=1)
    drifted = blind({}, words=[[0, "không đi", "V", "O"]])
    (tmp_path / "labels.jsonl").write_text(json.dumps(drifted, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differ from the current segmentation"):
        evaluate(labels, segmented, min_rows=1)


def test_short_label_file_is_rejected(tmp_path):
    labels, segmented = _setup(tmp_path, DOC, blind({}))
    with pytest.raises(ValueError, match="expected at least 2"):
        evaluate(labels, segmented, min_rows=2)
