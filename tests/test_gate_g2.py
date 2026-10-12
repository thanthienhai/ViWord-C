import sys
from pathlib import Path

import pytest

from scripts.gate_eval import G2_NFR_THRESHOLD, main as gate_eval_main
from scripts.gate_report import G2_TEST_PAIR_THRESHOLD, count_negated_test_pairs


def _segmented(identifier, syllables):
    return {
        "id": identifier,
        "doc": [[[*syllable, "O"] for syllable in syllables]],
    }


def test_g2_test_pair_count_ignores_nli_labels():
    segmented = [
        _segmented("p0", [("không", "R"), ("đi", "V")]),
        _segmented("p1", [("đi", "V")]),
    ]
    premises = [
        {"id": "p0", "cluster": "c0", "label": "entailment"},
        {"id": "p1", "cluster": "c1", "label": "contradiction"},
    ]
    pairs = [
        {"id": "x0", "cluster": "c0", "label": "neutral"},
        {"id": "x1", "cluster": "c0", "label": "entailment"},
        {"id": "x2", "cluster": "c1", "label": "contradiction"},
    ]

    original = count_negated_test_pairs(segmented, premises, pairs)
    for row in premises + pairs:
        row["label"] = "the label is deliberately changed"
    changed = count_negated_test_pairs(segmented, premises, pairs)

    assert original == changed
    assert original["pairs_with_negated_premise"] == 2
    assert original["pairs"] == 3


def test_g2_thresholds_match_the_registered_gate():
    assert G2_NFR_THRESHOLD == 0.05
    assert G2_TEST_PAIR_THRESHOLD == 300


def test_gate_eval_rejects_test_split(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["gate_eval.py", "--split", "test", "--rows", "does-not-need-to-exist.jsonl"],
    )
    with pytest.raises(SystemExit, match="dev-only"):
        gate_eval_main()

