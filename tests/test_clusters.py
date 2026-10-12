from pathlib import Path

import numpy as np
import pytest

from viword.data import CLUSTER_SEMANTICS, read_jsonl
from viword.stats import cluster_bootstrap


ROOT = Path(__file__).parents[1]


def test_registered_cluster_semantics_are_explicit():
    assert CLUSTER_SEMANTICS == {
        "vinli": "premise",
        "vimmrc": "passage",
        "belebele": "link",
        "vietnews": "article",
    }


@pytest.mark.parametrize(
    ("path", "expected"),
    [("data/tasks/belebele.jsonl", 488), ("data/tasks/vinli_test.jsonl", 377)],
)
def test_real_task_rows_have_registered_cluster_counts(path, expected):
    file_path = ROOT / path
    if not file_path.exists():
        pytest.skip(f"prepared data not present: {path}")
    rows = read_jsonl(file_path)
    assert len({row["cluster"] for row in rows}) == expected


def test_cluster_definition_matches_evaluate_output_shape():
    """Use the same fields written by evaluate.py, not a reduced fixture format."""
    path = ROOT / "data/tasks/vinli_test.jsonl"
    if not path.exists():
        pytest.skip("prepared ViNLI data not present")
    source_rows = read_jsonl(path)[:20]
    evaluate_rows = [
        {
            "task": "vinli",
            "reader": "fixture-reader",
            "id": row["id"],
            "cluster": row["cluster"],
            "method": "fixture",
            "ratio": 0.333,
            "budget": 1,
            "n_tokens_original": 1,
            "n_tokens_compressed": 1,
            "prediction": "neutral",
            "accuracy": float(i % 2 == 0),
        }
        for i, row in enumerate(source_rows)
    ]
    boot = cluster_bootstrap(
        np.asarray([row["accuracy"] for row in evaluate_rows]),
        [row["cluster"] for row in evaluate_rows],
        n_boot=200,
    )
    assert len(boot) == 200
    assert len({row["cluster"] for row in evaluate_rows}) == len(source_rows)
    assert all(set(row) >= {"task", "id", "cluster", "method", "ratio", "accuracy"}
               for row in evaluate_rows)


def test_cluster_bootstrap_rejects_misaligned_evaluate_columns():
    with pytest.raises(ValueError, match="length mismatch"):
        cluster_bootstrap([1.0, 0.0], ["premise-a"])
