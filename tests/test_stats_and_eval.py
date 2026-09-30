import math

import numpy as np
import pytest

from viword.eval import parse_prediction, rouge_l, rouge_n
from viword.stats import cluster_bootstrap, holm, mcnemar_exact, paired_test


def test_cluster_bootstrap_is_wider_than_example_bootstrap():
    # 10 clusters of 20 identical examples: the effective sample size is 10, not 200
    values = np.repeat(np.arange(10, dtype=float), 20)
    clusters = [str(v) for v in np.repeat(np.arange(10), 20)]
    by_cluster = np.std(cluster_bootstrap(values, clusters, 2000))
    by_example = np.std(cluster_bootstrap(values, [str(i) for i in range(200)], 2000))
    assert by_cluster > 3 * by_example


def test_single_cluster_is_an_error():
    with pytest.raises(ValueError):
        cluster_bootstrap([1.0, 2.0], ["a", "a"])


def test_paired_test_direction():
    a, b = [1.0] * 30, [0.0] * 30
    result = paired_test(a, b, [str(i % 10) for i in range(30)], n_boot=500)
    assert result["diff"] == 1.0 and result["p_one_sided"] == 0.0 and result["n_clusters"] == 10


def test_holm():
    adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert math.isclose(adjusted["a"], 0.03)
    assert math.isclose(adjusted["c"], 0.06)
    assert math.isclose(adjusted["b"], 0.06)  # monotone


def test_mcnemar():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(10, 0) < 0.01


def test_rouge_on_vietnamese():
    assert rouge_n("học sinh đi học", "học sinh đi học", 1) == 1.0
    # LCS = "hà nội mưa": P = 3/4, R = 3/5, F = 2/3
    assert rouge_l("Hà Nội mưa to", "hôm nay Hà Nội mưa") == pytest.approx(2 / 3)
    assert rouge_n("không", "có", 1) == 0.0


def test_parse_prediction():
    assert parse_prediction("mc", "Đáp án: B") == "B"
    assert parse_prediction("nli", "Contradiction.") == "contradiction"
    assert parse_prediction("nli", "không rõ") == ""
