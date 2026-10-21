import math

import numpy as np
import pytest

from viword.data import Example
from viword.eval import (budget_deviates, build_prompt, context_budget, normalize_number,
                         parse_prediction, rouge_l, rouge_n, score, tokens_per_budget)
from viword.stats import (cluster_bootstrap, flip_rate_test, holm, mcnemar_exact,
                          mcnemar_exact_one_sided, paired_test, paired_test_stratified,
                          stratified_cluster_bootstrap)


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


def test_seed_stratified_bootstrap_resamples_seed_then_cluster():
    values = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
    clusters = ["a", "b", "a", "b", "a", "b"]
    seeds = [0, 0, 1, 1, 2, 2]
    boot = stratified_cluster_bootstrap(values, clusters, seeds, n_boot=500)
    assert np.mean(boot) == pytest.approx(2.0, abs=0.05)
    result = paired_test_stratified(values, np.zeros(6), clusters, seeds, n_boot=500)
    assert result["diff"] == 2.0 and result["n_clusters"] == 2


def test_seed_bootstrap_shares_cluster_draws_across_seeds():
    # Crossed design: every seed scores the same 100 clusters, and cluster difficulty is
    # shared across seeds. The true SD of the 3-seed mean is ~ cluster SD / sqrt(100) = 0.1;
    # independent per-seed cluster draws would give ~0.1 / sqrt(3).
    rng = np.random.default_rng(1)
    effect = rng.normal(0, 1, 100)
    values = np.concatenate([effect + rng.normal(0, 0.05, 100) for _ in range(3)])
    clusters = [f"c{i}" for _ in range(3) for i in range(100)]
    seeds = [s for s in range(3) for _ in range(100)]
    se = np.std(stratified_cluster_bootstrap(values, clusters, seeds, n_boot=4000))
    assert se == pytest.approx(np.std(effect) / 10, rel=0.15)


def test_seed_bootstrap_matches_cluster_bootstrap_when_seeds_agree():
    values = np.tile(np.arange(20, dtype=float), 3)
    clusters = [str(i) for _ in range(3) for i in range(20)]
    seeds = [s for s in range(3) for _ in range(20)]
    crossed = np.std(stratified_cluster_bootstrap(values, clusters, seeds, n_boot=4000))
    plain = np.std(cluster_bootstrap(values[:20], clusters[:20], n_boot=4000))
    assert crossed == pytest.approx(plain, rel=0.1)


def test_flip_rate_test_counts_premises_not_pairs():
    # target flips less on every pair, but all 40 pairs come from 2 premises
    flips_a = [0] * 40
    flips_b = [1] * 20 + [0] * 20
    clusters = ["p1"] * 20 + ["p2"] * 20
    result = flip_rate_test(flips_a, flips_b, clusters, n_boot=2000)
    assert result["diff"] == -0.5 and result["n_clusters"] == 2
    assert result["p_one_sided"] > 0.1  # one exact McNemar on 20 discordant pairs would give ~1e-6
    assert mcnemar_exact_one_sided(0, 20) < 1e-5


def test_mcnemar_one_sided_direction():
    assert mcnemar_exact_one_sided(0, 10) < 0.001
    assert mcnemar_exact_one_sided(10, 0) == 1.0


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


def test_normalize_number():
    assert normalize_number("1.000") == normalize_number("1,000") == "1000"
    assert normalize_number("2,5") == normalize_number("2.5") == "2.5"
    assert normalize_number("18.0") == "18"


def test_math_parse_and_score():
    ex = Example("g", "g", "math", "demos", question="?", answer="1250")
    out = parse_prediction("math", "3 x 400 = 1200, cộng 50.\nĐáp án: 1.250")
    assert out == "1250" and score(ex, out)["accuracy"] == 1.0
    assert parse_prediction("math", "kết quả là 7") == "7"  # no "Đáp án:" line: last number


def test_qa_score_takes_best_answer():
    ex = Example("q", "q", "qa", "ctx", question="?", extra={"answers": ["Hà Nội", "thủ đô Hà Nội"]})
    assert score(ex, "Hà Nội.")["em"] == 1.0
    assert 0 < score(ex, "ở Hà Nội")["f1"] < 1


def test_ruler_prompt_and_recall():
    ex = Example("r", "r", "ruler", "haystack", extra={"answers": ["123", "456"], "subtask": "niah_multivalue",
                                                       "template": "<Văn bản>\n{context}\n</Văn bản> Hỏi?"})
    assert build_prompt(ex, "rơm") == "<Văn bản>\nrơm\n</Văn bản> Hỏi?"
    assert score(ex, "Các số: 123")["recall"] == 0.5


def test_reader_budget_excludes_prompt_and_marks_deviation():
    class Counter:
        @staticmethod
        def count(text):
            return len(text.split())

    original, budget = context_budget("một hai ba bốn", 0.5, Counter())
    assert (original, budget) == (4, 2)
    assert tokens_per_budget(2, budget) == 1.0
    assert not budget_deviates(1.05)
    assert budget_deviates(1.051)
    assert tokens_per_budget(0, 0) is None



def test_a_longer_output_budget_keeps_answers_read_within_the_shorter_one():
    from viword.eval import MAX_NEW_TOKENS
    assert MAX_NEW_TOKENS["nli"] == MAX_NEW_TOKENS["mc"] == 32  # the same for every reader and method
    # greedy decoding only appends text: the first label found in the short answer is still the one read
    for task, short, longer in [("nli", "neutral", "neutral. Tuy nhiên, không có entailment"),
                                ("mc", "B", "B. Vì đoạn văn nói A sai")]:
        assert parse_prediction(task, short) == parse_prediction(task, longer)
    # an opening sentence that used to exhaust 8 tokens now reaches the label
    assert parse_prediction("nli", "Quan hệ giữa tiền đề và giả") == ""
    assert parse_prediction("nli", "Quan hệ giữa tiền đề và giả thuyết là: entailment") == "entailment"
