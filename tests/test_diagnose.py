import math
import random

from conftest import W

from viword.diagnose import cbr, cbr_counts, chance_cbr, retention, retention_counts, spearman, word_type


def test_cbr_counts_partial_and_full(sentence):
    # syllables: Học sinh không đi học ở Hà Nội .
    mask = [True, False, True, True, True, True, True, True, True]  # "Học sinh" broken, "Hà Nội" full
    result = cbr(cbr_counts(sentence, mask))
    assert result["all"] == 0.5 and result["n_all"] == 2
    assert result["name"] == 0.0 and result["other"] == 1.0


def test_fully_dropped_words_are_not_counted(sentence):
    mask = [False, False] + [True] * 7
    assert cbr(cbr_counts(sentence, mask))["n_all"] == 1


def test_chance_cbr_formula_and_simulation():
    assert math.isclose(chance_cbr(1 / 3), 0.8)
    assert math.isclose(chance_cbr(0.5), 2 / 3)
    rng, partial, survived = random.Random(0), 0, 0
    for _ in range(200_000):
        kept = [rng.random() < 0.2, rng.random() < 0.2]
        if any(kept):
            survived += 1
            partial += not all(kept)
    assert abs(partial / survived - chance_cbr(0.2)) < 0.01


def test_word_type():
    assert word_type(W("Hà Nội", "Np", "B-LOC")) == "name"
    assert word_type(W("sạch sẽ", "A")) == "reduplicative"
    assert word_type(W("học sinh", "N")) == "other"


def test_retention_by_tier(sentence):
    mask = [True, True, False, True, True, True, True, True, False]  # drops "không"
    r = retention(retention_counts(sentence, mask))
    assert r["n_T1"] == 2  # "không" and "Hà Nội"
    assert r["T1"] == 0.5
    assert r["n_all"] == 6  # punctuation excluded


def test_spearman():
    assert math.isclose(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
    assert math.isclose(spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
