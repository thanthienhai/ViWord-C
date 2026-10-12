import math
import random
import unicodedata
from collections import Counter

from conftest import W

from viword.diagnose import (cbr, cbr_counts, chance_cbr, chance_cbr_distribution, retention,
                             recovery_rate, retention_counts, spearman, word_type)


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


def test_chance_cbr_uses_actual_multi_syllable_length_mix():
    words = [W("một hai"), W("một hai ba"), W("một hai ba bốn")]
    lengths = [2, 3, 4]
    expected = chance_cbr_distribution(0.2, lengths)
    assert not math.isclose(expected, chance_cbr(0.2))

    rng, counts = random.Random(1), Counter()
    for _ in range(100_000):
        mask = [rng.random() < 0.2 for n in lengths for _ in range(n)]
        counts += cbr_counts(words, mask)
    assert abs(cbr(counts)["all"] - expected) < 0.01


class _TwoPieceTokenizer:
    mask_token = "<mask>"

    def encode(self, text, add_special_tokens=False):
        return ["ha", "i"] if text == "hai" else [text]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)


class _TwoPieceMLM:
    tokenizer = _TwoPieceTokenizer()

    def __init__(self):
        self.inputs = []

    def __call__(self, text, top_k=1):
        self.inputs.append(text)
        return [[{"token_str": "ha"}], [{"token_str": "i"}]]


def test_recovery_rate_uses_one_mask_per_subword():
    mlm = _TwoPieceMLM()
    assert recovery_rate([W("một hai")], [True, False], mlm) == 1.0
    assert mlm.inputs == ["một <mask> <mask>"]


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


class _OnePieceMLM:
    """Predicts the NFC form of a syllable that is stored in NFD in the document."""

    class tokenizer:
        mask_token = "<mask>"

        @staticmethod
        def encode(text, add_special_tokens=False):
            return [text]

        @staticmethod
        def convert_tokens_to_string(tokens):
            return "".join(tokens)

    def __call__(self, text, top_k=1):
        return [{"token_str": unicodedata.normalize("NFC", "Hóa")}]


def test_recovery_rate_compares_unicode_normalized_forms():
    nfd = unicodedata.normalize("NFD", "Thanh Hóa")
    assert recovery_rate([W(nfd)], [True, False], _OnePieceMLM()) == 1.0
