"""Test helpers: tiny hand-segmented documents and a whitespace tokenizer.

The tests run on CPU without models, Java or network access.
"""
import pytest

from viword.segment import Word


class WhitespaceTokenizer:
    """One token per whitespace-separated piece; mimics the two HF calls we use."""

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def __call__(self, pieces, is_split_into_words=True, add_special_tokens=False, **kwargs):
        return {"input_ids": list(range(len(pieces)))}


def W(text, pos="N", ner="O"):
    return Word(text, pos, ner)


@pytest.fixture
def tokenizer():
    return WhitespaceTokenizer()


@pytest.fixture
def sentence():
    # "Học sinh không đi học ở Hà Nội ."
    return [W("Học sinh"), W("không", "R"), W("đi", "V"), W("học", "V"), W("ở", "E"),
            W("Hà Nội", "Np", "B-LOC"), W(".", "CH")]
