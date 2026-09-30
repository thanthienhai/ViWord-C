import numpy as np
from conftest import W

from viword.baselines import LeadK, NegationProbe, Truncation
from viword.compressor import Context, ScoredCompressor, SelectionConfig
from viword.model import make_windows, pool_to_words
from viword.segment import syllables_of
from viword.select import TokenCounter


class FixedScorer:
    """Returns given syllable logits, standing in for an encoder."""

    def __init__(self, logits):
        self.logits = np.array(logits, dtype=float)

    def syllable_logits(self, doc):
        return self.logits


def context_of(sentence):
    return Context(text=" ".join(w.text for w in sentence), doc=[sentence])


def test_word_unit_never_breaks_words(sentence, tokenizer):
    # "Học" is confident, "sinh" is not: a syllable-level decision breaks the word
    logits = [3.0, -3.0, 2.0, 1.0, 1.0, -2.0, 0.5, 0.5, -1.0]
    counter = TokenCounter(tokenizer)
    ctx = context_of(sentence)
    syllable = ScoredCompressor(FixedScorer(logits), SelectionConfig(unit="syllable"), "s")
    word = ScoredCompressor(FixedScorer(logits), SelectionConfig(unit="word"), "w")
    # syllable decision keeps "Học" without "sinh" (a broken word, P1)
    assert syllable.compress(ctx, 3, counter) == "Học không đi"
    # word decision scores "Học sinh" as a whole (mean logit 0) and keeps whole words only
    assert word.compress(ctx, 3, counter) == "không đi học"


def test_hard_protection_keeps_negation(sentence, tokenizer):
    logits = [2.0, 2.0, -5.0, 2.0, 2.0, 2.0, -5.0, -5.0, -5.0]  # "không" has a low score
    counter = TokenCounter(tokenizer)
    config = SelectionConfig(unit="word", protect="hard", tiers=("T1",))
    out = ScoredCompressor(FixedScorer(logits), config, "w").compress(context_of(sentence), 4, counter)
    assert "không" in out.split()


def test_lead_and_truncation(sentence, tokenizer):
    counter = TokenCounter(tokenizer)
    ctx = context_of(sentence)
    assert LeadK().compress(ctx, 3, counter) == "Học sinh không"
    # head and tail alternate: Học sinh (2) + "." (1) + không (1) + Hà Nội (2) = 6 tokens
    assert Truncation().compress(ctx, 6, counter) == "Học sinh không Hà Nội."


def test_negation_probe(sentence, tokenizer):
    out = NegationProbe().compress(context_of(sentence), 0, TokenCounter(tokenizer))
    assert out == "Học sinh đi học ở Hà Nội."


def test_pool_to_words(sentence):
    values = np.arange(len(syllables_of(sentence)), dtype=float)
    pooled = pool_to_words(values, sentence)
    assert pooled[0] == 0.5 and pooled[5] == 6.5


def test_windows_cover_document_and_overlap(tokenizer):
    doc = [[W(f"s{i}{k}") for k in range(4)] for i in range(5)]  # 5 sentences × 4 syllables
    windows = make_windows(doc, tokenizer, max_length=10)  # 8 subwords per window
    assert windows[0][0] == 0 and windows[-1][1] == 20
    for (a, b), (c, d) in zip(windows, windows[1:]):
        assert c < b  # consecutive windows overlap by one sentence
