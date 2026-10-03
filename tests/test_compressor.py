import numpy as np
from conftest import W

from viword.baselines import LeadK, NegationProbe, Precomputed, TfidfSentences, Truncation
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


def test_sentence_baseline_fills_with_head_of_next_sentence(sentence, tokenizer):
    counter = TokenCounter(tokenizer)
    # one sentence longer than the budget: keep its head instead of returning nothing
    assert TfidfSentences().compress(context_of(sentence), 4, counter) == "Học sinh không đi"
    # a short sentence that fits is kept whole, the rest of the budget goes to the long one
    short = [W("Trời"), W("mưa", "V"), W(".", "CH")]
    ctx = Context(text="", doc=[short, sentence])
    out = TfidfSentences().compress(ctx, 6, counter)
    assert out.startswith("Trời mưa.") and counter.count(out) <= 6


def test_precomputed_matches_rounded_ratio_and_trims(tmp_path, tokenizer):
    path = tmp_path / "teacher.jsonl"
    path.write_text('{"id": "a", "ratio": 0.333, "compressed": "một hai ba bốn năm"}\n', encoding="utf-8")
    counter = TokenCounter(tokenizer)
    ctx = Context(text="", example_id="a", ratio=1 / 3)
    assert Precomputed(str(path)).compress(ctx, 3, counter) == "một hai ba bốn năm"
    assert Precomputed(str(path), trim=True).compress(ctx, 3, counter) == "một hai ba"


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
