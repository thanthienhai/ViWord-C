"""A1 (one Unicode/tone normalization at every entry point), A4 (sliding windows),
D1 (the P table of DATN §3.4) and F5 (Selective Context token-to-word assignment)."""
import json
import unicodedata

import numpy as np
import pytest
from conftest import W

from viword.compressor import Context
from viword.protect import TIER_OF_WORD, protection_tiers
from viword.segment import align_compressed, normalize


# ---------------------------------------------------------------------------- A1

def test_load_contexts_normalizes_nfd_and_old_tone_placement(tmp_path):
    from viword.data import load_contexts, load_task
    nfd_old = unicodedata.normalize("NFD", "Thanh Hoá")
    (tmp_path / "t.jsonl").write_text(json.dumps({"id": "a", "cluster": "a", "premise": nfd_old,
                                                  "hypothesis": "h", "label": "neutral"}) + "\n",
                                      encoding="utf-8")
    (tmp_path / "s.jsonl").write_text(json.dumps({"id": "a", "doc": [[["Thanh Hóa", "Np", "B-LOC"]]]}) + "\n",
                                      encoding="utf-8")
    [ctx] = load_contexts(load_task("vinli", str(tmp_path / "t.jsonl")), str(tmp_path / "s.jsonl"))
    assert ctx.text == "Thanh Hóa"  # NFC and new-style tone placement


def test_budget_is_identical_for_nfd_and_nfc_contexts():
    from viword.eval import context_budget
    from viword.select import TokenCounter

    class Chars:
        def encode(self, text, add_special_tokens=False):
            return list(text)

    nfc = "Thanh Hóa"
    nfd = unicodedata.normalize("NFD", "Thanh Hoá")
    counter = TokenCounter(Chars())
    assert context_budget(normalize(nfd), 0.5, counter) == context_budget(nfc, 0.5, counter)


def test_teacher_output_in_the_other_tone_style_still_aligns():
    words = [W("hòa bình", "N"), W("khỏe", "A")]  # segmented, normalized source
    mask, unmatched = align_compressed(words, "hoà bình khoẻ")  # teacher kept the raw old style
    assert mask == [True, True, True] and unmatched == 0.0


def test_vncorenlp_word_forms_are_normalized(monkeypatch):
    from viword.segment import VnCoreNLPSegmenter

    class FakeModel:
        def annotate_text(self, text):
            # VnCoreNLP rewrites new-style tone placement to the old style
            return {0: [{"wordForm": "Thanh_Hoá", "posTag": "Np", "nerLabel": "B-LOC"}]}

    seg = VnCoreNLPSegmenter.__new__(VnCoreNLPSegmenter)
    seg.model = FakeModel()
    [[word]] = seg.segment("Thanh Hóa")
    assert word.text == "Thanh Hóa"


# ---------------------------------------------------------------------------- A4

class CountingTokenizer:
    """Every syllable is one subword; mimics the HF calls of make_windows/SyllableScorer."""

    class Encoding(dict):
        def __init__(self, n):
            import torch
            super().__init__(input_ids=torch.zeros((1, n + 2), dtype=torch.long))
            self.n = n

        def word_ids(self):
            return [None] + list(range(self.n)) + [None]

    def __call__(self, pieces, is_split_into_words=True, add_special_tokens=False, truncation=False,
                 max_length=None, return_tensors=None):
        if return_tensors is None:
            return {"input_ids": list(range(len(pieces)))}
        return self.Encoding(len(pieces))


def test_long_sentences_are_cut_and_every_syllable_is_scored():
    torch = pytest.importorskip("torch")
    from viword.model import SyllableScorer, make_windows

    doc = [[W(f"a{i}") for i in range(450)], [W(f"b{i}") for i in range(30)]]  # one 450-syllable sentence
    windows = make_windows(doc, CountingTokenizer(), max_length=212)  # 210 subwords per window
    covered = set()
    for start, end in windows:
        assert end - start <= 210
        covered.update(range(start, end))
    assert covered == set(range(480))

    class Model:
        def __call__(self, input_ids, **kwargs):
            class Out:
                logits = torch.ones((1, input_ids.shape[1], 1))
            return Out()

    scorer = SyllableScorer.__new__(SyllableScorer)
    scorer.torch, scorer.device, scorer.tokenizer = torch, "cpu", CountingTokenizer()
    scorer.model, scorer.max_length, scorer.keep_label = Model(), 212, 1
    logits = scorer.syllable_logits(doc)
    assert len(logits) == 480 and np.all(logits == 1.0)  # no syllable left at the neutral 0


# ---------------------------------------------------------------------------- D1

DATN_TABLE = {
    "T1": ["không", "chưa", "chẳng", "chả", "đừng", "chớ", "không hề", "chưa từng",
           "nếu", "trừ khi", "giá mà", "dù"],
    "T2": ["đã", "đang", "sẽ", "vừa", "mới", "sắp", "từng", "phải", "nên", "cần", "có thể", "được phép"],
    "T3": ["mọi", "mỗi", "tất cả", "chỉ", "hơn", "nhất", "kém"],
}


def test_word_list_matches_the_datn_table_exactly():
    expected = {w: tier for tier, words in DATN_TABLE.items() for w in words}
    assert TIER_OF_WORD == expected
    for excluded in ("các", "những", "cái", "con"):
        assert excluded not in TIER_OF_WORD


def test_every_table_entry_is_protected_in_a_plain_context():
    # POS chosen so that the disambiguation rules accept the word; "kém" needs "hơn" after it
    pos = {"mới": "R", "chỉ": "R", "phải": "V", "nên": "V"}
    for tier, words in DATN_TABLE.items():
        for w in words:
            sentence = [W(w, pos.get(w, "R")), W("hơn" if w == "kém" else "đi", "V")]
            assert protection_tiers(sentence)[0] == tier, w


def test_numbers_and_entities_are_t1_and_excluded_words_are_not_protected():
    sentence = [W("Hà Nội", "Np", "B-LOC"), W("có", "V"), W("3", "M"), W("các", "L"), W("con", "Nc")]
    assert protection_tiers(sentence) == ["T1", None, "T1", None, None]


# ---------------------------------------------------------------------------- F5

def test_selective_context_sums_token_surprisal_per_word():
    from viword.baselines import SelectiveContext
    from viword.select import TokenCounter

    class FakeLM:
        # tokens: "học" | " sinh" | " đi" with character spans; the leading space belongs to the word
        def surprisal(self, text):
            assert text == "học sinh đi"
            return [1.0, 2.0, 5.0], [(0, 3), (3, 8), (8, 11)]

    class Chars:
        def encode(self, text, add_special_tokens=False):
            return list(text)

    sc = SelectiveContext(FakeLM())
    words = [W("học sinh"), W("đi", "V")]
    # word scores: "học sinh" = 1 + 2 = 3, "đi" = 5; with room for one word only, "đi" wins
    out = sc.compress(Context(text="học sinh đi", doc=[words]), len(" đi"), TokenCounter(Chars()))
    assert out == "đi"
