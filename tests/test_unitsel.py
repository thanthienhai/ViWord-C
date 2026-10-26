"""Direction B's method: utility labels, the unit scorer's training targets, MIG ordering, and unit selection."""
import pytest
from conftest import W

from viword.compressor import Context, ScoredCompressor, SelectionConfig
from viword.select import TokenCounter

TEXT = ("Điều 1. Phạm vi\nThuế nhập khẩu hàng hóa .\n\nĐiều 2. Bảo đảm dự thầu\nNhà thầu nộp bảo đảm dự thầu .\n\n"
        "Điều 3. Hồ sơ\nHồ sơ dự thầu gồm đơn dự thầu .")
WORDS = [[W(x) for x in ["Điều", "1", ".", "Phạm vi", "Thuế", "nhập khẩu", "hàng hóa", "."]],
         [W(x) for x in ["Điều", "2", ".", "Bảo đảm", "dự thầu", "Nhà thầu", "nộp", "bảo đảm", "dự thầu", "."]],
         [W(x) for x in ["Điều", "3", ".", "Hồ sơ", "Hồ sơ", "dự thầu", "gồm", "đơn", "dự thầu", "."]]]


class Fixed:
    """A unit scorer with given scores."""

    def __init__(self, scores):
        self.given = scores

    def scores(self, question, units):
        assert len(units) == len(self.given)
        return list(self.given)


def test_mig_order_demotes_a_near_duplicate():
    from viword.unitsel import mig_order
    texts = ["thuế nhập khẩu hàng hóa", "thuế nhập khẩu hàng hóa nhập khẩu", "bảo đảm dự thầu"]
    assert mig_order([3.0, 2.9, 2.0], texts) == [0, 1, 2]
    assert mig_order([3.0, 2.9, 2.0], texts, lam=3.0) == [0, 2, 1]
    assert mig_order([1.0, 1.0], ["a", "b"]) == [0, 1]  # ties keep the context order


def test_unit_selection_keeps_whole_units_then_the_head_of_the_next(tokenizer):
    from viword.unitsel import UnitSelectCompressor
    counter = TokenCounter(tokenizer)
    ctx = Context(text=TEXT, doc=WORDS, question="bảo đảm dự thầu?")
    comp = UnitSelectCompressor(Fixed([0.0, 5.0, 1.0]))
    out = comp.compress(ctx, 18, counter)  # unit 2 (13 tokens + 2) fits whole, unit 3 gives its head
    assert out.startswith("Điều 2. Bảo đảm") and "Điều 3" in out and "Thuế" not in out
    assert counter.count(out) <= 18
    out = comp.compress(ctx, 5, counter)  # the best unit alone is too long: its head only
    assert out.startswith("Điều 2") and "Hồ sơ" not in out and counter.count(out) <= 5


def test_unit_selection_compresses_the_partial_unit_with_token_scores(tokenizer):
    import numpy as np

    from viword.unitsel import UnitSelectCompressor

    class Favour:  # syllable logits that favour the syllables of "nộp"
        def syllable_logits(self, doc):
            return np.array([5.0 if s == "nộp" else -5.0 for sent in doc for w in sent for s in w.syllables])

    fine = ScoredCompressor(Favour(), SelectionConfig(unit="word"), "fine")
    ctx = Context(text=TEXT, doc=WORDS, question="nộp gì?")
    out = UnitSelectCompressor(Fixed([0.0, 5.0, 1.0]), fine).compress(ctx, 3, TokenCounter(tokenizer))
    assert "nộp" in out and "Thuế" not in out


def test_sentence_level_selects_sentences(tokenizer):
    from viword.unitsel import UnitSelectCompressor
    ctx = Context(text=TEXT, doc=WORDS, question="?")  # here one sentence per unit; scores index the sentences
    out = UnitSelectCompressor(Fixed([0.0, 0.0, 5.0]), level="sentence").compress(ctx, 12, TokenCounter(tokenizer))
    assert out.startswith("Điều 3") and "Thuế" not in out
    with pytest.raises(ValueError):
        UnitSelectCompressor(Fixed([]), level="clause")


def test_unitsel_spec_builds_the_compressor():
    from viword.methods import build_compressor
    from viword.unitsel import UnitSelectCompressor
    comp = build_compressor("unitsel:scorer=bm25,fine=none,lam=0.3,name=bm25_units")
    assert isinstance(comp, UnitSelectCompressor) and comp.lam == 0.3 and comp.name == "bm25_units"


def test_targets_groups_and_split():
    from viword.unit_scorer import groups, split_by_cluster, targets
    row = {"id": "q", "cluster": "a", "question": "?", "texts": ["x", "y", "z"], "utility": [1.0, 0.0, 0.0],
           "gold": [False, True, True]}
    assert targets(row, "gold", 0.5) == [0.0, 0.5, 0.5]
    t = targets(row, "utility", 0.5)
    assert t[0] > t[1] == t[2] and abs(sum(t) - 1) < 1e-9
    assert groups([{**row, "gold": [False] * 3}], "gold", 0.5) == []
    rows = [{**row, "id": f"q{i}", "cluster": f"c{i % 5}"} for i in range(20)]
    tr, dev = split_by_cluster(rows, 0.2)
    assert len(tr) + len(dev) == 20 and not {r["cluster"] for r in tr} & {r["cluster"] for r in dev}


def test_listwise_loss_prefers_the_target_order():
    torch = pytest.importorskip("torch")
    from viword.unit_scorer import listwise_loss
    target = torch.tensor([0.8, 0.1, 0.1])
    assert listwise_loss(torch.tensor([3.0, 0.0, 0.0]), target, torch) < listwise_loss(
        torch.tensor([0.0, 3.0, 0.0]), target, torch)


class ChatTokenizer:
    """Whitespace tokens with a chat template ending in a newline, as Qwen's generation prompt does."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        return "".join(f"<{m['role']}> {m['content']} </{m['role']}> " for m in messages) + "<assistant>\n"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [hash(t) % 1000 for t in text.split()]}


def test_answer_ids_and_label_rows():
    from viword.unit_labels import answer_ids, label_rows
    tok = ChatTokenizer()
    ids, start = answer_ids(tok, "hỏi", "hai ba")
    assert len(ids) - start == 2

    class Scorer:  # a unit helps by its length: log p = −(answer tokens) + 0.1 × prompt words
        tokenizer = tok

        def logprobs(self, sequences):
            return [-(len(ids) - start) + 0.1 * start for ids, start in sequences]

    row = {"id": "vbt-1", "cluster": "c", "question": "Ai?", "answers": ["Chính phủ"],
           "context": "A b.\n\nC d e f.", "units": ["d|0|1|1", "d|0|1|2"], "gold_positions": [1],
           "heldout_article": True}
    out = label_rows([row], Scorer())[0]
    assert out["gold"] == [False, True] and out["n_answer_tokens"] == 2 and out["heldout_article"]
    assert out["utility"][1] > out["utility"][0] > 0
    with pytest.raises(ValueError):
        label_rows([{**row, "units": ["only one"]}], Scorer())
