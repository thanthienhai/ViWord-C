"""Direction B retrieval: BM25 over units and the gold-unit lookup of scripts/legal_rag.py."""
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("sklearn")
from viword.legal_index import BM25, tokens  # noqa: E402

ROOT = Path(__file__).parents[1]


def _legal_rag():
    spec = importlib.util.spec_from_file_location("legal_rag", ROOT / "scripts/legal_rag.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tokens_are_lower_case_syllables():
    assert tokens("Nghị định 77/2025/NĐ-CP, Điều 8.") == ["nghị", "định", "77", "2025", "nđ", "cp", "điều", "8"]


def test_bm25_ranks_the_matching_unit_first(tmp_path):
    texts = ["Điều 8. Xử lý tài sản do chủ sở hữu tự nguyện chuyển giao quyền sở hữu cho Nhà nước",
             "Điều 9. Thuế giá trị gia tăng đối với hàng hóa nhập khẩu",
             "Điều 10. Đất đai và quyền sử dụng đất của hộ gia đình"]
    bm25 = BM25.fit(texts)
    top, scores = bm25.search("tài sản tự nguyện chuyển giao cho Nhà nước", 3)
    assert top[0] == 0 and scores[0] > scores[1] >= scores[2] >= 0
    assert bm25.scores("từ không có").max() == 0
    bm25.save(str(tmp_path / "bm"))
    again = BM25.load(str(tmp_path / "bm"))
    assert list(again.search("thuế nhập khẩu", 1)[0]) == [1]


CONTEXT = ("Nghị định 77/2025/NĐ-CP Điều 8. Xử lý tài sản.\n1. Tài sản do chủ sở hữu tự nguyện chuyển giao.\n\n"
           "Luật Đất đai số 31/2024/QH15 Điều 81. Thu hồi đất.\nNhà nước thu hồi đất trong các trường hợp sau.")


def test_identifier_spans_cover_label_article_and_clause_number():
    from viword.legal import identifier_spans
    words = [CONTEXT[a:b] for a, b in identifier_spans(CONTEXT)]
    assert words == ["Nghị định 77/2025/NĐ-CP Điều 8", "1.", "Luật Đất đai số 31/2024/QH15 Điều 81"]


def test_keep_ids_puts_identifiers_first_in_both_units():
    import numpy as np
    from conftest import W

    from viword.compressor import Context, ScoredCompressor, SelectionConfig
    from viword.legal import identifier_flags
    words = [W(x) for x in ["Nghị định", "77/2025/NĐ-CP", "Điều", "8", ".", "Xử lý", "tài sản", ".", "1", ".", "Tài sản",
                            "do", "chủ sở hữu", "tự nguyện", "chuyển giao", ".", "Luật", "Đất đai", "số",
                            "31/2024/QH15", "Điều", "81", ".", "Thu hồi", "đất", ".", "Nhà nước", "thu hồi", "đất",
                            "trong", "các", "trường hợp", "sau", "."]]
    flags = identifier_flags(words, CONTEXT)
    assert [w.text for w, f in zip(words, flags) if f] == ["Nghị định", "77/2025/NĐ-CP", "Điều", "8", "1", ".", "Luật",
                                                           "Đất đai", "số", "31/2024/QH15", "Điều", "81"]

    class Flat:
        def syllable_logits(self, doc):
            return np.zeros(sum(len(w.syllables) for s in doc for w in s))

    ctx = Context(text=CONTEXT, doc=[words])
    for unit in ("word", "syllable"):
        _, scores, priority, _ = ScoredCompressor(Flat(), SelectionConfig(unit=unit, keep_ids=True), "x").unit_scores(ctx)
        assert len(priority) == len(scores) and sum(priority) == (12 if unit == "word" else 14)  # 12 words, 14 syllables


def test_gold_units_by_clause_or_article():
    rag = _legal_rag()
    by_article = {("d1", 0, "8"): ["d1|0|8|0", "d1|0|8|1", "d1|0|8|2"], ("d2", 0, "3"): ["d2|0|3|"]}
    q = {"gold": [{"doc_id": "d1", "attachment": 0, "article": "8", "clause": 1},
                  {"doc_id": "d2", "attachment": 0, "article": "3", "clause": None},
                  {"doc_id": "d3", "attachment": 0, "article": "1", "clause": None}]}
    assert rag.gold_units(q, by_article) == [{"d1|0|8|1"}, {"d2|0|3|"}]
    assert rag.uid("d1", 0, "8", None) == "d1|0|8|"
    assert rag.in_order({"d1|0|8|10", "d1|0|8|2", "d1|0|8|0"}) == ["d1|0|8|0", "d1|0|8|2", "d1|0|8|10"]
