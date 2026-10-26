"""Direction B on ViBidLQA: the task builder, the legal extractive-QA task, the unit-level BM25 baseline."""
import importlib.util
from pathlib import Path

import pytest
from conftest import W

ROOT = Path(__file__).parents[1]


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LAW = "Luật Đấu thầu số 22/2023/QH15"
UNITS = {("166581", "66"): [
    {"uid": "166581|0|66|1", "text": f"{LAW} Điều 66. Điều kiện ký kết hợp đồng\n1. Tại thời điểm ký kết, hồ sơ dự "
                                     "thầu của nhà thầu được lựa chọn còn hiệu lực."},
    {"uid": "166581|0|66|2", "text": f"{LAW} Điều 66. Điều kiện ký kết hợp đồng\n2. Bảo đảm thực hiện hợp đồng "
                                     "theo quy định."}],
    ("166583", "66"): [{"uid": "166583|0|66|", "text": "Nghị định 24/2024/NĐ-CP Điều 66. Thẩm định giá gói thầu"}]}


def test_passage_maps_to_the_article_that_covers_it_and_answer_to_its_unit():
    tasks = _script("make_vibid_tasks")
    passage = ("điều 66. điều kiện ký kết hợp đồng\n1. Tại thời điểm ký kết, hồ sơ dự thầu của nhà thầu được lựa chọn "
               "còn hiệu lực.")
    assert tasks.article_of(passage, UNITS) == ("166581", "66")
    assert tasks.article_of("Điều 66. Một đoạn không có trong luật lẫn nghị định", UNITS) is None
    assert tasks.article_of("Không nêu Điều nào", UNITS) is None
    assert tasks.answer_units("hồ sơ dự thầu ... còn hiệu lực", UNITS["166581", "66"]) == []
    assert tasks.answer_units("còn hiệu lực.", UNITS["166581", "66"]) == ["166581|0|66|1"]
    assert tasks.norm("Điều  66.\nĐiều-kiện") == "điều 66 điều kiện"


def test_gold_units_given_by_the_task():
    rag = _script("legal_rag")
    assert rag.gold_units({"gold_units": [["166581|0|66|1"], []]}, {}) == [{"166581|0|66|1"}]
    assert set(rag.DATASETS) == {"vlegal", "vibid"} and rag.DATASETS["vibid"]["docs"] == ("166581", "166583")


def test_lqa_reads_every_line_and_scores_f1():
    from viword.data import examples_from_rows
    from viword.eval import MAIN_METRIC, MAX_NEW_TOKENS, build_prompt, parse_prediction, score
    ex = examples_from_rows("vibid", [{"id": "vbq-1", "cluster": "166581|66", "context": "…",
                                       "question": "Khi nào?", "answers": ["a) Một; b) Hai"]}])[0]
    assert ex.task == "lqa" and MAIN_METRIC["vibid"] == "f1" and MAX_NEW_TOKENS["lqa"] == 64
    assert "Điều khoản: …" in build_prompt(ex, "…") and "Câu hỏi: Khi nào?" in build_prompt(ex, "…")
    prediction = parse_prediction("lqa", "a) Một;\n\n b) Hai\n")
    assert prediction == "a) Một; b) Hai" and score(ex, prediction) == {"f1": 1.0, "em": 1.0}
    assert parse_prediction("qa", "a) Một;\nb) Hai") == "a) Một;"  # the registered qa task is unchanged


def test_vibid_rag_rows_are_refused_outside_the_eval_stage(tmp_path):
    from viword.heldout import ALL_TEST_FILES, refuse_unseen_test_rows
    assert ALL_TEST_FILES["vibid_test"] == "vibid"
    row = {"id": "vbq-3", "cluster": "c", "context": "x", "question": "?", "answers": ["x"],
           "derived_from": "vibid_test"}
    with pytest.raises(SystemExit):
        refuse_unseen_test_rows([row], tasks_dir=tmp_path, exclusions_path=tmp_path / "none.json", env={})
    refuse_unseen_test_rows([{**row, "derived_from": "vibid_dev"}], tasks_dir=tmp_path,
                            exclusions_path=tmp_path / "none.json", env={})


TEXT = "Điều 1. Phạm vi\nThuế nhập khẩu hàng hóa .\n\nĐiều 2. Bảo đảm dự thầu\nNhà thầu nộp bảo đảm dự thầu ."
WORDS = [W(x) for x in ["Điều", "1", ".", "Phạm vi", "Thuế", "nhập khẩu", "hàng hóa", ".",
                        "Điều", "2", ".", "Bảo đảm", "dự thầu", "Nhà thầu", "nộp", "bảo đảm", "dự thầu", "."]]


def test_word_blocks_follow_the_blank_lines():
    from viword.legal import word_blocks
    assert word_blocks(WORDS, TEXT) == [0] * 8 + [1] * 10


def test_qunit_keeps_the_unit_that_matches_the_question(tokenizer):
    from viword.baselines import QuestionUnits
    from viword.compressor import Context
    from viword.select import TokenCounter
    counter = TokenCounter(tokenizer)
    ctx = Context(text=TEXT, doc=[WORDS[:8], WORDS[8:]], question="Nhà thầu nộp bảo đảm dự thầu khi nào?")
    out = QuestionUnits().compress(ctx, 10, counter)
    assert out.startswith("Điều 2") and "Thuế" not in out and counter.count(out) <= 10
    out = QuestionUnits().compress(ctx, 4, counter)  # the best unit does not fit: its head
    assert out.startswith("Điều 2") and counter.count(out) <= 4
