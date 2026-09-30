import pytest

from viword.distill import LeakIndex, make_labels


def test_soft_word_labels_and_teacher_cbr(sentence):
    labels = make_labels([sentence], "Học không đi học Hà Nội.")
    # "Học sinh" keeps 1 of 2 syllables -> soft label 0.5 (not rounded up to 1)
    assert labels["word_labels"][0] == 0.5
    assert labels["word_labels"][5] == 1.0  # Hà Nội
    assert labels["syllable_labels"][:2] == [1, 0]
    assert labels["teacher_cbr"] == 0.5
    assert labels["unmatched_ratio"] == 0.0


def test_leak_index():
    text = "một hai ba bốn năm sáu bảy tám chín mười mười một mười hai mười ba mười bốn"
    index = LeakIndex([text], n=5)
    assert index.leaks("xin chào, ba bốn năm sáu bảy là các số")
    assert not index.leaks("hoàn toàn khác nhau về nội dung và câu chữ ở đây")


def test_empty_leak_index_is_an_error():
    with pytest.raises(ValueError):
        LeakIndex(["quá ngắn"], n=13)
