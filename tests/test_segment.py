from conftest import W

from viword.segment import (align_compressed, doc_from_json, doc_to_json, expand_word_mask, normalize,
                            render, render_all, syllables_of)


def test_render_keeps_punctuation_attached(sentence):
    assert render_all(sentence) == "Học sinh không đi học ở Hà Nội."


def test_render_partial_word(sentence):
    # keep only "Học" of "Học sinh" and the final period
    mask = [True, False] + [False] * 6 + [True]
    assert render(sentence, mask) == "Học."


def test_expand_word_mask(sentence):
    mask = expand_word_mask(sentence, [True, False, False, False, False, True, False])
    assert mask == [True, True, False, False, False, False, True, True, False]


def test_align_exact_deletion(sentence):
    mask, unmatched = align_compressed(sentence, "Học sinh không đi Hà Nội")
    assert unmatched == 0.0
    assert [s for s, k in zip(syllables_of(sentence), mask) if k] == ["Học", "sinh", "không", "đi", "Hà", "Nội"]


def test_align_detects_rewriting(sentence):
    _, unmatched = align_compressed(sentence, "Các em nghỉ học")
    assert unmatched > 0.5


def test_align_is_case_and_spacing_insensitive():
    words = [W("COVID-19"), W(","), W("hôm nay")]
    mask, unmatched = align_compressed(words, "covid - 19, hôm nay")
    assert all(mask) and unmatched == 0.0


def test_normalize_tone_placement():
    assert normalize("Cộng hoà, sức khoẻ, Uỷ ban, thuỷ") == "Cộng hòa, sức khỏe, Ủy ban, thủy"
    # closed syllables, "qu" + y and other vowels are left alone
    assert normalize("hoàn toàn, quý, quỳ, hoè, chào") == "hoàn toàn, quý, quỳ, hòe, chào"


def test_align_matches_across_spelling_and_number_splits():
    # VnCoreNLP writes "hóa" and splits "6h30"; the teacher copies the source ("hoá", "6h30")
    words = [W("văn hóa"), W("lúc"), W("6"), W("h30")]
    mask, unmatched = align_compressed(words, "văn hoá lúc 6h30")
    assert all(mask) and unmatched == 0.0


def test_json_roundtrip(sentence):
    doc = [sentence, [W("Tốt", "A"), W(".", "CH")]]
    assert doc_from_json(doc_to_json(doc)) == doc


def test_inserted_punctuation_is_not_rewriting(sentence):
    # the teacher often joins fragments with commas; only inserted words count as rewriting
    _, unmatched = align_compressed(sentence, "Học sinh, không đi, Hà Nội.")
    assert unmatched == 0.0
    _, unmatched = align_compressed(sentence, "Học sinh không đi trường Hà Nội")
    assert unmatched == 1 / 7
