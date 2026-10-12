from conftest import W

from viword.baselines import NegationProbe
from viword.compressor import Context
from viword.diagnose import has_negation
from viword.protect import (force_syllable_mask, force_tokens_for_context, is_negation,
                            naive_syllable_protection, protection_tiers)


def tiers(words):
    return dict(zip([w.text for w in words], protection_tiers(words)))


def test_negation_entity_and_number_are_T1():
    words = [W("Hà Nội", "Np", "B-LOC"), W("không", "R"), W("có", "V"), W("3", "M"), W("trường", "N")]
    t = tiers(words)
    assert t["không"] == "T1" and t["Hà Nội"] == "T1" and t["3"] == "T1" and t["trường"] is None


def test_question_particle_is_not_negation():
    words = [W("Anh", "P"), W("có", "V"), W("đi", "V"), W("không", "T"), W("?", "CH")]
    assert tiers(words)["không"] is None
    words = [W("Đã", "R"), W("ăn", "V"), W("chưa", "T"), W("?", "CH")]
    assert tiers(words)["chưa"] is None


def test_question_particle_rule_has_positive_and_negative_cases():
    assert is_negation([W("không", "R"), W("đi", "V")], 0)
    assert not is_negation([W("đi", "V"), W("không", "T"), W("?", "CH")], 1)
    assert is_negation([W("chưa", "R"), W("xong", "A")], 0)
    assert not is_negation([W("đã", "R"), W("xong", "V"), W("chưa", "T"), W("?", "CH")], 2)


def test_question_with_final_particles_is_not_negation():
    assert not is_negation([W("có", "V"), W("đi", "V"), W("không", "T"), W("ạ", "T"), W("?", "CH")], 2)
    assert not is_negation([W("đã", "R"), W("xong", "V"), W("chưa", "T"), W("nhỉ", "T"), W("?", "CH")], 2)
    # a particle without a question mark is not enough: "không ạ." is a polite "no"
    assert is_negation([W("không", "R"), W("ạ", "T"), W(".", "CH")], 0)


def test_khong_read_as_zero_is_not_negation():
    zero = [W("không", "M"), W("độ", "N")]
    assert not is_negation(zero, 0) and not has_negation(zero)
    assert tiers(zero)["không"] == "T1"  # still protected, as a number
    assert not is_negation([W("không", "R"), W("phẩy", "N"), W("năm", "M")], 0)
    assert NegationProbe().compress(Context(text="không độ", doc=[zero]), 0, None) == "không độ"


def test_khong_before_number_is_negation_for_has_negation_and_probe():
    words = [W("không", "R"), W("3", "M")]
    assert is_negation(words, 0)
    assert has_negation(words)
    assert NegationProbe().compress(Context(text="không 3", doc=[words]), 0, None) == "3"


def test_compound_and_example_exceptions_are_not_negation():
    assert not is_negation([W("không khí", "N"), W("sạch", "A")], 0)
    assert not is_negation([W("không", "R"), W("những", "L"), W("người", "N")], 0)
    assert not is_negation([W("không", "R"), W("chỉ", "R"), W("mà", "C"), W("còn", "R")], 0)
    assert not is_negation([W("chẳng", "R"), W("hạn", "N")], 0)
    assert is_negation([W("chẳng", "R"), W("đi", "V")], 0)


def test_non_negating_khong_phrases_are_not_probed():
    airstrike = [W("không", "R"), W("kích", "V")]
    soon = [W("không", "R"), W("lâu", "A")]
    assert not is_negation(airstrike, 0)  # air-strike
    assert not is_negation(soon, 0)  # soon
    assert NegationProbe().compress(Context(text="không kích", doc=[airstrike]), 0, None) == "không kích"
    assert NegationProbe().compress(Context(text="không lâu", doc=[soon]), 0, None) == "không lâu"
    assert is_negation([W("không", "R"), W("được", "V")], 0)


def test_additive_construction_is_not_negation():
    words = [W("không", "R"), W("những", "L"), W("đẹp", "A")]
    assert tiers(words)["không"] is None


def test_pos_rules_for_ambiguous_words():
    assert tiers([W("nhà", "N"), W("mới", "A")])["mới"] is None  # new
    assert tiers([W("vừa", "R"), W("mới", "R"), W("đến", "V")])["mới"] == "T2"  # just
    assert tiers([W("sợi", "Nc"), W("chỉ", "N")])["chỉ"] is None  # thread
    assert tiers([W("chỉ", "R"), W("còn", "V")])["chỉ"] == "T3"  # only
    assert tiers([W("học", "V"), W("kém", "A")])["kém"] is None
    assert tiers([W("kém", "A"), W("hơn", "A")])["kém"] == "T3"


def test_phai_and_nen_have_positive_and_negative_pos_contexts():
    assert tiers([W("phải", "V"), W("đi", "V")])["phải"] == "T2"
    assert tiers([W("bên", "N"), W("phải", "N")])["phải"] is None
    assert tiers([W("phải", "V"), W("không", "T"), W("?", "CH")])["phải"] is None
    assert tiers([W("nên", "V"), W("đi", "V")])["nên"] == "T2"
    assert tiers([W("vì vậy", "C"), W("nên", "C")])["nên"] is None


def test_multiword_modality_depends_on_segmentation_and_duoc_is_unimplemented():
    assert tiers([W("có thể", "V")])["có thể"] == "T2"
    split = tiers([W("có", "V"), W("thể", "N")])
    assert split["có"] is None and split["thể"] is None
    assert tiers([W("được phép", "V")])["được phép"] == "T2"
    assert tiers([W("được", "V")])["được"] is None


def test_word_level_does_not_confuse_compounds():
    # "không khí" (air) is one word, so it is not a negation
    assert tiers([W("không khí", "N"), W("sạch", "A")])["không khí"] is None
    # the syllable-level force_tokens baseline protects its first syllable by mistake
    assert naive_syllable_protection("không")


def test_contextual_force_tokens_include_phrases_ner_and_numbers():
    words = [
        W("không hề", "R"), W("chưa từng", "R"), W("trừ khi", "C"), W("giá mà", "C"),
        W("có thể", "V"), W("được phép", "V"), W("tất cả", "L"), W("Hà Nội", "Np", "B-LOC"),
        W("42", "M"), W("không khí", "N"),
    ]
    assert force_tokens_for_context(words) == [
        "không hề", "chưa từng", "trừ khi", "giá mà", "có thể", "được phép", "tất cả", "Hà Nội", "42",
    ]
    assert force_syllable_mask(words) == [True] * 17 + [False, False]


def test_package_force_tokens_cannot_retain_occurrence_level_disambiguation():
    words = [W("mới", "A"), W("mới", "R")]
    assert force_tokens_for_context(words) == ["mới"]
    assert force_syllable_mask(words) == [False, True]
