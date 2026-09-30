from conftest import W

from viword.protect import naive_syllable_protection, protection_tiers


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


def test_word_level_does_not_confuse_compounds():
    # "không khí" (air) is one word, so it is not a negation
    assert tiers([W("không khí", "N"), W("sạch", "A")])["không khí"] is None
    # the syllable-level force_tokens baseline protects its first syllable by mistake
    assert naive_syllable_protection("không")
