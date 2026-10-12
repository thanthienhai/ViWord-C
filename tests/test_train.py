from viword.segment import Word, doc_to_json
from viword.train import syllable_targets


def test_word_targets_are_copied_to_each_syllable():
    doc = [[Word("học sinh"), Word("đi")]]
    record = {"doc": doc_to_json(doc), "word_labels": [0.5, 1.0]}
    assert syllable_targets(record, "word") == [0.5, 0.5, 1.0]


def test_syllable_targets_keep_hard_labels():
    record = {"doc": [], "syllable_labels": [1, 0, 1]}
    assert syllable_targets(record, "syllable") == [1.0, 0.0, 1.0]
