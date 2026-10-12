"""Protected set P for the linguistic protection prior π (DATN §3.4).

A word is protected according to its (text, POS, context), not its string alone. Tiers:
  T1  polarity, condition and facts: negation, conditionals, numbers, named entities
  T2  tense/aspect and modality
  T3  quantity and comparison
The disambiguation rules below are deliberately simple; their precision/recall is
measured on ~200 hand-annotated sentences (DATN §3.4) rather than assumed.
"""
from __future__ import annotations

from .segment import Word

NEGATION = {"không", "chưa", "chẳng", "chả", "đừng", "chớ", "không hề", "chưa từng"}
# These surface sequences contain ``không`` but do not negate a proposition:
# ``không kích`` is the verb "to air-strike" and ``không lâu`` means "soon".
NON_NEGATION_FOLLOWERS = {"kích", "lâu"}
CONDITIONAL = {"nếu", "trừ khi", "giá mà", "dù"}
TENSE_ASPECT = {"đã", "đang", "sẽ", "vừa", "mới", "sắp", "từng"}
MODALITY = {"phải", "nên", "cần", "có thể", "được phép"}
QUANTITY = {"mọi", "mỗi", "tất cả", "chỉ", "hơn", "nhất", "kém"}

TIER_OF_WORD = {
    **{w: "T1" for w in NEGATION | CONDITIONAL},
    **{w: "T2" for w in TENSE_ASPECT | MODALITY},
    **{w: "T3" for w in QUANTITY},
}
TIERS = ("T1", "T2", "T3")
NUMERAL_TAGS = {"M"}


def _next_text(words: list[Word], i: int) -> str:
    return words[i + 1].text.lower() if i + 1 < len(words) else ""


# Ambiguous function words: protect only when the POS/context rule says so.
POS_RULES = {
    "mới": lambda ws, i: ws[i].pos == "R",  # "vừa mới" (adverb), not "nhà mới" (adjective)
    "phải": lambda ws, i: ws[i].pos in {"V", "R"} and _next_text(ws, i) not in {"không", "?"},
    "nên": lambda ws, i: ws[i].pos != "C",  # modal "nên", not the conjunction "(vì vậy) nên"
    "chỉ": lambda ws, i: ws[i].pos == "R",  # "chỉ" = only, not "sợi chỉ" / "chỉ đường"
    "kém": lambda ws, i: _next_text(ws, i) == "hơn",  # "kém hơn", not "học kém"
}


# Sentence-final particles that may sit between a question "không/chưa" and "?":
# "có ... không ạ?", "đã ... chưa nhỉ?", "được không hả?".
QUESTION_PARTICLES = {"ạ", "à", "nhỉ", "hả", "hở", "nhé", "chứ", "vậy", "thế", "nào"}
# "không" read as the digit zero: "không độ", "không giờ", "không phẩy năm".
ZERO_FOLLOWERS = {"độ", "giờ", "phẩy", "chấm"}


def _is_question_final(words: list[Word], i: int) -> bool:
    """True if words[i] is followed only by sentence-final particles and then "?".
    Questions written without "?" are not detected (documented limitation, DATN §3.4)."""
    j = i + 1
    while j < len(words) and words[j].text.lower() in QUESTION_PARTICLES:
        j += 1
    return j < len(words) and words[j].text == "?"


def is_negation(words: list[Word], i: int) -> bool:
    text, nxt = words[i].text.lower(), _next_text(words, i)
    if text not in NEGATION:
        return False
    if text in {"không", "chưa"} and _is_question_final(words, i):  # "có ... không (ạ)?"
        return False
    if text == "không" and (words[i].pos in NUMERAL_TAGS or nxt in ZERO_FOLLOWERS):  # zero
        return False
    if text == "không" and nxt in {"những", "chỉ"}:  # "không những/không chỉ ... mà còn"
        return False
    if text == "không" and nxt in NON_NEGATION_FOLLOWERS:
        return False
    if text == "chẳng" and nxt == "hạn":  # "chẳng hạn" = "for example"
        return False
    return True


def protection_tier(words: list[Word], i: int) -> str | None:
    """Tier of words[i] ("T1", "T2", "T3") or None if not protected."""
    w = words[i]
    text = w.text.lower()
    if w.is_punct:
        return None
    if w.ner != "O":
        return "T1"
    # Numbers, including "không" used as zero before a numeral ("không độ"): T1 either way.
    if w.pos in NUMERAL_TAGS or any(c.isdigit() for c in text):
        return "T1"
    if text in NEGATION:
        return "T1" if is_negation(words, i) else None
    rule = POS_RULES.get(text)
    if rule is not None and not rule(words, i):
        return None
    return TIER_OF_WORD.get(text)


def protection_tiers(words: list[Word]) -> list[str | None]:
    return [protection_tier(words, i) for i in range(len(words))]


def force_tokens_for_context(words: list[Word], tiers: tuple[str, ...] = TIERS) -> list[str]:
    """Return the contextual P list for an LLMLingua ``force_tokens`` call.

    Unlike :func:`naive_syllable_protection`, this preserves multi-syllable
    protected words, NER spans, numbers, and the POS/context disambiguation in
    :func:`protection_tiers`.  Duplicate surface forms are passed once because
    LLMLingua forces all occurrences of a token string.
    """
    wanted = set(tiers)
    result = []
    for word, tier in zip(words, protection_tiers(words)):
        if tier in wanted and word.text not in result:
            result.append(word.text)
    return result


def force_syllable_mask(words: list[Word], tiers: tuple[str, ...] = TIERS) -> list[bool]:
    """Expand the contextual P list to a hard mask over the word's syllables."""
    wanted = set(tiers)
    return [tier in wanted for word, tier in zip(words, protection_tiers(words)) for _ in word.syllables]


def string_force_syllable_mask(words: list[Word], tiers: tuple[str, ...] = TIERS) -> list[bool]:
    """`force_tokens` semantics for the H2 control (DATN §3.4): the P word list of the
    selected tiers is matched as plain lowercase syllable sequences anywhere in the text,
    plus every syllable containing a digit (`force_reserve_digit`, tier T1). There is no
    POS/context disambiguation and no NER, so "không" in "không khí" is forced too —
    exactly the behaviour ViWord-C's word-level, disambiguated π is compared against."""
    wanted = set(tiers)
    phrases = sorted({tuple(w.split(" ")) for w, t in TIER_OF_WORD.items() if t in wanted},
                     key=len, reverse=True)
    sylls = [s.lower() for w in words for s in w.syllables]
    mask = [False] * len(sylls)
    for phrase in phrases:
        n = len(phrase)
        for i in range(len(sylls) - n + 1):
            if tuple(sylls[i:i + n]) == phrase:
                mask[i:i + n] = [True] * n
    if "T1" in wanted:
        mask = [m or any(c.isdigit() for c in s) for m, s in zip(mask, sylls)]
    return mask


def protected_mask(words: list[Word], tiers: tuple[str, ...] = TIERS) -> list[bool]:
    return [t in tiers for t in protection_tiers(words)]


def naive_syllable_protection(syllable: str) -> bool:
    """Legacy syllable-string baseline, retained only for non-H2 ablations.

    It has no context, phrase, or NER information and therefore is not the
    official H2 ``force_tokens`` implementation.
    """
    single = {w for w in TIER_OF_WORD if " " not in w}
    return syllable.lower() in single or any(c.isdigit() for c in syllable)
