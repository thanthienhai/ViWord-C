"""Protected set P for the linguistic protection prior π (DATN §3.4).

A word is protected according to its (text, POS, context), not its string alone. Tiers:
  T1  polarity, condition and facts: negation, conditionals, numbers, named entities
  T2  tense/aspect and modality
  T3  quantity and comparison
The disambiguation rules below are deliberately simple; their precision/recall is
measured on ~200 hand-annotated sentences (DATN §3.4) rather than assumed.

Detector v2 (default) fixes the two largest error groups of the blind-v1 audit: 141/142
missed T1 words were proper names the NER left untagged, so a proper-noun POS (Np) or a
capitalised word that is not sentence-initial (any position for a multi-syllable word
whose every syllable is capitalised) counts as a name; and 28/37 false positives were
"một" used as an article, so "một" before a classifier ("một con", "một cái") is not a
number. v2 must be audited on a new blind sheet (blind-v2), never on blind-v1, which was
used to find these errors. VIWORD_P_DETECTOR=v1 restores the audited v1 rules.
"""
from __future__ import annotations

import os

from .segment import Word

DETECTOR_VERSION = os.environ.get("VIWORD_P_DETECTOR", "v2")
if DETECTOR_VERSION not in {"v1", "v2"}:
    raise ValueError(f"VIWORD_P_DETECTOR must be v1 or v2, got {DETECTOR_VERSION!r}")

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


# "một" + classifier is an article ("một con mèo" = "a cat"), not the number one (v2).
CLASSIFIERS = {"con", "cái", "chiếc", "ngôi", "quyển", "cuốn", "bức", "tấm", "sự", "cuộc", "nỗi",
               "niềm", "bài", "căn", "tờ", "trái", "quả", "cây", "hạt", "bộ", "đôi", "mảnh", "cơn"}
CLASSIFIER_TAGS = {"Nc"}
# a word after one of these starts a sentence or a quotation, so its capital letter says nothing
SENTENCE_OPENERS = {".", "!", "?", "…", ":", ";", "\"", "“", "'", "‘", "(", "-", "–", "—"}


def _is_article_mot(words: list[Word], i: int) -> bool:
    if words[i].text.lower() != "một" or i + 1 >= len(words):
        return False
    nxt = words[i + 1]
    return nxt.pos in CLASSIFIER_TAGS or nxt.text.lower() in CLASSIFIERS


def _capitalised(syllable: str) -> bool:
    return syllable[:1].isupper()


def is_fallback_name(words: list[Word], i: int) -> bool:
    """Proper name the NER missed (v2): POS Np, or capitalised syllables away from a sentence
    start. A sentence-initial word counts only if it has several syllables, all capitalised
    ("Hà Nội"), since a common word there capitalises only its first syllable."""
    w = words[i]
    if w.pos == "Np":
        return True
    sylls = w.syllables
    if not any(c.isalpha() for c in w.text) or not all(_capitalised(s) for s in sylls):
        return False
    initial = i == 0 or words[i - 1].text in SENTENCE_OPENERS
    return len(sylls) > 1 or not initial


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
    v2 = DETECTOR_VERSION == "v2"
    if v2 and text not in TIER_OF_WORD and is_fallback_name(words, i):
        return "T1"
    if v2 and _is_article_mot(words, i):
        return None
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


# π_NEG (candidate amendment of 2026-10-25): protect the lexical polarity and condition operators alone, as the
# pseudo-tier "NEG" (tiers=NEG). In the distilled labels, names and numbers are ~94% of T1 occurrences and are kept
# more often than average already, while the teacher keeps negations barely above the average word (0.55 vs 0.39 at
# rate 0.5; 0.25 vs 0.19 at 0.25) and conditionals below it, so protecting T1 mostly spends budget on names.
NEG = "NEG"


def is_operator(words: list[Word], i: int) -> bool:
    """A real negation (`is_negation`, disambiguated) or a conditional/concessive word."""
    text = words[i].text.lower()
    if text in NEGATION:
        return is_negation(words, i)
    return text in CONDITIONAL


def protected_flags(words: list[Word], tiers: tuple[str, ...]) -> list[bool]:
    """Word i is protected if its tier is in `tiers`, or if "NEG" is in `tiers` and it is an operator."""
    wanted = set(tiers)
    neg = NEG in wanted
    return [t in wanted or (neg and is_operator(words, i)) for i, t in enumerate(protection_tiers(words))]


SCOPE_WINDOW = 4  # the head is looked for among the next few words of the clause


def scope_heads(words: list[Word], tiers: list[str | None] | None = None) -> list[tuple[int, int]]:
    """(operator, head) pairs: for every lexical T1/T2 operator (negation, condition,
    tense/aspect, modality; not names or numbers), the first verb or adjective (POS V*/A*)
    among the next SCOPE_WINDOW words, skipping adverbs (R: "hề", "đã") and stopping at
    punctuation. A modal verb is a head too, so "không được phép vào" gives (không, được phép)
    and (được phép, vào): keeping "không" while dropping "được phép" breaks the scope.
    Positional rule on VnCoreNLP POS; a dependency parse would be finer."""
    tiers = protection_tiers(words) if tiers is None else tiers
    pairs = []
    for i, (w, tier) in enumerate(zip(words, tiers)):
        if tier not in {"T1", "T2"} or w.text.lower() not in TIER_OF_WORD:
            continue
        for j in range(i + 1, min(len(words), i + 1 + SCOPE_WINDOW)):
            nxt = words[j]
            if nxt.is_punct:
                break
            if nxt.pos[:1] == "R":
                continue
            if nxt.pos[:1] in {"V", "A"}:
                pairs.append((i, j))
                break
    return pairs


def force_tokens_for_context(words: list[Word], tiers: tuple[str, ...] = TIERS) -> list[str]:
    """Return the contextual P list for an LLMLingua ``force_tokens`` call.

    Unlike :func:`naive_syllable_protection`, this preserves multi-syllable
    protected words, NER spans, numbers, and the POS/context disambiguation in
    :func:`protection_tiers`.  Duplicate surface forms are passed once because
    LLMLingua forces all occurrences of a token string.
    """
    result = []
    for word, flag in zip(words, protected_flags(words, tiers)):
        if flag and word.text not in result:
            result.append(word.text)
    return result


def force_syllable_mask(words: list[Word], tiers: tuple[str, ...] = TIERS) -> list[bool]:
    """Expand the contextual P list to a hard mask over the word's syllables."""
    return [flag for word, flag in zip(words, protected_flags(words, tiers)) for _ in word.syllables]


def string_force_syllable_mask(words: list[Word], tiers: tuple[str, ...] = TIERS) -> list[bool]:
    """`force_tokens` semantics for the H2 control (DATN §3.4): the P word list of the
    selected tiers is matched as plain lowercase syllable sequences anywhere in the text,
    plus every syllable containing a digit (`force_reserve_digit`, tier T1). There is no
    POS/context disambiguation and no NER, so "không" in "không khí" is forced too —
    exactly the behaviour ViWord-C's word-level, disambiguated π is compared against."""
    wanted = set(tiers)
    strings = {w for w, t in TIER_OF_WORD.items() if t in wanted}
    if NEG in wanted:  # the operator strings, matched without disambiguation (force_tokens semantics)
        strings |= NEGATION | CONDITIONAL
    phrases = sorted({tuple(w.split(" ")) for w in strings}, key=len, reverse=True)
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
    return protected_flags(words, tiers)


def naive_syllable_protection(syllable: str) -> bool:
    """Legacy syllable-string baseline, retained only for non-H2 ablations.

    It has no context, phrase, or NER information and therefore is not the
    official H2 ``force_tokens`` implementation.
    """
    single = {w for w in TIER_OF_WORD if " " not in w}
    return syllable.lower() in single or any(c.isdigit() for c in syllable)
