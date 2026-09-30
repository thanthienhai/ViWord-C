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


def is_negation(words: list[Word], i: int) -> bool:
    text, nxt = words[i].text.lower(), _next_text(words, i)
    if text not in NEGATION:
        return False
    if text in {"không", "chưa"} and nxt == "?":  # "có ... không?", "đã ... chưa?"
        return False
    if text == "không" and nxt in {"những", "chỉ"}:  # "không những/không chỉ ... mà còn"
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


def protected_mask(words: list[Word], tiers: tuple[str, ...] = TIERS) -> list[bool]:
    return [t in tiers for t in protection_tiers(words)]


def naive_syllable_protection(syllable: str) -> bool:
    """Syllable-level `force_tokens` baseline: plain string match, no disambiguation,
    plus digits (`force_reserve_digit`). Protects e.g. "không" in "không khí" by mistake."""
    single = {w for w in TIER_OF_WORD if " " not in w}
    return syllable.lower() in single or any(c.isdigit() for c in syllable)
