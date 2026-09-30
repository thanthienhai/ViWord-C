"""Diagnostic measures of DATN §2.2, computed on the week-1 gate (DATN §4.0).

All measures take the original segmented words and the syllable mask of a compression
(obtained with `segment.align_compressed` for systems that only return text).
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from .protect import TIERS, is_negation, protection_tiers
from .segment import Word, syllables_of

# ---------------------------------------------------------------------------- P1: broken words

VOWELS = set("aăâeêioôơuưyáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")


def _onset(syllable: str) -> str:
    s = syllable.lower()
    i = 0
    while i < len(s) and s[i] not in VOWELS:
        i += 1
    return s[:i]


def word_type(w: Word) -> str:
    """Coarse type of a multi-syllable word, used to split CBR (DATN §2.2).

    "name": named entity or proper noun. "reduplicative": heuristic, two syllables with
    the same onset (sạch sẽ, lung linh); coordinate compounds (quần áo) need a lexicon and
    fall into "other".
    """
    if w.ner != "O" or w.pos == "Np":
        return "name"
    sylls = w.syllables
    if len(sylls) == 2 and _onset(sylls[0]) and _onset(sylls[0]) == _onset(sylls[1]):
        return "reduplicative"
    return "other"


def broken_words(words: list[Word], syllable_mask: list[bool]) -> list[tuple[int, str]]:
    """(word index, status) for every multi-syllable, non-punctuation word that keeps at
    least one syllable. status is "partial" or "full"."""
    out, i = [], 0
    for wi, w in enumerate(words):
        n = len(w.syllables)
        kept = sum(syllable_mask[i:i + n])
        i += n
        if n > 1 and not w.is_punct and kept > 0:
            out.append((wi, "partial" if kept < n else "full"))
    return out


def cbr_counts(words: list[Word], syllable_mask: list[bool]) -> Counter:
    """Counts of (word type, "partial"/"full"); sum these over a corpus, then call cbr()."""
    counts = Counter()
    for wi, status in broken_words(words, syllable_mask):
        for key in ("all", word_type(words[wi])):
            counts[key, status] += 1
    return counts


def cbr(counts: Counter) -> dict[str, float]:
    """Compound Break Rate: partial / (partial + full), overall and by word type."""
    result = {}
    for key in ("all", "name", "reduplicative", "other"):
        n = counts[key, "partial"] + counts[key, "full"]
        result[key] = counts[key, "partial"] / n if n else float("nan")
        result[f"n_{key}"] = n
    return result


def chance_cbr(keep_rate: float, n_syllables: int = 2) -> float:
    """CBR of a compressor that drops syllables independently with keep rate r:
    P(partial | at least one kept) = (1 − r^n − (1−r)^n) / (1 − (1−r)^n).
    For n = 2 this is 2(1−r)/(2−r)."""
    r, n = keep_rate, n_syllables
    return (1 - r ** n - (1 - r) ** n) / (1 - (1 - r) ** n)


def recovery_rate(words: list[Word], syllable_mask: list[bool], mlm, max_words: int = 50,
                  window: int = 100) -> float:
    """RR (DATN §2.2): share of broken words whose missing syllables a masked LM restores.

    For each broken word, the missing syllables are replaced by mask tokens inside the
    compressed context (±`window` syllables), and the MLM's top-1 prediction is compared
    with the missing syllable. `mlm` is a transformers fill-mask pipeline, e.g.
    pipeline("fill-mask", model="xlm-roberta-large").
    """
    sylls = syllables_of(words)
    starts = np.cumsum([0] + [len(w.syllables) for w in words])
    hits, total = 0, 0
    for wi, status in broken_words(words, syllable_mask):
        if status != "partial" or total >= max_words:
            continue
        a, b = starts[wi], starts[wi + 1]
        lo, hi = max(0, a - window), min(len(sylls), b + window)
        tokens, targets = [], []
        for k in range(lo, hi):
            if a <= k < b and not syllable_mask[k]:
                tokens.append(mlm.tokenizer.mask_token)
                targets.append(sylls[k].lower())
            elif syllable_mask[k]:
                tokens.append(sylls[k])
        predictions = mlm(" ".join(tokens), top_k=1)
        if len(targets) == 1:  # the pipeline returns a flat list for a single mask
            predictions = [predictions]
        total += 1
        hits += all(p[0]["token_str"].strip().lower() == t for p, t in zip(predictions, targets))
    return hits / total if total else float("nan")


# ---------------------------------------------------------------------------- P2: function words

def retention_counts(words: list[Word], syllable_mask: list[bool]) -> Counter:
    """(group, "kept"/"total") counts for all non-punctuation words and for each
    protection tier. A word counts as kept if any of its syllables is kept."""
    counts, i = Counter(), 0
    for w, tier in zip(words, protection_tiers(words)):
        n = len(w.syllables)
        kept = any(syllable_mask[i:i + n])
        i += n
        for group in (["all"] if not w.is_punct else []) + ([tier] if tier else []):
            counts[group, "total"] += 1
            counts[group, "kept"] += kept
    return counts


def retention(counts: Counter) -> dict[str, float]:
    """Keep rate per tier, to compare with the keep rate of all words ("all")."""
    result = {}
    for group in ("all",) + TIERS:
        n = counts[group, "total"]
        result[group] = counts[group, "kept"] / n if n else float("nan")
        result[f"n_{group}"] = n
    return result


def has_negation(words: list[Word]) -> bool:
    return any(is_negation(words, i) for i in range(len(words)))


# ---------------------------------------------------------------------------- P3: token cost

def _ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x))
    for value in np.unique(x):  # average ranks for ties
        ranks[x == value] = ranks[x == value].mean()
    return ranks


def spearman(x, y) -> float:
    return float(np.corrcoef(_ranks(np.asarray(x, float)), _ranks(np.asarray(y, float)))[0, 1])


def token_cost_profile(words: list[Word], counter) -> dict[str, float]:
    """Fertility (tokens per syllable), CV_T of c_T(w), and Spearman(c_T(w), n_syl(w))
    over non-punctuation words. `counter` is a select.TokenCounter of the target LLM."""
    words = [w for w in words if not w.is_punct]
    costs = np.array([counter.unit_cost(w.text, False) for w in words], dtype=float)
    n_syl = np.array([len(w.syllables) for w in words], dtype=float)
    return {
        "fertility": float(costs.sum() / n_syl.sum()),
        "cv": float(costs.std() / costs.mean()),
        "spearman_cost_nsyl": spearman(costs, n_syl),
        "n_words": len(words),
    }
