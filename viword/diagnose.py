"""Diagnostic measures of DATN §2.2, computed on the week-1 gate (DATN §4.0).

All measures take the original segmented words and the syllable mask of a compression
(obtained with `segment.align_compressed` for systems that only return text).
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from .protect import TIERS, is_negation, protection_tiers, scope_heads
from .segment import Word, normalize, syllables_of

# ---------------------------------------------------------------------------- compressor families

# DATN §1/§4.0 reconcile Lost in Compression with LACC by the source of supervision, so
# Table 1 is reported per family rather than as one list.
FAMILY_ORDER = ("learned, English labels", "learned, Vietnamese labels", "perplexity / LM",
                "deterministic", "other")


def compressor_family(name: str) -> str:
    """Family of a diagnose/evaluate method name (see the run_pipeline.sh specs)."""
    if name.startswith(("llmlingua2vi", "viword", "lexprior")):
        return "learned, Vietnamese labels"
    if name.startswith("llmlingua2") or name.startswith("translate_llmlingua2"):
        return "learned, English labels"
    if name.startswith(("llmlingua", "longllmlingua", "selective_context", "ppl_sent")):
        return "perplexity / LM"
    if name.startswith(("random", "lead", "truncation", "stopword", "lexical", "tfidf_sent",
                        "probe_negation")):
        return "deterministic"
    return "other"


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


def chance_cbr_distribution(keep_rate: float, syllable_lengths) -> float:
    """Expected CBR for independent retention over the observed word lengths.

    CBR is conditioned on a multi-syllable word retaining at least one syllable,
    so numerator and denominator are aggregated before taking their ratio.
    """
    lengths = [int(n) for n in syllable_lengths if int(n) > 1]
    r = keep_rate
    if not lengths or not 0 < r < 1:
        return float("nan")
    partial = sum(1 - r ** n - (1 - r) ** n for n in lengths)
    survived = sum(1 - (1 - r) ** n for n in lengths)
    return partial / survived if survived else float("nan")


def _subword_count(tokenizer, text: str) -> int:
    """Return the number of non-special tokenizer pieces needed for *text*."""
    try:
        pieces = tokenizer.encode(text, add_special_tokens=False)
        return max(1, len(pieces))
    except (AttributeError, TypeError, ValueError):
        pieces = tokenizer.tokenize(text)
        return max(1, len(pieces))


def _top1_predictions(raw, n_masks: int) -> list[dict]:
    """Normalize transformers' single- and multi-mask fill-mask outputs."""
    if n_masks == 0:
        return []
    if isinstance(raw, dict):
        return [raw]
    if not isinstance(raw, list):
        return []
    if n_masks == 1:
        if raw and isinstance(raw[0], dict):
            return [raw[0]]
        if raw and isinstance(raw[0], list) and raw[0] and isinstance(raw[0][0], dict):
            return [raw[0][0]]
        return []
    result = []
    for group in raw:
        if isinstance(group, dict):
            result.append(group)
        elif isinstance(group, list) and group and isinstance(group[0], dict):
            result.append(group[0])
    return result[:n_masks]


def _decode_prediction(tokenizer, token_strs: list[str]) -> str:
    """Decode top-1 token pieces while supporting lightweight test tokenizers."""
    if hasattr(tokenizer, "convert_tokens_to_string"):
        text = tokenizer.convert_tokens_to_string(token_strs)
    else:
        text = "".join(token_strs).replace("▁", " ")
    # A tokenizer may expose a SentencePiece boundary as whitespace for each
    # independently masked position. The target is one Vietnamese syllable,
    # so those synthetic boundaries must not make a correct subword sequence fail.
    return "".join(text.strip().lower().split())


def chance_cbr_counts(words: list[Word], syllable_mask: list[bool]) -> dict[str, float]:
    """Expected partial and surviving multi-syllable words if the same number of syllables
    were kept independently at random (keep rate r of this mask over non-punctuation words).
    Sum both over a corpus and divide, as for `chance_cbr_distribution`."""
    lengths, kept, total, i = [], 0, 0, 0
    for w in words:
        n = len(w.syllables)
        if not w.is_punct:
            kept += sum(syllable_mask[i:i + n])
            total += n
            if n > 1:
                lengths.append(n)
        i += n
    r = kept / total if total else 0.0
    partial = sum(1 - r ** n - (1 - r) ** n for n in lengths)
    survived = sum(1 - (1 - r) ** n for n in lengths)
    return {"partial": partial, "survived": survived, "keep_rate": r}


def scope_counts(words: list[Word], syllable_mask: list[bool]) -> dict[str, int]:
    """Scope retention (item e): of the kept T1/T2 operators that have a head
    (`protect.scope_heads`), how many also keep their head. A word counts as kept if any
    of its syllables is kept."""
    starts = np.cumsum([0] + [len(w.syllables) for w in words])
    kept = [any(syllable_mask[starts[k]:starts[k + 1]]) for k in range(len(words))]
    pairs = scope_heads(words)
    ops = [(op, head) for op, head in pairs if kept[op]]
    return {"ops": len(pairs), "ops_kept": len(ops), "ops_kept_with_head": sum(kept[h] for _, h in ops)}


def recovery_rate(words: list[Word], syllable_mask: list[bool], mlm, max_words: int = 50,
                  window: int = 100) -> float:
    """RR (DATN §2.2): share of broken words whose missing syllables a masked LM restores.
    See `recovery_counts`."""
    hits, total = recovery_counts(words, syllable_mask, mlm, max_words, window)
    return hits / total if total else float("nan")


def recovery_counts(words: list[Word], syllable_mask: list[bool], mlm, max_words: int = 50,
                    window: int = 100) -> tuple[int, int]:
    """(recovered, checked) broken words; checked − recovered are the harmful cuts.

    For each broken word, the missing syllables are replaced by as many mask tokens as
    their tokenizer uses inside the compressed context (±`window` syllables). The MLM's
    top-1 token sequence is decoded and compared with each missing syllable. This avoids
    treating a multi-subword syllable as unrecoverable by construction. `mlm` is a
    transformers fill-mask pipeline, e.g. pipeline("fill-mask", model="xlm-roberta-large").
    """
    sylls = syllables_of(words)
    starts = np.cumsum([0] + [len(w.syllables) for w in words])
    hits, total = 0, 0
    for wi, status in broken_words(words, syllable_mask):
        if status != "partial" or total >= max_words:
            continue
        a, b = starts[wi], starts[wi + 1]
        lo, hi = max(0, a - window), min(len(sylls), b + window)
        tokens, targets, target_piece_counts = [], [], []
        for k in range(lo, hi):
            if a <= k < b and not syllable_mask[k]:
                n_pieces = _subword_count(mlm.tokenizer, sylls[k])
                tokens.extend([mlm.tokenizer.mask_token] * n_pieces)
                targets.append(sylls[k].lower())
                target_piece_counts.append(n_pieces)
            elif syllable_mask[k]:
                tokens.append(sylls[k])
        predictions = mlm(" ".join(tokens), top_k=1)
        predictions = _top1_predictions(predictions, sum(target_piece_counts))
        total += 1
        offset, recovered = 0, True
        for target, n_pieces in zip(targets, target_piece_counts):
            group = predictions[offset:offset + n_pieces]
            offset += n_pieces
            if len(group) != n_pieces:
                recovered = False
                break
            predicted = _decode_prediction(mlm.tokenizer,
                                            [p.get("token_str", "") for p in group])
            if normalize(predicted).lower() != normalize("".join(target.strip().split())).lower():
                recovered = False
                break
        hits += recovered
    return hits, total


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
