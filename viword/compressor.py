"""Compressors share one interface:

    compress(context: Context, budget: int, counter: TokenCounter) -> str

`budget` is counted in tokens of the target LLM (DATN §4.2). `ScoredCompressor` turns
per-syllable keep-logits (from any scorer) into a compressed text through the shared
`render()` path; it covers ViWord-C, the LLMLingua-2-vi control, all cells of the 2×2
design (DATN §2.3), and the lexical-prior control, which differ only in their scorer and
`SelectionConfig`.
The `protect=force` arm is the H2 control of DATN §3.4: `force_tokens` semantics, i.e. the
P word list matched as plain syllable strings plus digits, as a hard priority at syllable
level, without disambiguation or NER. `protect=force_ctx` is an extra ablation that uses
the disambiguated, NER-aware P at syllable level instead.

Two optional score terms are off by default (so every registered arm is unchanged):
`scope` adds a bonus to the head word an operator acts on (the next verb/adjective,
`protect.scope_heads`) when the operator is in the protected tiers, so that "không" is
not kept while "được phép" is dropped; `qrel` adds a question–word relevance bonus
(share of the word's content syllables that occur in the question), a descriptive,
query-aware variant toward RAG. Neither is part of H1 or H2.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import pool_to_words
from .protect import (force_syllable_mask, naive_syllable_protection, protected_flags,
                      scope_heads, string_force_syllable_mask)
from .segment import Document, Word, expand_word_mask, flatten, render, syllables_of
from .select import TokenCounter, exact_select, greedy_select, trim_to_budget


@dataclass
class Context:
    text: str  # raw (NFC) context
    doc: Document | None = None  # segmented context; None for text-only methods
    question: str | None = None  # only query-aware reference methods may use it
    text_en: str = ""  # parallel English context (Belebele), for translate-then-compress
    example_id: str = ""  # used by methods that read precomputed compressions
    ratio: float = 1.0


@dataclass
class SelectionConfig:
    unit: str = "word"  # decision unit: "word" or "syllable"
    protect: str = "none"  # "none", "soft", "hard", "force" (string P), "force_ctx" (contextual P)
    delta: float = 0.2  # soft protection bonus
    tiers: tuple[str, ...] = ("T1", "T2", "T3")
    alpha: float = 0.0  # cost exponent (H3)
    cost: str = "tokens"  # "tokens" = c_T(w); "syllables" = n_syl(w) (H3 control)
    exact: bool = False  # exact DP knapsack instead of greedy (cost-aware ranking, item d)
    scope: float = 0.0  # bonus to the head word of a protected operator (0 = off)
    qrel: float = 0.0  # question-relevance bonus (0 = off; query-aware, descriptive)


# The syllable mask of the latest select_units() call and the text it rendered. Every mask-based method
# (ScoredCompressor and the rule baselines) goes through select_units, so evaluate.py can take the exact mask
# for its diagnostics instead of re-aligning the text, which misplaces repeated syllables (it gave word-level
# systems a CBR of 0.5–4% in the 2026-10-25 Kaggle trial). It is used only when the text matches.
LAST_SELECTION: dict = {}


def select_units(words: list[Word], scores: list[float], budget: int, counter: TokenCounter,
                 unit: str = "word", priority: list[int] | None = None, alpha: float = 0.0,
                 cost: str = "tokens", exact: bool = False, fill: bool = True,
                 protected: list[bool] | None = None, stats: dict | None = None) -> str:
    """Pick units under the budget, then fix the real token length. Returns text.

    If `stats` is a dict it receives the selection bookkeeping reported by evaluate.py:
    Σ c of the selected units, the real token length before the trim step and its
    deviation from Σ c (DATN §3.5 step 4), the number of units the trim removed, and the
    tokens spent on protected units (`protected`, DATN §7 risk 3)."""
    if unit == "word":
        to_syllable_mask = lambda keep: expand_word_mask(words, keep)  # noqa: E731
    else:
        to_syllable_mask = lambda keep: keep  # noqa: E731

    costs = unit_costs(words, counter, unit, cost)

    if exact:
        keep = exact_select(scores, costs, budget, priority)
    else:
        keep = greedy_select(scores, costs, budget, alpha=alpha, priority=priority, fill=fill)
    count = lambda k: counter.count(render(words, to_syllable_mask(k)))  # noqa: E731
    selected = keep
    keep = trim_to_budget(keep, scores, budget, count, priority)
    if stats is not None:
        sum_cost = sum(c for c, k in zip(costs, selected) if k)
        real = count(selected)
        stats.update({
            "sum_unit_cost": sum_cost,
            "tokens_before_trim": real,
            "cost_deviation": real - sum_cost,
            "n_trimmed": sum(selected) - sum(keep),
        })
        if protected is not None:
            stats["p_unit_cost"] = sum(c for c, k, p in zip(costs, keep, protected) if k and p)
            stats["n_protected_kept"] = sum(1 for k, p in zip(keep, protected) if k and p)
            stats["n_protected"] = sum(protected)
    mask = to_syllable_mask(keep)
    text = render(words, mask)
    LAST_SELECTION.clear()
    LAST_SELECTION.update(text=text, mask=list(mask))
    return text


def unit_costs(words: list[Word], counter: TokenCounter, unit: str = "word", cost: str = "tokens") -> list[int]:
    """c_T of every decision unit (DATN §3.5 step 1), or n_syl for the H3 length control."""
    units = ([(w.text, w.is_punct) for w in words] if unit == "word"
             else [(s, w.is_punct) for w in words for s in w.syllables])
    if cost == "tokens":
        return [counter.unit_cost(text, is_punct) for text, is_punct in units]
    return [len(text.split(" ")) for text, _ in units]


def question_relevance(words: list[Word], question: str) -> list[float]:
    """Share of a word's content syllables (not stopwords, not punctuation) found in the question."""
    import re

    from .baselines import load_stopwords

    stop = load_stopwords()
    asked = {s for s in re.findall(r"\w+", question.lower()) if s not in stop}
    out = []
    for w in words:
        content = [s.lower() for s in w.syllables if not w.is_punct and s.lower() not in stop]
        out.append(sum(s in asked for s in content) / len(content) if content else 0.0)
    return out


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class ScoredCompressor:
    def __init__(self, scorer, config: SelectionConfig, name: str):
        self.scorer, self.config, self.name = scorer, config, name

    def unit_scores(self, context: Context):
        """(words, scores, priority, protected) of the decision units, before selection."""
        cfg = self.config
        words = flatten(context.doc)
        # The scorer has already applied the fixed hierarchical pooling:
        # subword -> syllable -> word. For word decisions, pool logits first and apply
        # sigmoid afterwards; do not average per-syllable probabilities. SelectionConfig
        # deliberately has no pool switch.
        logits = self.scorer.syllable_logits(context.doc)

        if cfg.unit == "word":
            scores = sigmoid(pool_to_words(logits, words))
            protected = protected_flags(words, cfg.tiers)
        else:
            scores = sigmoid(logits)
            if cfg.protect == "force":
                protected = string_force_syllable_mask(words, cfg.tiers)
            elif cfg.protect == "force_ctx":
                protected = force_syllable_mask(words, cfg.tiers)
            else:
                protected = [naive_syllable_protection(s) for s in syllables_of(words)]

        scores, priority = list(scores), None
        word_bonus = [0.0] * len(words)
        if cfg.scope:
            flags = protected_flags(words, cfg.tiers)
            for op, head in scope_heads(words):
                if flags[op]:
                    word_bonus[head] += cfg.scope
        if cfg.qrel and context.question:
            word_bonus = [b + cfg.qrel * r for b, r in zip(word_bonus, question_relevance(words, context.question))]
        if any(word_bonus):
            bonus = word_bonus if cfg.unit == "word" else [b for w, b in zip(words, word_bonus) for _ in w.syllables]
            scores = [s + b for s, b in zip(scores, bonus)]
        if cfg.protect == "soft":
            scores = [s + cfg.delta * p for s, p in zip(scores, protected)]
        elif cfg.protect in {"hard", "force", "force_ctx"}:
            priority = [int(p) for p in protected]
        return words, scores, priority, [bool(p) for p in protected]

    def compress(self, context: Context, budget: int, counter: TokenCounter) -> str:
        cfg = self.config
        words, scores, priority, protected = self.unit_scores(context)
        self.last_stats = {}
        return select_units(words, scores, budget, counter, unit=cfg.unit, priority=priority,
                            alpha=cfg.alpha, cost=cfg.cost, exact=cfg.exact,
                            protected=protected, stats=self.last_stats)
