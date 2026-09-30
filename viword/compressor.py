"""Compressors share one interface:

    compress(context: Context, budget: int, counter: TokenCounter) -> str

`budget` is counted in tokens of the target LLM (DATN §4.2). `ScoredCompressor` turns
per-syllable keep-logits (from any scorer) into a compressed text; it covers ViWord-C, the
LLMLingua-2-vi control, all cells of the 2×2 design (DATN §2.3), and the lexical-prior
control, which differ only in their scorer and `SelectionConfig`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import pool_to_words
from .protect import naive_syllable_protection, protection_tiers
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
    protect: str = "none"  # "none", "soft", "hard"; syllable unit uses naive force_tokens
    delta: float = 0.2  # soft protection bonus
    tiers: tuple[str, ...] = ("T1", "T2", "T3")
    alpha: float = 0.0  # cost exponent (H3)
    cost: str = "tokens"  # "tokens" = c_T(w); "syllables" = n_syl(w) (H3 control)
    exact: bool = False  # exact DP knapsack instead of greedy (reference only)


def select_units(words: list[Word], scores: list[float], budget: int, counter: TokenCounter,
                 unit: str = "word", priority: list[int] | None = None, alpha: float = 0.0,
                 cost: str = "tokens", exact: bool = False, fill: bool = True) -> str:
    """Pick units under the budget, then fix the real token length. Returns text."""
    if unit == "word":
        units = [(w.text, w.is_punct) for w in words]
        to_syllable_mask = lambda keep: expand_word_mask(words, keep)  # noqa: E731
    else:
        units = [(s, w.is_punct) for w in words for s in w.syllables]
        to_syllable_mask = lambda keep: keep  # noqa: E731

    if cost == "tokens":
        costs = [counter.unit_cost(text, is_punct) for text, is_punct in units]
    else:
        costs = [len(text.split(" ")) for text, _ in units]

    if exact:
        keep = exact_select(scores, costs, budget)
    else:
        keep = greedy_select(scores, costs, budget, alpha=alpha, priority=priority, fill=fill)
    keep = trim_to_budget(keep, scores, budget,
                          lambda k: counter.count(render(words, to_syllable_mask(k))))
    return render(words, to_syllable_mask(keep))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class ScoredCompressor:
    def __init__(self, scorer, config: SelectionConfig, name: str):
        self.scorer, self.config, self.name = scorer, config, name

    def compress(self, context: Context, budget: int, counter: TokenCounter) -> str:
        cfg = self.config
        words = flatten(context.doc)
        logits = self.scorer.syllable_logits(context.doc)

        if cfg.unit == "word":
            scores = sigmoid(pool_to_words(logits, words))
            protected = [t in cfg.tiers for t in protection_tiers(words)]
        else:
            scores = sigmoid(logits)
            protected = [naive_syllable_protection(s) for s in syllables_of(words)]

        scores, priority = list(scores), None
        if cfg.protect == "soft":
            scores = [s + cfg.delta * p for s, p in zip(scores, protected)]
        elif cfg.protect == "hard":
            priority = [int(p) for p in protected]
        return select_units(words, scores, budget, counter, unit=cfg.unit, priority=priority,
                            alpha=cfg.alpha, cost=cfg.cost, exact=cfg.exact)
