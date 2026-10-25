"""Budgeted selection of units (words or syllables) under a token budget (DATN §3.5).

    maximize  Σ s_i x_i   subject to  Σ c_i x_i <= B,   x_i ∈ {0, 1}

`greedy_select` ranks units by s_i / c_i^α (α = 0: score only, as in LLMLingua-2;
α = 1: classic greedy knapsack), then runs a fill pass. `exact_select` solves the 0/1
knapsack exactly by dynamic programming (`exact=1` in a method spec); with hard
protection the protected units are taken first, as in the greedy pass 1, and the DP fills
the remaining budget.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


class TokenCounter:
    """Counts tokens of the target LLM's tokenizer, with a cache for unit costs."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._cache: dict[str, int] = {}

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False)) if text else 0

    def unit_cost(self, text: str, is_punct: bool) -> int:
        """c_T(w): tokens w adds after a preceding word ("x" + " " + w minus "x"; punctuation without
        the space). Counting " " + w alone costs an extra "▁" with SentencePiece tokenizers that add a
        dummy prefix (SeaLLM-7B-v2)."""
        key = text if is_punct else " " + text
        if key not in self._cache:
            self._cache[key] = max(1, self.count("x" + key) - self.count("x"))
        return self._cache[key]


def greedy_select(scores: list[float], costs: list[int], budget: int, alpha: float = 0.0,
                  priority: list[int] | None = None, fill: bool = True) -> list[bool]:
    n = len(scores)
    priority = priority or [0] * n
    ratio = [s / (c ** alpha) for s, c in zip(scores, costs)]
    keep, used = [False] * n, 0

    # Pass 1: take units in (priority, ratio) order until the first one that does not fit.
    for i in sorted(range(n), key=lambda i: (-priority[i], -ratio[i])):
        if used + costs[i] > budget:
            break
        keep[i], used = True, used + costs[i]

    # Pass 2 (fill): add remaining units by score if they still fit.
    if fill:
        for i in sorted(range(n), key=lambda i: -scores[i]):
            if not keep[i] and used + costs[i] <= budget:
                keep[i], used = True, used + costs[i]
    return keep


def exact_select(scores: list[float], costs: list[int], budget: int,
                 priority: list[int] | None = None) -> list[bool]:
    """Exact 0/1 knapsack by dynamic programming, O(n · budget) (numpy over the budget axis).

    With `priority`, units of the highest priority are taken first by score while they fit
    (lexicographic: protection before value), and the DP runs on the rest."""
    n = len(scores)
    keep, budget = [False] * n, max(int(budget), 0)
    free = list(range(n))
    if priority and any(priority):
        for i in sorted((i for i in range(n) if priority[i]), key=lambda i: (-priority[i], -scores[i])):
            if costs[i] <= budget:
                keep[i], budget = True, budget - costs[i]
        free = [i for i in range(n) if not priority[i]]
    best = np.zeros(budget + 1)
    took = np.zeros((len(free), budget + 1), dtype=bool)
    for k, i in enumerate(free):
        c, s = costs[i], scores[i]
        if c > budget:
            continue
        candidate = best[:budget + 1 - c] + s
        better = candidate > best[c:]
        took[k, c:] = better
        best[c:] = np.where(better, candidate, best[c:])
    b = budget
    for k in range(len(free) - 1, -1, -1):
        if took[k, b]:
            keep[free[k]], b = True, b - costs[free[k]]
    return keep


def trim_to_budget(keep: list[bool], scores: list[float], budget: int,
                   count_tokens: Callable[[list[bool]], int],
                   priority: list[int] | None = None) -> list[bool]:
    """BPE is context dependent, so Σ c_i can differ from the real length of the joined
    text. Drop the lowest-scoring kept units until the real length fits (DATN §3.5, step 4).
    With hard protection, unprotected units are dropped before any protected one, so the
    trim step cannot undo the priority of the selection step."""
    keep = list(keep)
    priority = priority or [0] * len(keep)
    kept = sorted((i for i in range(len(keep)) if keep[i]), key=lambda i: (priority[i], scores[i]))
    while kept and count_tokens(keep) > budget:
        keep[kept.pop(0)] = False
    return keep
