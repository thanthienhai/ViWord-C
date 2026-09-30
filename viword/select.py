"""Budgeted selection of units (words or syllables) under a token budget (DATN §3.5).

    maximize  Σ s_i x_i   subject to  Σ c_i x_i <= B,   x_i ∈ {0, 1}

`greedy_select` ranks units by s_i / c_i^α (α = 0: score only, as in LLMLingua-2;
α = 1: classic greedy knapsack), then runs a fill pass. `exact_select` solves the 0/1
knapsack exactly by dynamic programming and is used as a reference for the greedy one.
"""
from __future__ import annotations

from typing import Callable


class TokenCounter:
    """Counts tokens of the target LLM's tokenizer, with a cache for unit costs."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._cache: dict[str, int] = {}

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False)) if text else 0

    def unit_cost(self, text: str, is_punct: bool) -> int:
        """c_T(w): tokens of " " + w (punctuation is counted without the leading space)."""
        key = text if is_punct else " " + text
        if key not in self._cache:
            self._cache[key] = max(1, self.count(key))
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


def exact_select(scores: list[float], costs: list[int], budget: int) -> list[bool]:
    """Exact 0/1 knapsack by dynamic programming, O(n · budget)."""
    n = len(scores)
    best = [0.0] * (budget + 1)
    took = [bytearray(budget + 1) for _ in range(n)]
    for i in range(n):
        c, s = costs[i], scores[i]
        for b in range(budget, c - 1, -1):
            if best[b - c] + s > best[b]:
                best[b] = best[b - c] + s
                took[i][b] = 1
    keep, b = [False] * n, budget
    for i in range(n - 1, -1, -1):
        if took[i][b]:
            keep[i], b = True, b - costs[i]
    return keep


def trim_to_budget(keep: list[bool], scores: list[float], budget: int,
                   count_tokens: Callable[[list[bool]], int]) -> list[bool]:
    """BPE is context dependent, so Σ c_i can differ from the real length of the joined
    text. Drop the lowest-scoring kept units until the real length fits (DATN §3.5, step 4)."""
    keep = list(keep)
    kept = sorted((i for i in range(len(keep)) if keep[i]), key=lambda i: scores[i])
    while kept and count_tokens(keep) > budget:
        keep[kept.pop(0)] = False
    return keep
