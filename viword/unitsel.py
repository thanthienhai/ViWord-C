"""The compressor of direction B: select clauses by a question-aware unit scorer, then compress finely inside.

For a legal RAG context (units separated by blank lines, scripts/legal_rag.py):
1. score every unit against the question (`viword.unit_scorer`: a trained cross-encoder, a reranker, or BM25);
2. order the units greedily by z-scored relevance minus lambda times their largest overlap with the units already
   ordered (Marginal Information Gain, after COMI; lambda = 0 is the plain ranking);
3. keep units whole in that order while they fit in the budget;
4. fill the rest of the budget from the next unit by token scores (`fine`, a ScoredCompressor, usually with the
   question-aware sentence bonus), or with its head when there is no fine compressor.
At 1/16 of a ViBidLQA context the gold clause alone is ~1.7 times the budget, so step 4 then works on the best unit.
The output keeps the context order and goes through the shared `select_units`/`render` path.
"""
from __future__ import annotations

import re

import numpy as np

from .compressor import Context, ScoredCompressor, select_units
from .legal import word_blocks
from .segment import flatten, render_all
from .select import TokenCounter

_SYLLABLE = re.compile(r"\w+")


def mig_order(scores: list[float], texts: list[str], lam: float = 0.0) -> list[int]:
    """Units in greedy order of z-scored relevance minus lam × the largest syllable-set Jaccard overlap with the
    units ordered before them."""
    s = np.asarray(scores, dtype=float)
    z = (s - s.mean()) / s.std() if len(s) > 1 and s.std() > 0 else np.zeros_like(s)
    if not lam:
        return sorted(range(len(s)), key=lambda i: (-z[i], i))
    bags = [set(_SYLLABLE.findall(t.lower())) for t in texts]
    jaccard = lambda a, b: len(a & b) / len(a | b) if a | b else 0.0  # noqa: E731
    order, left = [], list(range(len(s)))
    while left:
        best = max(left, key=lambda i: (z[i] - lam * max((jaccard(bags[i], bags[j]) for j in order), default=0.0),
                                        -i))
        order.append(best)
        left.remove(best)
    return order


class UnitSelectCompressor:
    """`level="sentence"` selects the segmenter's sentences instead of the clauses, with the same scorer (H2)."""

    def __init__(self, unit_scorer, fine: ScoredCompressor | None = None, lam: float = 0.0, name: str = "unitsel",
                 level: str = "unit"):
        if level not in ("unit", "sentence"):
            raise ValueError(f"level must be unit or sentence, got {level!r}")
        self.unit_scorer, self.fine, self.lam, self.name, self.level = unit_scorer, fine, lam, name, level

    def compress(self, context: Context, budget: int, counter: TokenCounter) -> str:
        words = flatten(context.doc)
        if self.level == "sentence":
            texts = [render_all(s) for s in context.doc]
            blocks = [i for i, s in enumerate(context.doc) for _ in s]
        else:
            texts = context.text.split("\n\n")
            blocks = word_blocks(words, context.text)
        members = [[] for _ in texts]
        for k, b in enumerate(blocks):
            members[b].append(k)
        present = [i for i, m in enumerate(members) if m]
        scores = self.unit_scorer.scores(context.question or "", [texts[i] for i in present])
        order = [present[i] for i in mig_order(scores, [texts[i] for i in present], self.lam)]

        whole, used, partial = set(), 0, None
        for i in order:
            cost = counter.count(render_all([words[k] for k in members[i]])) + 2
            if used + cost > budget:
                partial = i
                break
            whole.add(i)
            used += cost
        keep_units = whole | ({partial} if partial is not None else set())

        unit = self.fine.config.unit if self.fine is not None else "word"
        if self.fine is not None:
            _, token_scores, _, _ = self.fine.unit_scores(context)
        else:  # the head of the partial unit: earlier words first
            token_scores = [-k / max(1, len(words)) for k in range(len(words))]
            if unit == "syllable":
                token_scores = [s for w, s in zip(words, token_scores) for _ in w.syllables]
        # per decision unit (word or syllable): its block, then the words of the kept units only
        block_of = blocks if unit == "word" else [b for w, b in zip(words, blocks) for _ in w.syllables]
        kept_words = [w for w, b in zip(words, blocks) if b in keep_units]
        sub = [(s, b) for s, b in zip(token_scores, block_of) if b in keep_units]
        sub_scores = [float(s) for s, _ in sub]
        priority = [int(b in whole) for _, b in sub]
        return select_units(kept_words, sub_scores, budget, counter, unit=unit, priority=priority)
