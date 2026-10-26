"""BM25 over legal units (direction B): lower-case syllables, a sparse unit × term matrix of BM25 weights.

Scoring a question sums the weight columns of its distinct syllables, so one query over the ~1.3 M clause units
takes milliseconds. Okapi BM25 with k1 = 1.2, b = 0.75 and the Robertson–Spärck Jones idf (+1 inside the log,
never negative).
"""
from __future__ import annotations

import json
import re

import numpy as np

from .legal import normalize

_TOKEN = re.compile(r"\w+")


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(normalize(text).lower())


class BM25:
    def __init__(self, weights, vocab: dict[str, int]):
        self.weights = weights  # scipy.sparse.csc_matrix, units × terms
        self.vocab = vocab

    @classmethod
    def fit(cls, texts, k1: float = 1.2, b: float = 0.75) -> "BM25":
        from sklearn.feature_extraction.text import CountVectorizer

        vec = CountVectorizer(tokenizer=tokens, lowercase=False, token_pattern=None, dtype=np.float32)
        x = vec.fit_transform(texts).tocsr()
        n = x.shape[0]
        length = np.asarray(x.sum(axis=1)).ravel()
        df = np.bincount(x.indices, minlength=x.shape[1])
        idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        row_length = np.repeat(length, np.diff(x.indptr))
        tf = x.data
        x.data = (idf[x.indices] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * row_length / length.mean()))).astype(
            np.float32)
        return cls(x.tocsc(), {t: int(i) for t, i in vec.vocabulary_.items()})

    def scores(self, query: str) -> np.ndarray:
        ids = sorted({self.vocab[t] for t in tokens(query) if t in self.vocab})
        if not ids:
            return np.zeros(self.weights.shape[0], dtype=np.float32)
        return np.asarray(self.weights[:, ids].sum(axis=1)).ravel()

    def search(self, query: str, k: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """Indices of the k best units, best first, and their scores."""
        s = self.scores(query)
        k = min(k, len(s))
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top], kind="stable")]
        return top, s[top]

    def save(self, prefix: str) -> None:
        from scipy import sparse

        sparse.save_npz(prefix + ".npz", self.weights)
        with open(prefix + "_vocab.json", "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False)

    @classmethod
    def load(cls, prefix: str) -> "BM25":
        from scipy import sparse

        with open(prefix + "_vocab.json", encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(sparse.load_npz(prefix + ".npz").tocsc(), vocab)
