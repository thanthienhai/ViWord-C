"""Encoder scorer: one keep-logit per syllable from a token-classification model.

Works for our trained ViWord-C/LLMLingua-2-vi checkpoints (1 output logit) and for the
published LLMLingua-2 checkpoint (2 labels, label 1 = keep). Syllables are fed to the
tokenizer as pre-split words, so XLM-R sees the same "▁"-separated units as LLMLingua-2.

Pooling (DATN §3.1) is fixed and hierarchical: a syllable logit is the arithmetic mean
of its subword logits; a word logit is the arithmetic mean of its syllable logits
(`pool_to_words`). Thus syllables, rather than subwords, have equal weight within a
word; no alternative pooling variant is selected on dev.

Long documents are split into windows of whole sentences (≤ max_length subwords) that
overlap by one sentence; logits in the overlap are averaged.

Syllable logits depend only on the checkpoint and the segmented document, not on the
ratio, π, α, the budget or the reader, so they are computed once: in memory for every
spec sharing a scorer, and on disk when VIWORD_LOGIT_CACHE (a directory) is set. The
encoder is loaded only on a cache miss, so every selection variant of a cached
checkpoint × data set runs on CPU.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .segment import Document, Word, flatten, syllables_of


def pool_to_words(syllable_values: np.ndarray, words: list[Word]) -> np.ndarray:
    """Pool fixed syllable logits into one arithmetic-mean logit per word."""
    out, i = [], 0
    for w in words:
        n = len(w.syllables)
        out.append(float(np.mean(syllable_values[i:i + n])))
        i += n
    return np.array(out)


def make_windows(doc: Document, tokenizer, max_length: int = 512,
                 max_segment_syllables: int = 200) -> list[tuple[int, int]]:
    """Windows as (start, end) syllable indices over the flattened document.

    Sentences longer than `max_segment_syllables` are cut into pieces first, so that
    every piece fits in one window. Consecutive windows share one segment.
    """
    segments, start = [], 0  # (start, end, n_subwords)
    for sentence in doc:
        sylls = syllables_of(sentence)
        for k in range(0, len(sylls), max_segment_syllables):
            piece = sylls[k:k + max_segment_syllables]
            n_sub = len(tokenizer(piece, is_split_into_words=True, add_special_tokens=False)["input_ids"])
            segments.append((start + k, start + k + len(piece), n_sub))
        start += len(sylls)

    budget = max_length - 2  # special tokens
    windows, i = [], 0
    while i < len(segments):
        j, used = i, 0
        while j < len(segments) and used + segments[j][2] <= budget:
            used += segments[j][2]
            j += 1
        j = max(j, i + 1)  # a single over-long segment still gets its own (truncated) window
        windows.append((segments[i][0], segments[j - 1][1]))
        if j >= len(segments):
            break
        i = j - 1 if j - 1 > i else j  # overlap by one segment
    return windows


def model_fingerprint(model_name_or_path: str) -> str:
    """Identity of a checkpoint for the logit cache: the hub id, or for a local directory its
    path plus the size and mtime of its config and weight files (a retrained run is a miss)."""
    path = Path(model_name_or_path)
    parts = [str(model_name_or_path)]
    if path.is_dir():
        for f in sorted(path.iterdir()):
            if f.suffix in {".json", ".safetensors", ".bin"} and f.name != "train_meta.json":
                st = f.stat()
                parts.append(f"{f.name}:{st.st_size}:{int(st.st_mtime)}")
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]


class SyllableScorer:
    def __init__(self, model_name_or_path: str, device: str | None = None,
                 max_length: int = 512, keep_label: int = 1, cache_dir: str | None = None):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.max_length = max_length
        self.keep_label = keep_label
        cache_dir = cache_dir or os.environ.get("VIWORD_LOGIT_CACHE") or None
        self.cache_dir = (Path(cache_dir) / f"{model_fingerprint(model_name_or_path)}_{max_length}_{keep_label}"
                          if cache_dir else None)
        self.model = None

    def _load(self):
        if getattr(self, "model", None) is not None:
            return
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.torch = torch
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name_or_path)
        self.model.to(self.device).eval()

    def _keep_logits(self, logits):
        if logits.shape[-1] == 1:
            return logits[..., 0]
        # two-label model: log-odds of "keep" vs "drop"
        return logits[..., self.keep_label] - logits[..., 1 - self.keep_label]

    def syllable_logits(self, doc: Document) -> np.ndarray:
        # Logits depend only on the syllables and sentence boundaries, not on the ratio or the
        # selection config, so every spec sharing this scorer (methods.py) reuses them.
        key = tuple(tuple(syllables_of(sentence)) for sentence in doc)
        cache = self.__dict__.setdefault("_logit_cache", {})
        if key not in cache:
            path = None
            if getattr(self, "cache_dir", None) is not None:
                digest = hashlib.sha1(json.dumps(key, ensure_ascii=False).encode("utf-8")).hexdigest()
                path = self.cache_dir / digest[:2] / f"{digest}.npy"
            if path is not None and path.exists():
                cache[key] = np.load(path)
            else:
                cache[key] = self._syllable_logits(doc)
                if path is not None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_suffix(f".{os.getpid()}.tmp.npy")
                    np.save(tmp, cache[key])
                    os.replace(tmp, path)
        return cache[key].copy()

    def _syllable_logits(self, doc: Document) -> np.ndarray:
        self._load()
        sylls = syllables_of(flatten(doc))
        total, count = np.zeros(len(sylls)), np.zeros(len(sylls))
        for start, end in make_windows(doc, self.tokenizer, self.max_length):
            enc = self.tokenizer(sylls[start:end], is_split_into_words=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt")
            with self.torch.no_grad():
                logits = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits[0]
            keep = self._keep_logits(logits.float()).cpu().numpy()
            sums, ns = np.zeros(end - start), np.zeros(end - start)
            for pos, wid in enumerate(enc.word_ids()):
                if wid is not None:
                    sums[wid] += keep[pos]
                    ns[wid] += 1
            covered = ns > 0
            total[start:end][covered] += sums[covered] / ns[covered]
            count[start:end][covered] += 1
        # syllables never seen (truncated windows) get a neutral logit of 0
        return np.divide(total, count, out=np.zeros_like(total), where=count > 0)


class LexicalPriorScorer:
    """Control 'tra từ' (DATN §4.2): keep-probability of a word type, no context.

    p_keep(w) = teacher keep rate of the lowercased word w on the training set (with
    add-one smoothing); unseen words get the global keep rate.
    """

    def __init__(self, table: dict[str, float], default: float):
        self.table, self.default = table, default

    @classmethod
    def fit(cls, records: list[dict]) -> "LexicalPriorScorer":
        from .segment import doc_from_json

        kept, seen, total_kept, total = {}, {}, 0.0, 0
        for r in records:
            words = flatten(doc_from_json(r["doc"]))
            for w, label in zip(words, r["word_labels"]):
                key = w.text.lower()
                kept[key] = kept.get(key, 0.0) + label
                seen[key] = seen.get(key, 0) + 1
                total_kept, total = total_kept + label, total + 1
        default = total_kept / max(total, 1)
        table = {k: (kept[k] + default) / (seen[k] + 1) for k in seen}
        return cls(table, default)

    def syllable_logits(self, doc: Document) -> np.ndarray:
        out = []
        for w in flatten(doc):
            p = min(max(self.table.get(w.text.lower(), self.default), 1e-4), 1 - 1e-4)
            out += [np.log(p / (1 - p))] * len(w.syllables)
        return np.array(out)
