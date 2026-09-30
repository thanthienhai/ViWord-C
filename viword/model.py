"""Encoder scorer: one keep-logit per syllable from a token-classification model.

Works for our trained ViWord-C/LLMLingua-2-vi checkpoints (1 output logit) and for the
published LLMLingua-2 checkpoint (2 labels, label 1 = keep). Syllables are fed to the
tokenizer as pre-split words, so XLM-R sees the same "▁"-separated units as LLMLingua-2.

Pooling (DATN §3.1): syllable logit = mean of its subword logits; word logit = mean of
its syllable logits (`pool_to_words`).

Long documents are split into windows of whole sentences (≤ max_length subwords) that
overlap by one sentence; logits in the overlap are averaged.
"""
from __future__ import annotations

import numpy as np

from .segment import Document, Word, flatten, syllables_of


def pool_to_words(syllable_values: np.ndarray, words: list[Word]) -> np.ndarray:
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


class SyllableScorer:
    def __init__(self, model_name_or_path: str, device: str | None = None,
                 max_length: int = 512, keep_label: int = 1):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)
        self.model.to(self.device).eval()
        self.max_length = max_length
        self.keep_label = keep_label

    def _keep_logits(self, logits):
        if logits.shape[-1] == 1:
            return logits[..., 0]
        # two-label model: log-odds of "keep" vs "drop"
        return logits[..., self.keep_label] - logits[..., 1 - self.keep_label]

    def syllable_logits(self, doc: Document) -> np.ndarray:
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
