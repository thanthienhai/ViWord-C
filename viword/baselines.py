"""Baselines (DATN §4.2). All follow the `compress(context, budget, counter) -> str` interface.

Rule-based baselines assign a score per word and reuse `select_units`, so every system is
cut to the same target-token budget with the same rendering.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from importlib import resources

from .compressor import Context, select_units
from .protect import is_negation
from .segment import flatten, render, render_all
from .select import TokenCounter, greedy_select


def load_stopwords() -> set[str]:
    text = resources.files("viword").joinpath("resources/stopwords_vi.txt").read_text(encoding="utf-8")
    return {line.strip().lower() for line in text.splitlines() if line.strip() and not line.startswith("#")}


class NoCompression:
    name = "none"

    def compress(self, context: Context, budget: int, counter: TokenCounter) -> str:
        return context.text


class NoContext:
    name = "no_context"

    def compress(self, context: Context, budget: int, counter: TokenCounter) -> str:
        return ""


class RandomWords:
    def __init__(self, seed: int = 0):
        self.seed, self.name = seed, f"random_s{seed}"

    def compress(self, context, budget, counter):
        words = flatten(context.doc)
        rng = random.Random(self.seed)
        return select_units(words, [rng.random() for _ in words], budget, counter)


class LeadK:
    """Keep the beginning of the context (strong on news, DATN §4.1)."""
    name = "lead"

    def compress(self, context, budget, counter):
        words = flatten(context.doc)
        return select_units(words, [-i for i in range(len(words))], budget, counter, fill=False)


class Truncation:
    """Keep the head and the tail of the context, roughly half each."""
    name = "truncation"

    def compress(self, context, budget, counter):
        words = flatten(context.doc)
        n = len(words)
        return select_units(words, [-min(i, n - 1 - i) for i in range(n)], budget, counter, fill=False)


class StopwordRemoval:
    """Drop stopwords first, then later words (earlier words are preferred)."""
    name = "stopword"

    def __init__(self):
        self.stopwords = load_stopwords()

    def compress(self, context, budget, counter):
        words = flatten(context.doc)
        n = len(words)
        scores = [(0.0 if w.text.lower() in self.stopwords else 1.0) + 0.5 * (1 - i / n)
                  for i, w in enumerate(words)]
        return select_units(words, scores, budget, counter)


class LexicalRules:
    """Deterministic lexical pipeline (stopwords + POS + named entities), after 2609.13154."""
    name = "lexical"
    CONTENT_TAGS = {"N", "Np", "Ny", "V", "A", "M"}

    def __init__(self):
        self.stopwords = load_stopwords()

    def compress(self, context, budget, counter):
        words = flatten(context.doc)
        n = len(words)
        scores = []
        for i, w in enumerate(words):
            if w.ner != "O" or w.pos == "M":
                level = 3
            elif w.pos in self.CONTENT_TAGS:
                level = 2
            elif w.text.lower() not in self.stopwords and not w.is_punct:
                level = 1
            else:
                level = 0
            scores.append(level + 0.5 * (1 - i / n))
        return select_units(words, scores, budget, counter)


def _select_sentences(context: Context, sentence_scores: list[float], budget: int,
                      counter: TokenCounter) -> str:
    """Keep whole sentences by score until the budget is used; restore original order."""
    sentences = context.doc
    costs = [counter.count(render_all(s)) + 1 for s in sentences]
    keep = greedy_select(sentence_scores, costs, budget)
    words = [w for s, k in zip(sentences, keep) if k for w in s]
    return render_all(words) if words else ""


class TfidfSentences:
    """Sentence extraction by mean TF-IDF of the sentence's words (IDF over the sentences)."""
    name = "tfidf_sent"

    def compress(self, context, budget, counter):
        bags = [[w.text.lower() for w in s if not w.is_punct] for s in context.doc]
        df = Counter(t for bag in bags for t in set(bag))
        n = len(bags)
        scores = []
        for bag in bags:
            tf = Counter(bag)
            tfidf = [tf[t] * math.log((1 + n) / (1 + df[t])) for t in tf]
            scores.append(sum(tfidf) / max(len(bag), 1))
        return _select_sentences(context, scores, budget, counter)


class CausalLM:
    """Small helper around a causal LM: per-token surprisal (−log p) of a text."""

    def __init__(self, model_name: str, device: str | None = None, max_length: int = 1024):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").to(self.device).eval()
        self.max_length = max_length

    def surprisal(self, text: str) -> tuple[list[float], list[tuple[int, int]]]:
        """Surprisal and character span of every token; chunks of `max_length` tokens."""
        enc = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        ids, spans = enc["input_ids"], enc["offset_mapping"]
        out = []
        for start in range(0, len(ids), self.max_length):
            chunk = self.torch.tensor([ids[start:start + self.max_length]], device=self.device)
            with self.torch.no_grad():
                logp = self.torch.log_softmax(self.model(chunk).logits[0].float(), dim=-1)
            # the first token of each chunk has no left context: give it the chunk mean later
            token_s = [-logp[k - 1, chunk[0, k]].item() for k in range(1, chunk.shape[1])]
            first = sum(token_s) / len(token_s) if token_s else 0.0
            out += [first] + token_s
        return out, spans


class PerplexitySentences:
    """Keep the most surprising sentences (as `lacc_sentence`, the best LACC arm)."""

    def __init__(self, lm: CausalLM):
        self.lm, self.name = lm, "ppl_sent"

    def compress(self, context, budget, counter):
        scores = []
        for sentence in context.doc:
            s, _ = self.lm.surprisal(render_all(sentence))
            scores.append(sum(s) / max(len(s), 1))
        return _select_sentences(context, scores, budget, counter)


class SelectiveContext:
    """Selective Context (Li et al., 2023) re-implemented at the word level: a word's
    self-information is the sum of its tokens' surprisal; keep the most informative words."""

    def __init__(self, lm: CausalLM):
        self.lm, self.name = lm, "selective_context"

    def compress(self, context, budget, counter):
        words = flatten(context.doc)
        text, owner = "", []  # owner[c] = index of the word covering character c
        for i, w in enumerate(words):
            if text:
                text += " "
                owner.append(i)
            text += w.text
            owner += [i] * len(w.text)
        surprisal, spans = self.lm.surprisal(text)
        scores = [0.0] * len(words)
        for s, (a, b) in zip(surprisal, spans):
            while a < b and text[a] == " ":  # a token " học" belongs to the word "học"
                a += 1
            if a < len(owner):
                scores[owner[a]] += s
        return select_units(words, scores, budget, counter)


class NegationProbe:
    """Upper-bound probe for H2: delete only negation words, nothing else (no budget)."""
    name = "probe_negation"

    def compress(self, context, budget, counter):
        words = flatten(context.doc)
        mask = [not is_negation(words, i) for i, w in enumerate(words) for _ in w.syllables]
        return render(words, mask)


class Precomputed:
    """Compressions produced elsewhere (e.g. the teacher upper bound, DATN §4.2).
    File: JSONL with fields id, ratio, compressed. Missing entries raise an error."""

    def __init__(self, path: str, name: str = "precomputed"):
        import json

        self.name = name
        with open(path, encoding="utf-8") as f:
            self.table = {(r["id"], float(r["ratio"])): r["compressed"] for r in map(json.loads, f)}

    def compress(self, context, budget, counter):
        return self.table[(context.example_id, float(context.ratio))]


# ---------------------------------------------------------------------------- llmlingua

def _search_rate(compress_at, budget: int, counter: TokenCounter, tolerance: float = 0.03,
                 steps: int = 12) -> str:
    """Binary search the method's own `rate` so that the output length measured with the
    target tokenizer is within ±tolerance of the budget (DATN §4.2). Returns the best text
    that does not exceed budget·(1+tolerance)."""
    low, high, best = 0.01, 1.0, ""
    for _ in range(steps):
        rate = (low + high) / 2
        text = compress_at(rate)
        n = counter.count(text)
        if n <= budget * (1 + tolerance):
            best = text
            if n >= budget * (1 - tolerance):
                break
            low = rate
        else:
            high = rate
    return best


class LLMLinguaOfficial:
    """Published LLMLingua / LLMLingua-2 / LongLLMLingua through the `llmlingua` package."""

    LLMLINGUA2 = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"

    def __init__(self, version: str = "llmlingua2", lm_name: str = "Qwen/Qwen2.5-1.5B",
                 force_tokens: list[str] | None = None, device: str = "cuda"):
        from llmlingua import PromptCompressor

        self.version, self.force_tokens = version, force_tokens or []
        self.name = version + ("_force" if force_tokens else "")
        if version == "llmlingua2":
            self.pc = PromptCompressor(model_name=self.LLMLINGUA2, use_llmlingua2=True, device_map=device)
        else:  # "llmlingua" or "longllmlingua": perplexity of a small causal LM
            self.pc = PromptCompressor(model_name=lm_name, device_map=device)

    def _at_rate(self, context: Context, rate: float) -> str:
        if self.version == "llmlingua2":
            out = self.pc.compress_prompt(context.text, rate=rate, force_tokens=self.force_tokens,
                                          force_reserve_digit=bool(self.force_tokens))
        elif self.version == "longllmlingua":
            out = self.pc.compress_prompt([context.text], question=context.question or "", rate=rate,
                                          condition_in_question="after_condition",
                                          rank_method="longllmlingua", condition_compare=True)
        else:
            out = self.pc.compress_prompt(context.text, rate=rate)
        return out["compressed_prompt"]

    def compress(self, context, budget, counter):
        return _search_rate(lambda r: self._at_rate(context, r), budget, counter)
