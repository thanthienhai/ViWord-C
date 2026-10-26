"""Baselines (DATN §4.2). All follow the `compress(context, budget, counter) -> str` interface.

Rule-based baselines assign a score per word and reuse `select_units`, so every
mask-based system is cut to the same target-token budget with the same rendering.
`LLMLinguaOfficial` uses the package's own text output, while `Precomputed` returns an
external compression; both are deliberately outside the H1 2×2 mask comparison.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from importlib import resources

from .compressor import Context, select_units
from .protect import TIERS, force_tokens_for_context, is_negation
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
    """Keep whole sentences by score until the budget is used, then fill the rest of the
    budget with the head of the best sentence that did not fit; restore original order.
    Without the fill, a context of one long sentence (most ViNLI premises) compresses to
    nothing and the baseline degenerates into `no_context`."""
    words, groups = [], []
    for s in context.doc:
        groups.append(list(range(len(words), len(words) + len(s))))
        words += s
    return _select_groups(words, groups, sentence_scores, budget, counter)


def _select_groups(words: list, groups: list[list[int]], group_scores: list[float], budget: int,
                   counter: TokenCounter, fill: bool = True) -> str:
    """`_select_sentences` for any grouping of the words (sentences, or the units of a legal RAG context).
    Without `fill`, groups are taken in score order only until the first one that does not fit, whose head then
    fills the budget (no smaller, lower-ranked group jumps the queue)."""
    costs = [counter.count(render_all([words[k] for k in g])) + 1 for g in groups]
    keep = greedy_select(group_scores, costs, budget, fill=fill)
    rest = [i for i in range(len(groups)) if not keep[i]]
    partial = max(rest, key=lambda i: group_scores[i]) if rest else None
    scores = [0.0] * len(words)
    for i, g in enumerate(groups):
        for j, k in enumerate(g):
            if keep[i]:
                scores[k] = 2.0
            elif i == partial:
                scores[k] = 1.0 - j / len(g)  # earlier words of the partial group first
            else:
                scores[k] = -1.0 - (k + 1) / 1e6
    return select_units(words, scores, budget, counter, fill=False)


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


class QuestionSentences:
    """Query-aware sentence extraction: sentences ranked by their coverage of the question's
    content syllables (with `choices`, of the mc options too), earlier sentences first on ties.
    The lexical control of `qrel_unit=sentence`: question awareness without a learned scorer."""

    def __init__(self, choices: bool = False):
        self.choices = choices
        self.name = "qsent_choices" if choices else "qsent"

    def compress(self, context, budget, counter):
        from .compressor import query_text, sentence_coverage

        coverage = sentence_coverage(context.doc, query_text(context, self.choices))
        n = len(coverage)
        return _select_sentences(context, [c - i / (1e3 * n) for i, c in enumerate(coverage)], budget, counter)


class QuestionUnits:
    """Query-aware unit extraction for the legal RAG contexts (direction B): the units of the context (blocks
    separated by blank lines, scripts/legal_rag.py) ranked by BM25 against the question, idf over the units of
    this context, earlier units first on ties; whole units in rank order while they fit, then the head of the next
    one, as a RAG pipeline that reranks and truncates. The lexical unit-level control of H1 ("BM25 over units");
    `qsent` is its sentence-level counterpart."""
    name = "qunit"

    def compress(self, context, budget, counter):
        from .legal import word_blocks
        from .legal_index import BM25

        words = flatten(context.doc)
        units = context.text.split("\n\n")
        groups = [[] for _ in units]
        for k, b in enumerate(word_blocks(words, context.text)):
            groups[b].append(k)
        bm25 = BM25.fit(units).scores(context.question or "")
        n = len(units)
        scores = [float(s) - i / (1e3 * n) for i, s in enumerate(bm25)]
        kept = [i for i, g in enumerate(groups) if g]
        return _select_groups(words, [groups[i] for i in kept], [scores[i] for i in kept], budget, counter,
                              fill=False)


def lm_dtype(torch, device: str):
    """Load dtype of the causal LMs (bf16 checkpoints): their own dtype ("auto"), except fp16
    on GPUs without native bf16 (compute capability < 8, e.g. T4), where bf16 is emulated."""
    if str(device).startswith("cuda") and torch.cuda.is_available() \
            and torch.cuda.get_device_capability()[0] < 8:
        return torch.float16
    return "auto"


class CausalLM:
    """Small helper around a causal LM: per-token surprisal (−log p) of a text."""

    def __init__(self, model_name: str, device: str | None = None, max_length: int = 1024):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=lm_dtype(torch, self.device)).to(self.device).eval()
        self.max_length = max_length
        self._cache = {}

    def surprisal(self, text: str) -> tuple[list[float], list[tuple[int, int]]]:
        """Surprisal and character span of every token; chunks of `max_length` tokens.
        Cached by text: the same context is scored once for all ratios."""
        if text not in self._cache:
            self._cache[text] = self._surprisal(text)
        out, spans = self._cache[text]
        return list(out), list(spans)

    def _surprisal(self, text: str) -> tuple[list[float], list[tuple[int, int]]]:
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
    File: JSONL with fields id, ratio, compressed. Missing entries raise an error.
    Ratios are matched to 3 decimals (0.333 = 1/3). The teacher is not budget-controlled;
    with trim=True its output is cut from the end to the budget, like every other method."""

    def __init__(self, path: str, name: str = "precomputed", trim: bool = False):
        import json

        self.name, self.trim, self.path = name, trim, path
        with open(path, encoding="utf-8") as f:
            self.table = {(r["id"], round(float(r["ratio"]), 3)): r["compressed"] for r in map(json.loads, f)}

    def compress(self, context, budget, counter):
        key = (context.example_id, round(float(context.ratio), 3))
        if key not in self.table:
            raise KeyError(f"{self.path} has no compression of {key[0]} at ratio {key[1]}: the file does not "
                           "cover these rows (for the teacher upper bound: run_pipeline.sh teacher_test)")
        text = self.table[key]
        if self.trim:
            words = text.split()
            while words and counter.count(" ".join(words)) > budget:
                words.pop()
            text = " ".join(words)
        return text


# ---------------------------------------------------------------------------- llmlingua

def _search_rate(compress_at, budget: int, counter: TokenCounter, tolerance: float = 0.03,
                 steps: int = 12) -> str:
    """Binary search the method's own `rate` so that the output length measured with the
    target tokenizer lands in [budget·(1−tolerance), budget] (DATN §4.2). Returns the
    longest text found that does not exceed the budget. The upper bound is the budget
    itself, the same hard limit every mask-based method gets from `trim_to_budget`, so the
    package baselines cannot gain up to 3% extra context.

    Some methods cannot compress a short context at all (LLMLingua returns a 32-token ViNLI
    premise unchanged at every rate), so no rate fits. Then the output at the lowest rate is
    cut to its leading words that fit, rather than returning an empty context."""
    low, high, best, best_n = 0.01, 1.0, "", -1
    text = ""
    for _ in range(steps):
        rate = (low + high) / 2
        text = compress_at(rate)
        n = counter.count(text)
        if n <= budget:
            if n > best_n:
                best, best_n = text, n
            if n >= budget * (1 - tolerance):
                break
            low = rate
        else:
            high = rate
    return best if best_n >= 0 else _leading_words(text, budget, counter)


def _leading_words(text: str, budget: int, counter: TokenCounter) -> str:
    """The longest prefix of `text`, in whole words, that fits in `budget` tokens."""
    words = text.split()
    low, high = 0, len(words)
    while low < high:
        mid = (low + high + 1) // 2
        if counter.count(" ".join(words[:mid])) <= budget:
            low = mid
        else:
            high = mid - 1
    return " ".join(words[:low])


class _ForwardCache:
    """Memoize a token classifier's forward pass by its input tensors (LLMLingua-2's rate search).
    LLMLingua-2 chunks the text independently of the rate and thresholds the probabilities, so
    the search over rates then costs one encoder pass per chunk instead of one per step. A
    perplexity compressor (LLMLingua) conditions on the rate-dependent kept prefix, so it is not
    wrapped. Bounded: the cache is cleared after `max_entries` distinct inputs."""

    def __init__(self, model, max_entries: int = 256):
        self._model, self._cache, self._max = model, {}, max_entries

    def __getattr__(self, name):
        return getattr(self._model, name)

    def __call__(self, *args, **kwargs):
        try:
            key = tuple((k, v.shape, v.cpu().numpy().tobytes()) if hasattr(v, "shape") else (k, repr(v))
                        for k, v in sorted(kwargs.items())) + tuple(
                (a.shape, a.cpu().numpy().tobytes()) if hasattr(a, "shape") else repr(a) for a in args)
        except Exception:  # unexpected input type: no caching
            return self._model(*args, **kwargs)
        if key not in self._cache:
            if len(self._cache) >= self._max:
                self._cache.clear()
            self._cache[key] = self._model(*args, **kwargs)
        return self._cache[key]


class _LegacyKVModel:
    """llmlingua 0.2.2 passes, indexes (``kv[0][0]``) and slices the KV cache as nested
    ``[k, v]`` lists. transformers 5 accepts and returns only ``Cache`` objects, so
    LLMLingua and LongLLMLingua crash. This wrapper converts on the way in and out and
    delegates everything else to the model; with an older transformers it is a no-op."""

    def __init__(self, model):
        self._model = model

    def __getattr__(self, name):
        return getattr(self._model, name)

    def __call__(self, *args, past_key_values=None, **kwargs):
        if past_key_values is not None and not hasattr(past_key_values, "get_seq_length"):
            from transformers import DynamicCache

            legacy = tuple(tuple(kv) for kv in past_key_values)
            past_key_values = (DynamicCache.from_legacy_cache(legacy)
                               if hasattr(DynamicCache, "from_legacy_cache") else DynamicCache(legacy))
        out = self._model(*args, past_key_values=past_key_values, **kwargs)
        cache = getattr(out, "past_key_values", None)
        if cache is not None and hasattr(cache, "layers"):  # transformers >= 5
            out.past_key_values = [[layer.keys, layer.values] for layer in cache.layers]
        return out


class LLMLinguaOfficial:
    """Published LLMLingua / LLMLingua-2 / LongLLMLingua through the package.

    ``force_from_context`` is the package-level control: it builds the exact
    contextual P list from VnCoreNLP words for every example, including phrases,
    numbers and NER spans. ``units`` (LongLLMLingua only) passes the blank-line units of a legal
    RAG context as separate documents, so that its coarse step ranks clauses (direction B).
    """

    LLMLINGUA2 = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"

    def __init__(self, version: str = "llmlingua2", lm_name: str = "Qwen/Qwen2.5-1.5B",
                 force_tokens: list[str] | None = None, force_from_context: bool = False,
                 force_tiers: tuple[str, ...] = TIERS, device: str = "cuda", units: bool = False):
        from llmlingua import PromptCompressor

        self.version, self.units = version, units
        self.force_tokens = force_tokens or []
        self.force_from_context = force_from_context
        self.force_tiers = force_tiers
        self.name = version + ("_force" if self.force_tokens or force_from_context else "")
        if version == "llmlingua2":
            self.pc = PromptCompressor(model_name=self.LLMLINGUA2, use_llmlingua2=True, device_map=device)
            # the binary search over `rate` re-runs the classifier on the same chunks at every step;
            # its keep probabilities do not depend on the rate, so one forward pass per chunk suffices
            if hasattr(self.pc, "model"):
                self.pc.model = _ForwardCache(self.pc.model)
        else:  # "llmlingua" or "longllmlingua": perplexity of a small causal LM
            import torch
            self.pc = PromptCompressor(model_name=lm_name, device_map=device,
                                       model_config={"torch_dtype": lm_dtype(torch, device)})
            self.pc.model = _LegacyKVModel(self.pc.model)

    def _at_rate(self, context: Context, rate: float) -> str:
        if self.version == "llmlingua2":
            force_tokens = self.force_tokens
            if self.force_from_context:
                if context.doc is None:
                    raise ValueError("contextual force_tokens requires the segmented document")
                force_tokens = force_tokens_for_context(flatten(context.doc), self.force_tiers)
            out = self.pc.compress_prompt(context.text, rate=rate, force_tokens=force_tokens,
                                          force_reserve_digit=bool(force_tokens))
        elif self.version == "longllmlingua":
            docs = context.text.split("\n\n") if self.units else [context.text]
            out = self.pc.compress_prompt(docs, question=context.question or "", rate=rate,
                                          condition_in_question="after_condition",
                                          rank_method="longllmlingua", condition_compare=True)
        else:
            out = self.pc.compress_prompt(context.text, rate=rate)
        return out["compressed_prompt"]

    def compress(self, context, budget, counter):
        return _search_rate(lambda r: self._at_rate(context, r), budget, counter)
