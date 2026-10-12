"""Build compressors from short text specs, shared by the diagnose and evaluate scripts.

Spec syntax: `name` or `name:key=value,key=value`. Examples:

  none | no_context | lead | truncation | stopword | lexical | tfidf_sent | probe_negation
  random:seed=1
  ppl_sent:lm=Qwen/Qwen2.5-1.5B            selective_context:lm=Qwen/Qwen2.5-1.5B
  llmlingua2 | llmlingua2_force | llmlingua:lm=Qwen/Qwen2.5-1.5B | longllmlingua:lm=...
  translate:inner=llmlingua2                (Belebele only: compress the English passage)
  scored:model=runs/viword_s0,unit=word,protect=soft,delta=0.2,alpha=0,name=viword
  scored:model=runs/llmlingua2vi_s0,unit=syllable,protect=force,tiers=T1+T2,name=...
  lexprior:train=data/distill/distilled.jsonl,exclude=runs/viword_s0/dev_ids.jsonl
  precomputed:path=results/teacher_dev/vinli_dev.jsonl,trim=1,name=teacher

`scored` covers ViWord-C, LLMLingua-2-vi and the 2×2 cells. Every `scored` arm uses the
shared Vietnamese `render()` after selecting its syllable mask. For every `unit=word` arm,
it pools logits by word and applies sigmoid after pooling; with
model=microsoft/llmlingua-2-xlm-roberta-large-meetingbank it scores with the published
LLMLingua-2 weights (e.g. unit=word for "LLMLingua-2 + word pooling").
The package baseline `llmlingua2`, `translate`, and `precomputed` are text-only/external
outputs and are not H1 2×2 cells.
`llmlingua2_force` is the package-level P diagnostic; the preregistered H2 row is the
`scored:...,unit=syllable,protect=force` arm named `llmlingua2vi_force_*` in the pipeline:
`force_tokens` semantics, i.e. plain string matching of the P word list (multi-syllable
entries matched as syllable sequences) plus digits, without POS/NER disambiguation
(DATN §3.4). `protect=force_ctx` uses the disambiguated, NER-aware P at syllable level;
it is an extra ablation that separates "word unit" from "disambiguation" in H2.
"""
from __future__ import annotations

from . import baselines as B
from .compressor import Context, ScoredCompressor, SelectionConfig


def parse_spec(spec: str) -> tuple[str, dict[str, str]]:
    name, _, rest = spec.partition(":")
    args = dict(item.split("=", 1) for item in rest.split(",") if item) if rest else {}
    return name, args


class TranslateThenCompress:
    def __init__(self, inner):
        self.inner, self.name = inner, f"translate_{inner.name}"

    def compress(self, context: Context, budget, counter):
        if not context.text_en:
            raise ValueError("translate-then-compress needs the parallel English context")
        # `budget` was computed from the reader-facing source context. The parallel
        # English text changes only what the inner compressor sees; it must not get a
        # separate budget and thereby gain an unfair length allowance.
        return self.inner.compress(Context(text=context.text_en), budget, counter)


def build_compressor(spec: str, cache: dict | None = None):
    """`cache` keeps loaded LMs/scorers alive across specs that share them."""
    cache = {} if cache is None else cache
    name, a = parse_spec(spec)

    def lm(model_name):
        key = ("lm", model_name)
        if key not in cache:
            cache[key] = B.CausalLM(model_name)
        return cache[key]

    simple = {"none": B.NoCompression, "no_context": B.NoContext, "lead": B.LeadK,
              "truncation": B.Truncation, "stopword": B.StopwordRemoval,
              "lexical": B.LexicalRules, "tfidf_sent": B.TfidfSentences,
              "probe_negation": B.NegationProbe}
    if name in simple:
        return simple[name]()
    if name == "random":
        return B.RandomWords(int(a.get("seed", 0)))
    if name == "ppl_sent":
        return B.PerplexitySentences(lm(a.get("lm", "Qwen/Qwen2.5-1.5B")))
    if name == "selective_context":
        return B.SelectiveContext(lm(a.get("lm", "Qwen/Qwen2.5-1.5B")))
    if name == "llmlingua2":
        return B.LLMLinguaOfficial("llmlingua2")
    if name == "llmlingua2_force":
        return B.LLMLinguaOfficial("llmlingua2", force_from_context=True)
    if name in {"llmlingua", "longllmlingua"}:
        return B.LLMLinguaOfficial(name, lm_name=a.get("lm", "Qwen/Qwen2.5-1.5B"))
    if name == "translate":
        return TranslateThenCompress(build_compressor(a.get("inner", "llmlingua2"), cache))
    if name == "precomputed":
        return B.Precomputed(a["path"], a.get("name", "precomputed"), trim=a.get("trim", "0") == "1")
    if name in {"scored", "lexprior"}:
        if name == "scored":
            from .model import SyllableScorer
            key = ("scorer", a["model"])
            if key not in cache:
                cache[key] = SyllableScorer(a["model"])
            scorer = cache[key]
        else:
            from .data import read_jsonl
            from .model import LexicalPriorScorer
            records = read_jsonl(a["train"])
            if "exclude" in a:  # e.g. runs/viword_s0/dev_ids.jsonl: fit on the encoder's train split only
                held_out = {r["id"] for r in read_jsonl(a["exclude"])}
                records = [r for r in records if r["id"] not in held_out]
                if not records:
                    raise ValueError(f"lexprior: no training records left after excluding {a['exclude']}")
            scorer = LexicalPriorScorer.fit(records)
        config = SelectionConfig(
            unit=a.get("unit", "word"),
            protect=a.get("protect", "none"),
            delta=float(a.get("delta", 0.2)),
            tiers=tuple(a.get("tiers", "T1+T2+T3").split("+")),
            alpha=float(a.get("alpha", 0.0)),
            cost=a.get("cost", "tokens"),
            exact=a.get("exact", "0") == "1",
        )
        return ScoredCompressor(scorer, config, a.get("name", spec))
    raise ValueError(f"unknown method spec: {spec}")
