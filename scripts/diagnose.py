"""Week-1 diagnostics (DATN §4.0), no reader needed.

For every method and ratio: CBR next to its chance level, retention of each protection
tier, actual keep rate, mean ``n_tokens_compressed / budget`` and the share of examples
whose budget deviation exceeds 5%. Also the token-cost profile of each target tokenizer
(P3, gate G5) and, for ViNLI, how many premises contain a negation (gate G2).

Usage:
  python scripts/diagnose.py --task vinli --data data/tasks/vinli_test.jsonl \
      --segmented data/segmented/vinli_test.jsonl --limit 500 \
      --budget-tokenizer Qwen/Qwen2.5-7B-Instruct \
      --cost-tokenizers Qwen/Qwen2.5-7B-Instruct meta-llama/Llama-3.1-8B-Instruct \
          SeaLLMs/SeaLLM-7B-v2 SeaLLMs/SeaLLMs-v3-7B-Chat google/gemma-2-9b-it \
      --methods random lead truncation llmlingua2 llmlingua:lm=Qwen/Qwen2.5-1.5B \
          selective_context:lm=Qwen/Qwen2.5-1.5B \
          scored:model=microsoft/llmlingua-2-xlm-roberta-large-meetingbank,unit=syllable,name=llmlingua2_syl \
          scored:model=microsoft/llmlingua-2-xlm-roberta-large-meetingbank,unit=word,name=llmlingua2_wordpool \
      --out results/diagnose/vinli.json
"""
import argparse
import json
import math
import os
from collections import Counter

from tqdm import tqdm

from viword import diagnose as D
from viword.data import load_contexts, load_task
from viword.eval import budget_deviates, context_budget, tokens_per_budget
from viword.methods import build_compressor
from viword.segment import align_compressed, flatten
from viword.select import TokenCounter


def mean_ignoring_nan(values):
    values = [v for v in values if not math.isnan(v)]
    return sum(values) / len(values) if values else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--segmented", required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.5, 1 / 3, 0.2])
    ap.add_argument("--budget-tokenizer", required=True, help="tokenizer of the main reader")
    ap.add_argument("--cost-tokenizers", nargs="*", default=[])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--unique-clusters", action="store_true", help="keep one example per source document")
    ap.add_argument("--rr-model", default="xlm-roberta-large",
                    help="fill-mask model for RR (default: xlm-roberta-large)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer, pipeline

    examples = load_task(args.task, args.data)
    if args.unique_clusters:  # count every source document once (Belebele passages, ViNLI premises)
        seen = set()
        examples = [ex for ex in examples if not (ex.cluster in seen or seen.add(ex.cluster))]
    examples = examples[:args.limit]
    contexts = load_contexts(examples, args.segmented)
    counter = TokenCounter(AutoTokenizer.from_pretrained(args.budget_tokenizer))
    # load every target tokenizer of the G5 profile now: a gated or missing repo must fail
    # before the GPU work, not after it
    try:
        cost_counters = {name: TokenCounter(AutoTokenizer.from_pretrained(name)) for name in args.cost_tokenizers}
    except OSError as exc:
        raise SystemExit(f"diagnose: cannot load a --cost-tokenizers entry ({exc}); gated repos need an HF "
                         "token with access") from exc
    mlm = pipeline("fill-mask", model=args.rr_model) if args.rr_model else None
    multi_syllable_lengths = [len(w.syllables) for ctx in contexts for w in flatten(ctx.doc)
                              if not w.is_punct and len(w.syllables) > 1]
    report = {"task": args.task, "n_examples": len(contexts),
              "budget_tokenizer": args.budget_tokenizer,
              "multi_syllable_length_distribution": dict(Counter(multi_syllable_lengths)),
              "methods": {}}

    cache = {}
    for spec in args.methods:
        compressor = build_compressor(spec, cache)
        report["methods"][compressor.name] = {}
        for ratio in args.ratios:
            cbr_c, ret_c, kept, total, budget_ratios, budget_deviations, rr = (
                Counter(), Counter(), 0, 0, [], 0, []
            )
            for ctx in tqdm(contexts, desc=f"{compressor.name} @ {ratio:.2f}"):
                words = flatten(ctx.doc)
                ctx.ratio = ratio
                _, budget = context_budget(ctx.text, ratio, counter)
                compressed = compressor.compress(ctx, budget, counter)
                actual_ratio = tokens_per_budget(counter.count(compressed), budget)
                if actual_ratio is not None:
                    budget_ratios.append(actual_ratio)
                    budget_deviations += budget_deviates(actual_ratio)
                mask, _ = align_compressed(words, compressed)
                cbr_c += D.cbr_counts(words, mask)
                ret_c += D.retention_counts(words, mask)
                # keep rate over word syllables only: punctuation is not a syllable, and the
                # chance CBR it feeds assumes the rate at which syllables are kept
                content = [keep for w in words for keep in [not w.is_punct] * len(w.syllables)]
                kept += sum(m for m, c in zip(mask, content) if c)
                total += sum(content)
                if mlm is not None:
                    rr.append(D.recovery_rate(words, mask, mlm))
            keep_rate = kept / total
            mean_budget_ratio = mean_ignoring_nan(budget_ratios)
            report["methods"][compressor.name][f"{ratio:.3f}"] = {
                "family": D.compressor_family(compressor.name),
                "keep_rate": keep_rate,
                "cbr": D.cbr(cbr_c),
                "chance_cbr": D.chance_cbr_distribution(keep_rate, multi_syllable_lengths),
                "chance_cbr_2syl": D.chance_cbr(keep_rate) if 0 < keep_rate < 1 else None,
                "retention": D.retention(ret_c),
                "mean_tokens_per_budget": mean_budget_ratio,
                "share_budget_deviation_5pct": budget_deviations / len(budget_ratios) if budget_ratios else None,
                "budget_deviation_5pct": budget_deviates(mean_budget_ratio),
                "recovery_rate": mean_ignoring_nan(rr),
            }

    all_words = [w for ctx in contexts for w in flatten(ctx.doc)]
    report["token_cost"] = {name: D.token_cost_profile(all_words, c) for name, c in cost_counters.items()}
    if args.task == "vinli":
        with_neg = sum(D.has_negation(flatten(ctx.doc)) for ctx in contexts)
        report["premises_with_negation"] = {"count": with_neg, "share": with_neg / len(contexts)}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n| family | method | ratio | keep | CBR | chance CBR (length mix) | T1 kept | T2 kept | T3 kept | "
          "all kept | tok/budget | >5% |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    ordered = sorted(report["methods"].items(),
                     key=lambda item: (D.FAMILY_ORDER.index(D.compressor_family(item[0])), item[0]))
    for name, by_ratio in ordered:
        for ratio, m in by_ratio.items():
            r = m["retention"]
            chance_value = m.get("chance_cbr")
            chance = f"{chance_value:.2f}" if chance_value is not None and not math.isnan(chance_value) else "-"
            adherence = f"{m['mean_tokens_per_budget']:.3f}" if m["mean_tokens_per_budget"] is not None else "-"
            flag = "⚠" if m["budget_deviation_5pct"] else ""
            print(f"| {D.compressor_family(name)} | {name} | {ratio} | {m['keep_rate']:.2f} | "
                  f"{m['cbr']['all']:.3f} | {chance} | "
                  f"{r['T1']:.2f} | {r['T2']:.2f} | {r['T3']:.2f} | {r['all']:.2f} | {adherence} | {flag} |")
    print(f"\nfull report: {args.out}")


if __name__ == "__main__":
    main()
