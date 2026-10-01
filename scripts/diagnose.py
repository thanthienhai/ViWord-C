"""Week-1 diagnostics (DATN §4.0), no reader needed.

For every method and ratio: CBR next to its chance level, retention of each protection
tier, actual keep rate and budget adherence. Also the token-cost profile of each target
tokenizer (P3, gate G5) and, for ViNLI, how many premises contain a negation (gate G2).

Usage:
  python scripts/diagnose.py --task vinli --data data/tasks/vinli_test.jsonl \
      --segmented data/segmented/vinli_test.jsonl --limit 500 \
      --budget-tokenizer Qwen/Qwen2.5-7B-Instruct \
      --cost-tokenizers Qwen/Qwen2.5-7B-Instruct meta-llama/Llama-3.1-8B-Instruct \
          Viet-Mistral/Vistral-7B-Chat SeaLLMs/SeaLLMs-v3-7B-Chat google/gemma-2-9b-it \
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
from viword.methods import build_compressor
from viword.segment import align_compressed, flatten, syllables_of
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
    ap.add_argument("--rr-model", default=None, help="fill-mask model for RR, e.g. xlm-roberta-large")
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
    mlm = pipeline("fill-mask", model=args.rr_model) if args.rr_model else None
    report = {"task": args.task, "n_examples": len(contexts), "methods": {}}

    cache = {}
    for spec in args.methods:
        compressor = build_compressor(spec, cache)
        report["methods"][compressor.name] = {}
        for ratio in args.ratios:
            cbr_c, ret_c, kept, total, over_budget, rr = Counter(), Counter(), 0, 0, 0, []
            for ctx in tqdm(contexts, desc=f"{compressor.name} @ {ratio:.2f}"):
                words = flatten(ctx.doc)
                ctx.ratio = ratio
                budget = round(ratio * counter.count(ctx.text))
                compressed = compressor.compress(ctx, budget, counter)
                mask, _ = align_compressed(words, compressed)
                cbr_c += D.cbr_counts(words, mask)
                ret_c += D.retention_counts(words, mask)
                kept, total = kept + sum(mask), total + len(syllables_of(words))
                over_budget += counter.count(compressed) > budget * 1.05
                if mlm is not None:
                    rr.append(D.recovery_rate(words, mask, mlm))
            keep_rate = kept / total
            report["methods"][compressor.name][f"{ratio:.3f}"] = {
                "keep_rate": keep_rate,
                "cbr": D.cbr(cbr_c),
                "chance_cbr_2syl": D.chance_cbr(keep_rate) if 0 < keep_rate < 1 else None,
                "retention": D.retention(ret_c),
                "share_over_budget_5pct": over_budget / len(contexts),
                "recovery_rate": mean_ignoring_nan(rr),
            }

    all_words = [w for ctx in contexts for w in flatten(ctx.doc)]
    report["token_cost"] = {name: D.token_cost_profile(all_words, TokenCounter(AutoTokenizer.from_pretrained(name)))
                            for name in args.cost_tokenizers}
    if args.task == "vinli":
        with_neg = sum(D.has_negation(flatten(ctx.doc)) for ctx in contexts)
        report["premises_with_negation"] = {"count": with_neg, "share": with_neg / len(contexts)}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n| method | ratio | keep | CBR | chance CBR | T1 kept | T2 kept | T3 kept | all kept |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, by_ratio in report["methods"].items():
        for ratio, m in by_ratio.items():
            r = m["retention"]
            chance = f"{m['chance_cbr_2syl']:.2f}" if m["chance_cbr_2syl"] is not None else "-"
            print(f"| {name} | {ratio} | {m['keep_rate']:.2f} | {m['cbr']['all']:.3f} | {chance} | "
                  f"{r['T1']:.2f} | {r['T2']:.2f} | {r['T3']:.2f} | {r['all']:.2f} |")
    print(f"\nfull report: {args.out}")


if __name__ == "__main__":
    main()
