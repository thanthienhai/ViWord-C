"""Quality review of teacher compressions (gate G4 inputs, and distillation outputs).

Per ratio: requested vs achieved keep rate (syllables and target-LLM tokens), rewriting
(share of output words not in the source), outputs that did not compress, the teacher's
own CBR, and retention of the protected tiers (DATN §2.2), plus a few random examples.

Usage:
  python scripts/review_teacher.py --teacher results/teacher_dev/vinli_dev.jsonl \
      --data data/tasks/vinli_dev.jsonl --field premise --segmented data/segmented/vinli_dev.jsonl
"""
import argparse
import random
from collections import Counter, defaultdict

import numpy as np

from viword import diagnose as D
from viword.data import read_jsonl
from viword.segment import align_compressed, doc_from_json, flatten
from viword.select import TokenCounter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--field", default="context")
    ap.add_argument("--segmented", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max-unmatched", type=float, default=0.1)
    ap.add_argument("--examples", type=int, default=3)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    counter = TokenCounter(AutoTokenizer.from_pretrained(args.tokenizer))
    texts = {r["id"]: r[args.field] for r in read_jsonl(args.data)}
    docs = {r["id"]: doc_from_json(r["doc"]) for r in read_jsonl(args.segmented)}
    rows = read_jsonl(args.teacher)
    print(f"{args.teacher}: {len(rows)} outputs, {len({texts[r['id']] for r in rows})} unique source texts")

    by_ratio = defaultdict(list)
    for r in rows:
        by_ratio[r["ratio"]].append(r)

    print("\n| ratio | n | empty | rewritten | not compressed | keep (syll) | keep (tokens) | teacher CBR "
          "| T1 kept | T2 kept | T3 kept | all kept | negation kept |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for ratio in sorted(by_ratio, reverse=True):
        group = by_ratio[ratio]
        cbr_c, ret_c, neg_kept, neg_total = Counter(), Counter(), 0, 0
        keep_syl, keep_tok, unmatched, empty = [], [], [], 0
        for r in group:
            words = flatten(docs[r["id"]])
            if not r["compressed"].strip():
                empty += 1
                continue
            mask, u = align_compressed(words, r["compressed"])
            unmatched.append(u)
            keep_syl.append(np.mean(mask))
            keep_tok.append(counter.count(r["compressed"]) / max(counter.count(texts[r["id"]]), 1))
            cbr_c += D.cbr_counts(words, mask)
            ret_c += D.retention_counts(words, mask)
            start = 0
            for i, w in enumerate(words):
                n = len(w.syllables)
                if D.is_negation(words, i):
                    neg_total += 1
                    neg_kept += any(mask[start:start + n])
                start += n
        unmatched = np.array(unmatched)
        ret = D.retention(ret_c)
        print(f"| {ratio} | {len(group)} | {empty} | {np.mean(unmatched > args.max_unmatched):.0%} "
              f"(mean {unmatched.mean():.3f}) | {np.mean(np.array(keep_syl) > 0.9):.0%} | "
              f"{np.mean(keep_syl):.2f} ± {np.std(keep_syl):.2f} | {np.mean(keep_tok):.2f} | "
              f"{D.cbr(cbr_c)['all']:.1%} | {ret['T1']:.2f} | {ret['T2']:.2f} | {ret['T3']:.2f} | "
              f"{ret['all']:.2f} | {neg_kept}/{neg_total} |")

    rng = random.Random(0)
    for r in rng.sample(rows, min(args.examples, len(rows))):
        print(f"\n--- {r['id']} @ {r['ratio']}\nSOURCE: {texts[r['id']][:400]}\nTEACHER: {r['compressed'][:400]}")


if __name__ == "__main__":
    main()
