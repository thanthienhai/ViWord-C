"""Train one encoder (DATN §3.3). Run once per seed and per label unit.

Usage:
  # ViWord-C (soft word labels)
  python scripts/train.py --distilled data/distill/distilled.jsonl --label-unit word \
      --out runs/viword_s0 --seed 0
  # LLMLingua-2-vi control (syllable labels)
  python scripts/train.py --distilled data/distill/distilled.jsonl --label-unit syllable \
      --out runs/llmlingua2vi_s0 --seed 0
  # controls of the label-unit effect (cells iii-rand and iv-wordloss, viword/train.py)
  python scripts/train.py ... --label-unit syllable_softrand --out runs/softrand_s0 --seed 0
  python scripts/train.py ... --label-unit word_pooled --out runs/wordloss_s0 --seed 0

The train/dev split is made by paragraph id, so both compression rates of one paragraph
land in the same split. The split is fixed (seed 0) and independent of --seed.
"""
import argparse
import os
import random

from viword.data import read_jsonl, write_jsonl
from viword.train import LABEL_UNITS, train


def split_by_paragraph(records, dev_paragraphs):
    ids = sorted({r["id"] for r in records})
    random.Random(0).shuffle(ids)
    dev_ids = set(ids[:dev_paragraphs])
    return [r for r in records if r["id"] not in dev_ids], [r for r in records if r["id"] in dev_ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distilled", required=True)
    ap.add_argument("--label-unit", required=True, choices=list(LABEL_UNITS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-name", default="xlm-roberta-large")
    ap.add_argument("--dev-paragraphs", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ids-from", default=None,
                    help="train only on the paragraph ids of this JSONL file (e.g. the distilled labels of a "
                         "teacher variant, for a checkpoint of the main teacher on the same paragraphs)")
    ap.add_argument("--limit-paragraphs", type=int, default=None,
                    help="train on the first N paragraph ids only (small descriptive checkpoints: "
                         "neutral-prompt or open-teacher labels)")
    args = ap.parse_args()

    records = read_jsonl(args.distilled)
    if args.ids_from:
        wanted = {r["id"] for r in read_jsonl(args.ids_from)}
        records = [r for r in records if r["id"] in wanted]
    if args.limit_paragraphs:
        keep = set(sorted({r["id"] for r in records})[:args.limit_paragraphs])
        records = [r for r in records if r["id"] in keep]
    train_records, dev_records = split_by_paragraph(records, args.dev_paragraphs)
    os.makedirs(args.out, exist_ok=True)
    write_jsonl(os.path.join(args.out, "dev_ids.jsonl"), [{"id": i} for i in sorted({r["id"] for r in dev_records})])
    print(f"train records: {len(train_records)}  dev records: {len(dev_records)}")
    train(train_records, dev_records, args.out, args.label_unit, model_name=args.model_name,
          epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, seed=args.seed)


if __name__ == "__main__":
    main()
