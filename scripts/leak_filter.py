"""Drop distillation paragraphs that share a 13-token n-gram with any evaluation context
(DATN §3.2). Run before querying the teacher, so no API calls are spent on them.

Usage:
  python scripts/leak_filter.py --input data/distill/paragraphs.jsonl \
      --eval-files data/tasks/*.jsonl --output data/distill/paragraphs_clean.jsonl
"""
import argparse
from collections import Counter

from viword.data import read_jsonl, write_jsonl
from viword.distill import LeakIndex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--eval-files", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    texts = [r.get("context") or r.get("premise") or "" for path in args.eval_files for r in read_jsonl(path)]
    index = LeakIndex(texts)
    rows = read_jsonl(args.input)
    clean = [r for r in rows if not index.leaks(r["text"])]
    dropped = Counter(r.get("source", "?") for r in rows if index.leaks(r["text"]))
    write_jsonl(args.output, clean)
    print(f"{len(texts)} evaluation contexts indexed; kept {len(clean)} of {len(rows)} paragraphs; "
          f"dropped by source: {dict(dropped)}")


if __name__ == "__main__":
    main()
