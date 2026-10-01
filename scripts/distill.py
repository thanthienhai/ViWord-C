"""Build the distilled training labels from teacher outputs (DATN §3.2).

Inputs: the (leak-filtered) paragraphs, their segmentation, and the teacher outputs written
by scripts/teacher_compress.py. Outputs that rewrote the text (share of output words not in
the source above --max-unmatched) or did not compress (keep rate above --max-keep-rate)
are dropped. Prints the teacher's own statistics (keep rate, CBR).

Usage:
  python scripts/distill.py --paragraphs data/distill/paragraphs_clean.jsonl \
      --segmented data/segmented/paragraphs_clean.jsonl --teacher data/distill/teacher.jsonl \
      --output data/distill/distilled.jsonl
"""
import argparse
import json
from collections import Counter

from viword.data import read_jsonl, write_jsonl
from viword.distill import make_labels
from viword.segment import doc_from_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paragraphs", required=True, help="JSONL with id, text")
    ap.add_argument("--segmented", required=True, help="segment_data.py output for --paragraphs")
    ap.add_argument("--teacher", required=True, help="teacher_compress.py output")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-unmatched", type=float, default=0.1,
                    help="drop outputs where more than this share of output words is not in the source")
    ap.add_argument("--max-keep-rate", type=float, default=0.9,
                    help="drop outputs that kept more than this share of syllables (no compression)")
    args = ap.parse_args()

    texts = {r["id"]: r["text"] for r in read_jsonl(args.paragraphs)}
    docs = {r["id"]: r["doc"] for r in read_jsonl(args.segmented)}
    records, stats = [], Counter()
    for t in read_jsonl(args.teacher):
        if t["id"] not in docs:
            stats["missing_segmentation"] += 1
            continue
        labels = make_labels(doc_from_json(docs[t["id"]]), t["compressed"])
        if not t["compressed"] or labels["unmatched_ratio"] > args.max_unmatched:
            stats["dropped_rewritten"] += 1
            continue
        if labels["keep_rate"] > args.max_keep_rate:
            stats["dropped_not_compressed"] += 1
            continue
        stats["kept"] += 1
        stats[f"kept@{t['ratio']}"] += 1
        stats["sum_teacher_cbr"] += labels["teacher_cbr"]
        stats["sum_keep_rate"] += labels["keep_rate"]
        records.append({"id": t["id"], "rate": t["ratio"], "text": texts.get(t["id"], ""),
                        "compressed": t["compressed"], "doc": docs[t["id"]], **labels})

    write_jsonl(args.output, records)
    n = max(stats["kept"], 1)
    summary = {k: v for k, v in stats.items() if not k.startswith("sum_")}
    summary.update(mean_teacher_cbr=stats["sum_teacher_cbr"] / n, mean_keep_rate=stats["sum_keep_rate"] / n)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
