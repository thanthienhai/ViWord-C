"""Build the distilled training data (DATN §3.2).

The teacher is reached through an OpenAI-compatible endpoint, e.g.
  vllm serve Qwen/Qwen2.5-72B-Instruct-AWQ --port 8000

Usage:
  python scripts/distill.py --paragraphs data/distill/paragraphs.jsonl \
      --segmented data/segmented/paragraphs.jsonl \
      --eval-files data/tasks/*.jsonl --output data/distill/distilled.jsonl \
      --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-72B-Instruct-AWQ

Teacher sanity check (gate G4) on dev paragraphs: add --limit 200 and --save-compressions
to also write {id, ratio, compressed} for the `precomputed:` evaluation method.
"""
import argparse
import json
from collections import Counter

from tqdm import tqdm

from viword.data import read_jsonl, write_jsonl
from viword.distill import LeakIndex, make_labels, teacher_compress
from viword.segment import doc_from_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paragraphs", required=True, help="JSONL with id, text")
    ap.add_argument("--segmented", required=True, help="output of segment_data.py for --paragraphs")
    ap.add_argument("--eval-files", nargs="+", required=True, help="normalized task files for the leak check")
    ap.add_argument("--output", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--model", required=True)
    ap.add_argument("--rates", type=float, nargs="+", default=[0.5, 0.25])
    ap.add_argument("--max-unmatched", type=float, default=0.1,
                    help="drop outputs where more than this share of tokens was rewritten")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-compressions", default=None)
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    eval_texts = []
    for path in args.eval_files:
        for r in read_jsonl(path):
            eval_texts.append(r.get("context") or r.get("premise") or "")
    leak_index = LeakIndex(eval_texts)
    docs = {r["id"]: r["doc"] for r in read_jsonl(args.segmented)}

    records, compressions, stats = [], [], Counter()
    for p in tqdm(read_jsonl(args.paragraphs)[:args.limit]):
        if leak_index.leaks(p["text"]):
            stats["dropped_leak"] += 1
            continue
        for rate in args.rates:
            compressed = teacher_compress(client, args.model, p["text"], rate)
            labels = make_labels(doc_from_json(docs[p["id"]]), compressed)
            compressions.append({"id": p["id"], "ratio": rate, "compressed": compressed})
            if labels["unmatched_ratio"] > args.max_unmatched:
                stats["dropped_rewritten"] += 1
                continue
            stats["kept"] += 1
            stats["sum_teacher_cbr"] += labels["teacher_cbr"]
            stats["sum_keep_rate"] += labels["keep_rate"]
            records.append({"id": p["id"], "rate": rate, "text": p["text"], "compressed": compressed,
                            "doc": docs[p["id"]], **labels})

    write_jsonl(args.output, records)
    if args.save_compressions:
        write_jsonl(args.save_compressions, compressions)
    n = max(stats["kept"], 1)
    summary = {"kept": stats["kept"], "dropped_leak": stats["dropped_leak"],
               "dropped_rewritten": stats["dropped_rewritten"],
               "mean_teacher_cbr": stats["sum_teacher_cbr"] / n,
               "mean_keep_rate": stats["sum_keep_rate"] / n}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
