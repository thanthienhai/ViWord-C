"""Segment the compressible field of a JSONL file once and cache the result.

Usage:
  python scripts/segment_data.py --input data/tasks/vinli_test.jsonl --field premise \
      --output data/segmented/vinli_test.jsonl
  python scripts/segment_data.py --input data/distill/paragraphs.jsonl --field text \
      --output data/segmented/paragraphs.jsonl

Output lines: {"id": ..., "doc": [[[word, pos, ner], ...], ...]}
"""
import argparse

from tqdm import tqdm

from viword.data import read_jsonl, write_jsonl
from viword.segment import doc_to_json, get_segmenter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--field", default="context", help="context | premise | text | context_en")
    ap.add_argument("--segmenter", default="vncorenlp", choices=["vncorenlp", "underthesea", "pyvi"])
    args = ap.parse_args()

    segmenter = get_segmenter(args.segmenter)
    rows = [{"id": r["id"], "doc": doc_to_json(segmenter.segment(r[args.field]))}
            for r in tqdm(read_jsonl(args.input))]
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} documents to {args.output}")


if __name__ == "__main__":
    main()
