"""Segment the compressible field of a JSONL file once and cache the result.

Usage:
  python scripts/segment_data.py --input data/tasks/vinli_test.jsonl --field premise \
      --output data/segmented/vinli_test.jsonl
  python scripts/segment_data.py --input data/distill/paragraphs_clean.jsonl --field text \
      --output data/segmented/paragraphs_clean.jsonl --segmenter underthesea --workers 16

Output lines: {"id": ..., "doc": [[[word, pos, ner], ...], ...]}
Identical texts are segmented once (ViNLI has ~8 hypotheses per premise). --workers > 1
runs one segmenter per process (useful for underthesea/pyvi).
"""
import argparse
from multiprocessing import Pool

from tqdm import tqdm

from viword.data import read_jsonl, write_jsonl
from viword.segment import doc_to_json, get_segmenter

_segmenter = None


def _init(name):
    global _segmenter
    _segmenter = get_segmenter(name)


def _segment(text):
    return doc_to_json(_segmenter.segment(text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--field", default="context", help="context | premise | text | context_en")
    ap.add_argument("--segmenter", default="vncorenlp", choices=["vncorenlp", "underthesea", "pyvi"])
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    rows = read_jsonl(args.input)
    texts = list(dict.fromkeys(r[args.field] for r in rows))  # unique texts, in order
    if args.workers > 1:
        with Pool(args.workers, initializer=_init, initargs=(args.segmenter,)) as pool:
            docs = list(tqdm(pool.imap(_segment, texts, chunksize=4), total=len(texts)))
    else:
        _init(args.segmenter)
        docs = [_segment(t) for t in tqdm(texts)]
    by_text = dict(zip(texts, docs))
    write_jsonl(args.output, [{"id": r["id"], "doc": by_text[r[args.field]]} for r in rows])
    print(f"wrote {len(rows)} documents ({len(texts)} unique texts) to {args.output}")


if __name__ == "__main__":
    main()
