"""Segment the compressible field of a JSONL file once and cache the result.

Usage:
  python scripts/segment_data.py --input data/tasks/vinli_test.jsonl --field premise \
      --output data/segmented/vinli_test.jsonl
  python scripts/segment_data.py --input data/distill/paragraphs_clean.jsonl --field text \
      --output data/segmented/paragraphs_clean.jsonl --segmenter vncorenlp --workers 16

Output lines: {"id": ..., "doc": [[[word, pos, ner], ...], ...]}.  A
``.jsonl.meta.json`` sidecar records the segmenter, annotators, source field and
the output SHA-256.  ``underthesea`` and ``pyvi`` remain available for the
explicit segmenter-sensitivity analysis, but are not valid inputs to the main
pipeline.
Identical texts are segmented once (ViNLI has ~8 hypotheses per premise). --workers > 1
runs one segmenter per process (useful for underthesea/pyvi).
"""
import argparse
import time
from multiprocessing import Pool

from tqdm import tqdm

from viword.data import read_jsonl, write_jsonl
from viword.segment import doc_to_json, get_segmenter
from viword.segment_meta import build_metadata, write_metadata

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
    start = time.perf_counter()
    if args.workers > 1:
        with Pool(args.workers, initializer=_init, initargs=(args.segmenter,)) as pool:
            docs = list(tqdm(pool.imap(_segment, texts, chunksize=4), total=len(texts)))
    else:
        _init(args.segmenter)
        docs = [_segment(t) for t in tqdm(texts)]
    seconds = time.perf_counter() - start  # wall time; with --workers > 1, divided over the processes
    by_text = dict(zip(texts, docs))
    write_jsonl(args.output, [{"id": r["id"], "doc": by_text[r[args.field]]} for r in rows])
    metadata = build_metadata(
        args.output,
        segmenter=args.segmenter,
        input_path=args.input,
        field=args.field,
        rows=len(rows),
        unique_texts=len(texts),
    )
    # DATN §4.4 latency: the segmenter is part of ViWord-C's end-to-end compression time
    metadata["timing"] = {"seconds": round(seconds, 3), "workers": args.workers,
                          "ms_per_text": round(1000 * seconds * args.workers / max(len(texts), 1), 3)}
    sidecar = write_metadata(args.output, metadata)
    print(f"wrote {len(rows)} documents ({len(texts)} unique texts) to {args.output}")
    print(f"wrote segmentation metadata to {sidecar}")


if __name__ == "__main__":
    main()
