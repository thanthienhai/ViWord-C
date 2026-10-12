"""Verify legacy segmentation caches by re-segmenting a random sample with VnCoreNLP.

A sidecar whose provenance is not ``scripts/segment_data.py`` only *claims* that the
cache came from VnCoreNLP.  This script re-runs VnCoreNLP on ``--sample`` random unique
texts of each such file and compares the full (word, POS, NER) output.  When every
sampled document matches, the sidecar receives a ``verification`` block, which
``check_segmented.py`` accepts in place of fresh provenance.  Any mismatch leaves the
sidecar untouched and exits non-zero: re-run ``scripts/segment_data.py`` for that file.

Usage:
  python scripts/verify_segmentation.py --directory data/segmented --sample 30
"""
import argparse
import datetime
import json
import random
from pathlib import Path

from viword.data import read_jsonl
from viword.segment import doc_to_json, get_segmenter
from viword.segment_meta import metadata_path, needs_verification, verify_metadata


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--directory", default="data/segmented")
    ap.add_argument("--sample", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    segmenter = get_segmenter("vncorenlp")
    failed = 0
    for path in sorted(Path(args.directory).glob("*.jsonl")):
        sidecar = metadata_path(path)
        errors = verify_metadata(path, "vncorenlp", require_verified=False)
        if errors:
            print("\n".join(f"ERROR: {e}" for e in errors))
            failed += 1
            continue
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if not needs_verification(metadata):
            continue
        source, field = metadata.get("input"), metadata.get("field")
        if not source or not field or not Path(source).is_file():
            print(f"ERROR: {sidecar}: cannot verify without an existing input/field")
            failed += 1
            continue
        texts = {r["id"]: r[field] for r in read_jsonl(source)}
        cached = {r["id"]: r["doc"] for r in read_jsonl(str(path))}
        unique = {}
        for doc_id, text in texts.items():
            if doc_id in cached:
                unique.setdefault(text, doc_id)
        sample = random.Random(args.seed).sample(sorted(unique.values()), min(args.sample, len(unique)))
        mismatched = [i for i in sample if doc_to_json(segmenter.segment(texts[i])) != cached[i]]
        if mismatched:
            print(f"FAIL: {path}: {len(mismatched)}/{len(sample)} sampled documents differ, "
                  f"e.g. {mismatched[:3]}; re-segment this file with scripts/segment_data.py")
            failed += 1
            continue
        metadata["verification"] = {
            "method": "vncorenlp re-segmentation of a random sample (scripts/verify_segmentation.py)",
            "sample_size": len(sample),
            "matched": len(sample),
            "seed": args.seed,
            "date": datetime.date.today().isoformat(),
        }
        sidecar.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"VERIFIED: {path}: {len(sample)}/{len(sample)} sampled documents identical")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
