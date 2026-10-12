"""Validate provenance and integrity of cached segmented JSONL files.

The production pipeline requires every ``data/segmented/*.jsonl`` file to have
a matching ``.jsonl.meta.json`` sidecar whose segmenter is VnCoreNLP and whose
SHA-256 matches the JSONL file.
"""
import argparse
from pathlib import Path

from viword.segment_meta import SEGMENTER_INFO, verify_metadata


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--directory", default="data/segmented")
    ap.add_argument("--segmenter", default="vncorenlp", choices=sorted(SEGMENTER_INFO))
    args = ap.parse_args()

    files = sorted(Path(args.directory).glob("*.jsonl"))
    if not files:
        print(f"no segmented JSONL files found in {args.directory}")
        return 1

    errors = [error for path in files for error in verify_metadata(path, args.segmenter)]
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"OK: {len(files)} segmented files use {args.segmenter} with matching metadata")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
