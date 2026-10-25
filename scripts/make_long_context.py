"""Long-context variant of a multiple-choice set (item 10), in the spirit of Lost in Compression.

Each example keeps its question, choices and answer; its passage is placed at a random
position among distractor passages of *other* source documents (other clusters) until the
context reaches about --target-syllables syllables (≈ 3–4k reader tokens at 2,500). This is
the setting where token cost matters and where strong compression can fall to the
no-context level. The distractors come from the same file, so no new text is introduced.

The rows are new (id + "#long"), but they embed the source passages, so a test source is
only built and evaluated in the eval stage, after its own checks, and rows readers scored
during development are dropped first (--exclude), as for the main test run.

Usage:
  python scripts/make_long_context.py --input data/tasks/belebele.jsonl --exclude config/test_exclusions.json \
      --target-syllables 2500 --limit 300 --out data/tasks/belebele_long.jsonl
"""
import argparse
import random
from pathlib import Path

from viword.data import read_jsonl, write_jsonl
from viword.heldout import TEST_FILES, excluded_ids, load_exclusions


def long_rows(rows: list[dict], target_syllables: int, limit: int | None, seed: int = 0) -> list[dict]:
    passages = {}
    for r in rows:
        passages.setdefault(r["cluster"], r["context"])
    clusters = sorted(passages)
    rng = random.Random(seed)
    out = []
    for r in rows[:limit]:
        pool = [c for c in clusters if c != r["cluster"]]
        rng.shuffle(pool)
        parts, n = [r["context"]], len(r["context"].split())
        for c in pool:
            if n >= target_syllables:
                break
            parts.append(passages[c])
            n += len(passages[c].split())
        gold = parts.pop(0)
        position = rng.randrange(len(parts) + 1)
        parts.insert(position, gold)
        row = {k: v for k, v in r.items() if k != "context_en"}  # no parallel English for the long variant
        row.update({"id": f"{r['id']}#long", "context": "\n\n".join(parts),
                    "gold_position": position, "n_passages": len(parts)})
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="a multiple-choice task file (belebele or vimmrc format)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-syllables", type=int, default=2500)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", default=None, help="config/test_exclusions.json, for a test file")
    args = ap.parse_args()
    rows = read_jsonl(args.input)
    if args.exclude:
        skip = excluded_ids(load_exclusions(args.exclude), args.input)
        rows = [r for r in rows if r["id"] not in skip]
    out = long_rows(rows, args.target_syllables, args.limit, args.seed)
    stem = Path(args.input).name.removesuffix(".jsonl")
    if stem in TEST_FILES:  # evaluate.py refuses these rows outside the eval stage (viword/heldout.py)
        for row in out:
            row["derived_from"] = stem
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, out)
    mean = sum(len(r["context"].split()) for r in out) / max(len(out), 1)
    print(f"wrote {len(out)} long-context rows to {args.out} (mean {mean:.0f} syllables)")


if __name__ == "__main__":
    main()
