"""Minimal pairs for NFR (item 9): ViNLI-format pairs whose label depends only on the operator.

From every premise with a real negation (`protect.is_negation`), two pairs are built:

  hypothesis = the premise itself                   -> entailment
  hypothesis = the premise with its negation removed -> contradiction

The hypothesis is never compressed, so a compressor that drops the negation from the premise
turns the first pair into an apparent contradiction and the second into an apparent
entailment: the flip is caused by the operator alone, not by general information loss. Only
premises with exactly one negation are used, so removing it flips the polarity once.

Premises come from ViNLI dev and train by default (no test row is shown to a reader). Every
pair has `needs_review: true`: a human checks that removing the word gives a fluent
contradiction ("chưa" → "" can need "đã") and sets `needs_review: false` or drops the row;
`--reviewed-only` in a later step keeps the checked ones.

Usage:
  python scripts/make_minimal_pairs.py --sources vinli_dev vinli_train --n 300 \
      --out data/tasks/vinli_minpairs.jsonl
  python scripts/segment_data.py --input data/tasks/vinli_minpairs.jsonl --field premise \
      --output data/segmented/vinli_minpairs.jsonl
  python scripts/evaluate.py --task vinli --data data/tasks/vinli_minpairs.jsonl ...
"""
import argparse
import random
from pathlib import Path

from viword.data import read_jsonl, write_jsonl
from viword.protect import is_negation
from viword.segment import doc_from_json, flatten, render_all


def minimal_pairs(task_rows: list[dict], segmented: dict[str, list], n: int, seed: int = 0) -> list[dict]:
    by_cluster = {}
    for r in task_rows:
        if r["id"] in segmented:
            by_cluster.setdefault(r["cluster"], r)
    candidates = []
    for cluster, r in sorted(by_cluster.items()):
        words = flatten(doc_from_json(segmented[r["id"]]))
        negations = [i for i in range(len(words)) if is_negation(words, i)]
        if len(negations) != 1:
            continue
        positive = render_all([w for i, w in enumerate(words) if i != negations[0]])
        candidates.append((cluster, r["premise"], positive, words[negations[0]].text))
    random.Random(seed).shuffle(candidates)
    rows = []
    for cluster, premise, positive, negation in candidates[:n]:
        base = f"minpair:{cluster}"
        rows.append({"id": f"{base}:e", "cluster": base, "premise": premise, "hypothesis": premise,
                     "label": "entailment", "operator": negation, "needs_review": True})
        rows.append({"id": f"{base}:c", "cluster": base, "premise": premise, "hypothesis": positive,
                     "label": "contradiction", "operator": negation, "needs_review": True})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", default="data/tasks")
    ap.add_argument("--segmented-dir", default="data/segmented")
    ap.add_argument("--sources", nargs="+", default=["vinli_dev", "vinli_train"])
    ap.add_argument("--n", type=int, default=300, help="premises (two pairs each)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reviewed-only", default=None, metavar="REVIEWED_FILE",
                    help="instead of building: copy the rows of REVIEWED_FILE with needs_review false to --out")
    args = ap.parse_args()
    if args.reviewed_only:
        rows = [r for r in read_jsonl(args.reviewed_only) if r.get("needs_review") is False]
    else:
        rows_all, segmented = [], {}
        for source in args.sources:
            rows_all += read_jsonl(f"{args.tasks_dir}/{source}.jsonl")
            segmented.update({r["id"]: r["doc"] for r in read_jsonl(f"{args.segmented_dir}/{source}.jsonl")})
        rows = minimal_pairs(rows_all, segmented, args.n, args.seed)
    if Path(args.out).exists() and not args.reviewed_only:
        raise SystemExit(f"{args.out} exists; refusing to overwrite a sheet that may be under review")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, rows)
    print(f"wrote {len(rows)} pairs to {args.out}")


if __name__ == "__main__":
    main()
