"""Write blind annotation sheets for the P detector (DATN §3.4) and the negation probe (G6).

The previous label files were filled from the detector's own output, so precision and
recall were 1.000 by construction. These sheets never contain a detector prediction:

* protect sheet: 100 ViNLI dev premises (one per premise) and 100 single sentences from
  VietNews dev articles, drawn uniformly at random (seeded), so sentences without any
  protected word are included and recall is measurable. Each row lists the segmented
  words with index, POS and NER; the annotator fills ``gold_tiers`` ({"index": "T1"|"T2"|"T3"},
  ``{}`` if the sentence has no protected word) and ``annotator``.
* probe sheet: 50 distinct ViNLI premises (dev + train; dev alone has too few with a
  negation) that the probe edits and 50 distinct premises it leaves unchanged, shuffled together, without saying which is which. The
  annotator fills ``gold_negation`` (true/false: does the premise contain a real
  negation?) and ``annotator``.

Annotation guide: data/README.md, section "Gán nhãn bộ phát hiện P và probe phủ định".

Usage:
  python scripts/make_annotation_sheets.py --segmented-dir data/segmented --tasks-dir data/tasks \
      --protect-out data/annot/protect_sheet.jsonl --probe-out data/annot/probe_sheet.jsonl
"""
import argparse
import json
import random
from pathlib import Path

from viword.data import read_jsonl
from viword.diagnose import has_negation
from viword.segment import doc_from_json, flatten

SHEET = "blind-v1"


def _words(doc_words) -> list[list]:
    return [[i, w.text, w.pos, w.ner] for i, w in enumerate(doc_words)]


def protect_sheet(segmented_dir: str, tasks_dir: str, n_per_source: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    premises = {}
    clusters = {r["id"]: r["cluster"] for r in read_jsonl(f"{tasks_dir}/vinli_dev.jsonl")}
    for r in read_jsonl(f"{segmented_dir}/vinli_dev.jsonl"):
        premises.setdefault(clusters[r["id"]], r)
    for cluster in rng.sample(sorted(premises), n_per_source):
        r = premises[cluster]
        rows.append({"id": r["id"], "source": "vinli_dev", "sentence": None, "sheet": SHEET,
                     "words": _words(flatten(doc_from_json(r["doc"]))), "gold_tiers": None, "annotator": ""})
    articles = read_jsonl(f"{segmented_dir}/vietnews_dev.jsonl")
    for r in rng.sample(articles, n_per_source):
        doc = doc_from_json(r["doc"])
        k = rng.randrange(len(doc))
        rows.append({"id": r["id"], "source": "vietnews_dev", "sentence": k, "sheet": SHEET,
                     "words": _words(doc[k]), "gold_tiers": None, "annotator": ""})
    return rows


def probe_sheet(segmented_dir: str, tasks_dir: str, n_per_group: int, seed: int,
                sources=("vinli_dev", "vinli_train")) -> list[dict]:
    """ViNLI dev alone has only ~42 premises with a negation, fewer than the 50 that G6
    needs, so premises are pooled from dev and train (neither is a test split)."""
    rng = random.Random(seed)
    by_cluster = {}
    for source in sources:
        tasks = {r["id"]: r for r in read_jsonl(f"{tasks_dir}/{source}.jsonl")}
        for r in read_jsonl(f"{segmented_dir}/{source}.jsonl"):
            task = tasks[r["id"]]
            by_cluster.setdefault(task["cluster"], (source, r, task["premise"]))
    edited = sorted(c for c, (_, r, _) in by_cluster.items() if has_negation(flatten(doc_from_json(r["doc"]))))
    edited_set = set(edited)
    kept = sorted(c for c in by_cluster if c not in edited_set)
    if len(edited) < n_per_group or len(kept) < n_per_group:
        raise ValueError(f"not enough distinct premises: {len(edited)} edited, {len(kept)} unchanged")
    chosen = rng.sample(edited, n_per_group) + rng.sample(kept, n_per_group)
    rng.shuffle(chosen)
    return [{"id": by_cluster[c][1]["id"], "source": by_cluster[c][0], "cluster": c, "sheet": SHEET,
             "premise": by_cluster[c][2], "gold_negation": None, "annotator": ""}
            for c in chosen]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segmented-dir", default="data/segmented")
    ap.add_argument("--tasks-dir", default="data/tasks")
    ap.add_argument("--protect-out", default="data/annot/protect_sheet.jsonl")
    ap.add_argument("--probe-out", default="data/annot/probe_sheet.jsonl")
    ap.add_argument("--per-source", type=int, default=100)
    ap.add_argument("--per-group", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    for path, rows in ((args.protect_out, protect_sheet(args.segmented_dir, args.tasks_dir, args.per_source, args.seed)),
                       (args.probe_out, probe_sheet(args.segmented_dir, args.tasks_dir, args.per_group, args.seed))):
        if Path(path).exists():
            raise SystemExit(f"{path} exists; refusing to overwrite annotations")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print(f"wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
