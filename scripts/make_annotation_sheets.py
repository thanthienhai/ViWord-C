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

* segmentation sheet (item a): N distinct ViNLI test premises with their syllables only (no
  segmenter output, so the annotator is not anchored on VnCoreNLP). The annotator fills
  ``gold_words`` (the premise's syllables grouped into words, joined by "_", e.g.
  ["học_sinh", "đi", "học", "."]) and ``annotator``; scripts/gold_cbr.py reads it.

Re-auditing a changed detector (P detector v2) needs a *new* blind sheet: pass
``--sheet blind-v2 --exclude-sheets data/annot/protect_sheet.jsonl`` so no sentence used to
find the v1 errors is drawn again, and a larger ``--per-group`` for the probe so the 0.95
precision threshold does not hinge on one premise. A second annotator for Cohen's κ works on
a fresh copy of the same sheet (scripts/annot_agreement.py), without LLM suggestions.

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


def protect_sheet(segmented_dir: str, tasks_dir: str, n_per_source: int, seed: int,
                  sheet: str = SHEET, exclude: set | None = None) -> list[dict]:
    """`exclude`: ids already used by an earlier sheet (a ViNLI premise's cluster is excluded with it)."""
    rng = random.Random(seed)
    exclude = exclude or set()
    rows = []
    premises = {}
    clusters = {r["id"]: r["cluster"] for r in read_jsonl(f"{tasks_dir}/vinli_dev.jsonl")}
    used_clusters = {clusters[i] for i in exclude if i in clusters}
    for r in read_jsonl(f"{segmented_dir}/vinli_dev.jsonl"):
        if clusters[r["id"]] not in used_clusters:
            premises.setdefault(clusters[r["id"]], r)
    for cluster in rng.sample(sorted(premises), n_per_source):
        r = premises[cluster]
        rows.append({"id": r["id"], "source": "vinli_dev", "sentence": None, "sheet": sheet,
                     "words": _words(flatten(doc_from_json(r["doc"]))), "gold_tiers": None, "annotator": ""})
    articles = [r for r in read_jsonl(f"{segmented_dir}/vietnews_dev.jsonl") if r["id"] not in exclude]
    for r in rng.sample(articles, n_per_source):
        doc = doc_from_json(r["doc"])
        k = rng.randrange(len(doc))
        rows.append({"id": r["id"], "source": "vietnews_dev", "sentence": k, "sheet": sheet,
                     "words": _words(doc[k]), "gold_tiers": None, "annotator": ""})
    return rows


def probe_sheet(segmented_dir: str, tasks_dir: str, n_per_group: int, seed: int,
                sources=("vinli_dev", "vinli_train"), sheet: str = SHEET, exclude: set | None = None) -> list[dict]:
    """ViNLI dev alone has only ~42 premises with a negation, fewer than the 50 that G6
    needs, so premises are pooled from dev and train (neither is a test split)."""
    rng = random.Random(seed)
    by_cluster = {}
    for source in sources:
        tasks = {r["id"]: r for r in read_jsonl(f"{tasks_dir}/{source}.jsonl")}
        for r in read_jsonl(f"{segmented_dir}/{source}.jsonl"):
            task = tasks[r["id"]]
            if r["id"] not in (exclude or set()):
                by_cluster.setdefault(task["cluster"], (source, r, task["premise"]))
    edited = sorted(c for c, (_, r, _) in by_cluster.items() if has_negation(flatten(doc_from_json(r["doc"]))))
    edited_set = set(edited)
    kept = sorted(c for c in by_cluster if c not in edited_set)
    if len(edited) < n_per_group or len(kept) < n_per_group:
        raise ValueError(f"not enough distinct premises: {len(edited)} edited, {len(kept)} unchanged")
    chosen = rng.sample(edited, n_per_group) + rng.sample(kept, n_per_group)
    rng.shuffle(chosen)
    return [{"id": by_cluster[c][1]["id"], "source": by_cluster[c][0], "cluster": c, "sheet": sheet,
             "premise": by_cluster[c][2], "gold_negation": None, "annotator": ""}
            for c in chosen]


def segmentation_sheet(tasks_dir: str, n: int, seed: int, source: str = "vinli_test") -> list[dict]:
    """Distinct premises of `source`, syllables only (whitespace tokens of the NFC text)."""
    from viword.segment import normalize

    by_cluster = {}
    for r in read_jsonl(f"{tasks_dir}/{source}.jsonl"):
        by_cluster.setdefault(r["cluster"], r)
    chosen = random.Random(seed).sample(sorted(by_cluster), min(n, len(by_cluster)))
    return [{"id": by_cluster[c]["id"], "source": source, "sheet": "seg-v1",
             "syllables": normalize(by_cluster[c]["premise"]).split(), "gold_words": None, "annotator": ""}
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
    ap.add_argument("--sheet", default=SHEET, help="sheet version written into every row (blind-v2 for detector v2)")
    ap.add_argument("--exclude-sheets", nargs="*", default=[],
                    help="earlier sheets whose rows must not be drawn again")
    ap.add_argument("--only", nargs="+", default=["protect", "probe"], choices=["protect", "probe", "segmentation"])
    ap.add_argument("--segmentation-out", default="data/annot/segmentation_sheet.jsonl")
    ap.add_argument("--segmentation-n", type=int, default=300)
    ap.add_argument("--segmentation-source", default="vinli_test")
    args = ap.parse_args()
    used = {r["id"] for path in args.exclude_sheets for r in read_jsonl(path)}
    sheets = []
    if "protect" in args.only:
        sheets.append((args.protect_out, protect_sheet(args.segmented_dir, args.tasks_dir, args.per_source,
                                                       args.seed, args.sheet, used)))
    if "probe" in args.only:
        sheets.append((args.probe_out, probe_sheet(args.segmented_dir, args.tasks_dir, args.per_group, args.seed,
                                                   sheet=args.sheet, exclude=used)))
    if "segmentation" in args.only:
        sheets.append((args.segmentation_out, segmentation_sheet(args.tasks_dir, args.segmentation_n, args.seed,
                                                                 args.segmentation_source)))
    for path, rows in sheets:
        if Path(path).exists():
            raise SystemExit(f"{path} exists; refusing to overwrite annotations")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print(f"wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
