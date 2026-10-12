"""Evaluate the linguistic-protection detector on a blind, hand-annotated sample.

Labels must come from a sheet written by ``scripts/make_annotation_sheets.py``
(``"sheet": "blind-v1"``): the sheet shows the words, POS and NER but never the
detector's output, and samples sentences uniformly, so sentences without any protected
word are included. Each row covers one ViNLI premise (``sentence`` null) or one sentence
of a VietNews article (``sentence`` = index). ``gold_tiers`` maps a word index to a tier;
``{}`` means the sentence has no protected word. The words stored in the sheet must match
the current segmentation, so labels cannot silently drift after re-segmentation.

The detector runs on the whole document (its rules look at the next word) and is then
scored on the annotated span.

Usage:
  python scripts/evaluate_protect.py --labels data/annot/protect_sheet.jsonl \
      --segmented-dir data/segmented --out results/protect_eval.md --json-out results/protect_eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from viword.protect import TIERS, protection_tiers
from viword.segment import doc_from_json, flatten

SHEET = "blind-v1"


def load_segmented(directory: str, source: str) -> dict[str, list]:
    path = Path(directory) / f"{source}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing segmented source: {path}")
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row["id"] in rows:
                    raise ValueError(f"duplicate segmented id {source}:{row['id']}")
                rows[row["id"]] = row["doc"]
    return rows


def _span(doc, sentence):
    """(all words, start, end) of the annotated span within the flattened document."""
    words = flatten(doc)
    if sentence is None:
        return words, 0, len(words)
    start = sum(len(sent) for sent in doc[:sentence])
    return words, start, start + len(doc[sentence])


def evaluate(labels_path: str, segmented_dir: str, min_rows: int = 200) -> dict:
    labels = []
    with open(labels_path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                labels.append(json.loads(line))
    if len(labels) < min_rows:
        raise ValueError(f"expected at least {min_rows} annotated sentences, got {len(labels)}")

    sources = {}
    seen = set()
    no_p = 0
    counts = {tier: Counter(tp=0, fp=0, fn=0, gold=0, predicted=0) for tier in TIERS}
    for row in labels:
        key = (row["source"], row["id"], row.get("sentence"))
        if key in seen:
            raise ValueError(f"duplicate annotation: {key}")
        seen.add(key)
        if row.get("sheet") != SHEET:
            raise ValueError(f"{key}: labels must come from a blind sheet ({SHEET}, "
                             "scripts/make_annotation_sheets.py); detector-prefilled labels are circular")
        if not row.get("annotator"):
            raise ValueError(f"missing annotator for {key}")
        gold_map = row.get("gold_tiers")
        if not isinstance(gold_map, dict):
            raise ValueError(f"{key}: gold_tiers not filled in (use {{}} for no protected word)")
        if row["source"] not in sources:
            sources[row["source"]] = load_segmented(segmented_dir, row["source"])
        docs = sources[row["source"]]
        if row["id"] not in docs:
            raise ValueError(f"annotation id not found: {key}")
        doc = doc_from_json(docs[row["id"]])
        words, start, end = _span(doc, row.get("sentence"))
        span = words[start:end]
        if [w[1] for w in row.get("words", [])] != [w.text for w in span]:
            raise ValueError(f"{key}: sheet words differ from the current segmentation; re-annotate")
        unknown = [i for i in gold_map if not str(i).isdigit() or int(i) >= len(span)
                   or gold_map[i] not in TIERS]
        if unknown:
            raise ValueError(f"invalid gold index/tier for {key}: {unknown[:3]}")
        gold = [gold_map.get(str(i)) for i in range(len(span))]
        no_p += not gold_map
        predicted = protection_tiers(words)[start:end]
        for tier in TIERS:
            c = counts[tier]
            for actual, expected in zip(predicted, gold):
                actual_yes, expected_yes = actual == tier, expected == tier
                c["tp"] += actual_yes and expected_yes
                c["fp"] += actual_yes and not expected_yes
                c["fn"] += expected_yes and not actual_yes
                c["predicted"] += actual_yes
                c["gold"] += expected_yes

    report = {"n_sentences": len(labels), "n_without_p": no_p, "sheet": SHEET,
              "sources": sorted(sources), "tiers": {}}
    for tier in TIERS:
        c = counts[tier]
        precision = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
        recall = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report["tiers"][tier] = {**c, "precision": precision, "recall": recall, "f1": f1}
    return report


def verify(path: str) -> None:
    """Refuse to continue (select_pi, eval) without a blind-sheet report."""
    if not Path(path).is_file():
        raise SystemExit(f"missing {path}; annotate the blind sheet and run evaluate_protect.py")
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if report.get("sheet") != SHEET:
        raise SystemExit(f"{path} was not computed from a blind annotation sheet")
    print(f"protect evaluation OK: {path}")


def render_report(report: dict) -> str:
    lines = [
        "# Linguistic-protection detector evaluation",
        "",
        f"Annotated sentences: **{report['n_sentences']}** (without any protected word: {report['n_without_p']}); "
        f"blind sheet `{report['sheet']}`",
        f"Sources: {', '.join(report['sources'])}",
        "",
        "| tier | gold positives | predicted positives | TP | FP | FN | precision | recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tier in TIERS:
        c = report["tiers"][tier]
        lines.append(f"| {tier} | {c['gold']} | {c['predicted']} | {c['tp']} | {c['fp']} | {c['fn']} | "
                     f"{c['precision']:.3f} | {c['recall']:.3f} | {c['f1']:.3f} |")
    lines += [
        "",
        "The report is an input to π configuration selection; do not select π before this file is regenerated.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels")
    parser.add_argument("--segmented-dir", default="data/segmented")
    parser.add_argument("--out")
    parser.add_argument("--json-out")
    parser.add_argument("--min-rows", type=int, default=200)
    parser.add_argument("--verify", help="check an existing JSON report and exit")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.verify:
        verify(args.verify)
        return
    if not args.labels or not args.out:
        parser.error("--labels and --out are required unless --verify is used")
    try:
        report = evaluate(args.labels, args.segmented_dir, args.min_rows)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise SystemExit(f"evaluate_protect: {exc}") from exc
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(render_report(report), encoding="utf-8")
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
    print(render_report(report), end="")


if __name__ == "__main__":
    main()
