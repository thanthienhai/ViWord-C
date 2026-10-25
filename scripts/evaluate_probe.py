"""Evaluate whether the H2 negation probe targets real negation premises (gate G6).

Labels must come from the blind sheet of ``scripts/make_annotation_sheets.py``
(``"sheet": "blind-v1"``): distinct ViNLI premises, half edited by the probe and half
not, shuffled, with no hint of the probe's decision. G6 (DATN §4.0) needs at least 50
*distinct* premises edited by the probe, of which >= 95% must contain a real negation.
Recall is reported from the unedited half (premises with a negation the probe missed).

Usage:
  python scripts/evaluate_probe.py --labels data/annot/probe_sheet.jsonl \
      --segmented-dir data/segmented --out results/probe_negation_eval.md \
      --json-out results/probe_negation_eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from viword.baselines import NegationProbe
from viword.compressor import Context
from viword.diagnose import has_negation
from viword.segment import doc_from_json, flatten, render_all


def load_segmented(directory: str, source: str) -> dict[str, dict]:
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
                rows[row["id"]] = row
    return rows


SHEET = "blind-v1"
SHEETS = {"blind-v1", "blind-v2"}  # blind-v2: the larger probe sheet drawn with the detector-v2 audit


def evaluate(labels_path: str, segmented_dir: str, min_rows: int = 50) -> dict:
    """`min_rows` is the minimum number of distinct premises edited by the probe."""
    with open(labels_path, encoding="utf-8") as handle:
        labels = [json.loads(line) for line in handle if line.strip()]

    sources, seen, clusters = {}, set(), set()
    n_unedited = 0
    tp = fp = fn = 0
    probe = NegationProbe()
    for row in labels:
        key = (row["source"], row["id"])
        if key in seen:
            raise ValueError(f"duplicate annotation: {key[0]}:{key[1]}")
        seen.add(key)
        if row.get("sheet") not in SHEETS:
            raise ValueError(f"{key[0]}:{key[1]}: labels must come from a blind sheet ({sorted(SHEETS)}, "
                             "scripts/make_annotation_sheets.py)")
        if not row.get("annotator"):
            raise ValueError(f"missing annotator for {key[0]}:{key[1]}")
        cluster = row.get("cluster") or row["id"]
        if cluster in clusters:
            raise ValueError(f"premise {cluster} is annotated twice; G6 needs distinct premises")
        clusters.add(cluster)
        if not isinstance(row.get("gold_negation"), bool):
            raise ValueError(f"gold_negation must be boolean for {key[0]}:{key[1]}")
        if row["source"] not in sources:
            sources[row["source"]] = load_segmented(segmented_dir, row["source"])
        docs = sources[row["source"]]
        if row["id"] not in docs:
            raise ValueError(f"annotation id not found: {key[0]}:{key[1]}")
        doc = doc_from_json(docs[row["id"]]["doc"])
        words = flatten(doc)
        original = render_all(words)
        compressed = probe.compress(Context(text=original, doc=doc), 0, None)
        predicted = compressed != original
        if predicted != has_negation(words):
            raise ValueError(f"probe/detector mismatch for {key[0]}:{key[1]}")
        gold = row["gold_negation"]
        n_unedited += not predicted
        tp += predicted and gold
        fp += predicted and not gold
        fn += gold and not predicted

    predicted = tp + fp
    if predicted < min_rows:
        raise ValueError(f"expected at least {min_rows} distinct premises edited by the probe, got {predicted}")
    gold = tp + fn
    precision = tp / predicted if predicted else 0.0
    # recall is only meaningful when premises the probe left alone were annotated too
    recall = tp / gold if gold and n_unedited else None
    f1 = 2 * precision * recall / (precision + recall) if recall and precision + recall else None
    return {
        "sheet": labels[0]["sheet"] if labels else SHEET,
        "n_unedited": n_unedited,
        "n_premises": len(labels),
        "sources": sorted(sources),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gold_negation": gold,
        "predicted_probe_targets": predicted,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": 0.95,
        "passed": precision >= 0.95,
    }


def render_report(report: dict) -> str:
    return "\n".join([
        "# Negation-probe audit",
        "",
        f"Hand-checked premises: **{report['n_premises']}**",
        f"Sources: {', '.join(report['sources'])}",
        "",
        "| measure | value |",
        "|---|---:|",
        f"| gold negation | {report['gold_negation']} |",
        f"| probe targets | {report['predicted_probe_targets']} |",
        f"| TP / FP / FN | {report['tp']} / {report['fp']} / {report['fn']} |",
        f"| precision | {report['precision']:.3f} |",
        f"| recall (from {report['n_unedited']} unedited premises) | "
        f"{'n/a' if report['recall'] is None else format(report['recall'], '.3f')} |",
        f"| F1 | {'n/a' if report['f1'] is None else format(report['f1'], '.3f')} |",
        f"| G6 (precision ≥ {report['threshold']:.2f}) | {'PASS' if report['passed'] else 'FAIL'} |",
        "",
        "The probe deletes only words selected by `is_negation()` after POS/context disambiguation.",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--segmented-dir", default="data/segmented")
    parser.add_argument("--out", required=True)
    parser.add_argument("--json-out")
    parser.add_argument("--min-rows", type=int, default=50)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        report = evaluate(args.labels, args.segmented_dir, args.min_rows)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise SystemExit(f"evaluate_probe: {exc}") from exc
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(render_report(report), encoding="utf-8")
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
    print(render_report(report), end="")


if __name__ == "__main__":
    main()
