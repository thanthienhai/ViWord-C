"""Agreement between two teachers' distilled labels on the same paragraphs (items 11 and g).

Compares two `scripts/distill.py` outputs (e.g. the main teacher and an open teacher, or
the default and the neutral prompt) on the (paragraph id, rate) pairs both contain:
syllable-label agreement and Cohen's κ, the keep rate and teacher CBR of each, the mean
absolute difference of the soft word labels, and the retention of T1 words (does the
neutral prompt keep fewer negations, numbers and names?).

Usage:
  python scripts/teacher_agreement.py --a data/distill/distilled.jsonl \
      --b data/distill/distilled_open.jsonl --out results/teacher_agreement_open.md
"""
import argparse
import json
from pathlib import Path

import numpy as np

from viword.data import read_jsonl
from viword.protect import protection_tiers
from viword.segment import doc_from_json, flatten
from viword.stats import cohen_kappa


def t1_retention(records: list[dict]) -> float:
    kept = total = 0
    for r in records:
        words = flatten(doc_from_json(r["doc"]))
        for tier, label in zip(protection_tiers(words), r["word_labels"]):
            if tier == "T1":
                total += 1
                kept += label > 0
    return kept / total if total else float("nan")


def compare(rows_a: list[dict], rows_b: list[dict]) -> dict:
    b = {(r["id"], r["rate"]): r for r in rows_b}
    pairs = [(r, b[r["id"], r["rate"]]) for r in rows_a if (r["id"], r["rate"]) in b]
    pairs = [(x, y) for x, y in pairs if len(x["syllable_labels"]) == len(y["syllable_labels"])]
    if not pairs:
        raise SystemExit("no (id, rate) pair in common")
    sa = [s for x, _ in pairs for s in x["syllable_labels"]]
    sb = [s for _, y in pairs for s in y["syllable_labels"]]
    wa = np.array([w for x, _ in pairs for w in x["word_labels"]])
    wb = np.array([w for _, y in pairs for w in y["word_labels"]])
    side = lambda rows, key: float(np.mean([r[key] for r in rows]))  # noqa: E731
    xs, ys = [x for x, _ in pairs], [y for _, y in pairs]
    return {"pairs": len(pairs), "syllables": len(sa),
            "syllable_agreement": float(np.mean(np.array(sa) == np.array(sb))),
            "syllable_kappa": cohen_kappa(sa, sb),
            "word_label_mae": float(np.mean(np.abs(wa - wb))),
            "keep_rate": [side(xs, "keep_rate"), side(ys, "keep_rate")],
            "teacher_cbr": [side(xs, "teacher_cbr"), side(ys, "teacher_cbr")],
            "t1_retention": [t1_retention(xs), t1_retention(ys)]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="distilled.jsonl of the reference teacher")
    ap.add_argument("--b", required=True, help="distilled.jsonl of the other teacher / prompt")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    result = compare(read_jsonl(args.a), read_jsonl(args.b))
    lines = ["# Teacher label agreement", "", f"A = `{args.a}`, B = `{args.b}`; {result['pairs']} (paragraph, rate) "
             f"pairs, {result['syllables']} syllables.", "",
             f"- syllable labels: agreement {result['syllable_agreement']:.4f}, Cohen's κ {result['syllable_kappa']:.4f}",
             f"- soft word labels: mean |A − B| {result['word_label_mae']:.4f}",
             f"- keep rate A / B: {result['keep_rate'][0]:.4f} / {result['keep_rate'][1]:.4f}",
             f"- teacher CBR A / B: {result['teacher_cbr'][0]:.4f} / {result['teacher_cbr'][1]:.4f}",
             f"- T1 retention A / B: {result['t1_retention'][0]:.4f} / {result['t1_retention'][1]:.4f}", ""]
    text = "\n".join(lines)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
        Path(args.out).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
