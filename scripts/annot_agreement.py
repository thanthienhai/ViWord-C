"""Inter-annotator agreement (item 12): Cohen's κ between two filled annotation sheets.

The second annotator works from a fresh copy of the blind sheet, without the first
annotator's labels and without any LLM suggestion. Rows are matched by (id, sentence).

* protect sheets: one category per word (T1, T2, T3 or none), κ over all words, plus κ of
  "protected vs not";
* probe sheets: one boolean per premise (gold_negation);
* segmentation sheets: one boolean per syllable gap (word boundary or not).

Usage:
  python scripts/annot_agreement.py --kind protect --a data/annot/protect_sheet.jsonl \
      --b data/annot/protect_sheet_annot2.jsonl --out results/annot_agreement_protect.md
"""
import argparse
import json
from pathlib import Path

from viword.data import read_jsonl
from viword.stats import cohen_kappa


def _key(row: dict):
    return row["id"], row.get("sentence")


def paired_labels(kind: str, rows_a: list[dict], rows_b: list[dict]) -> tuple[list, list]:
    b = {_key(r): r for r in rows_b}
    la, lb = [], []
    for ra in rows_a:
        rb = b.get(_key(ra))
        if rb is None:
            continue
        if kind == "protect":
            if ra.get("gold_tiers") is None or rb.get("gold_tiers") is None:
                continue
            for index, *_ in ra["words"]:
                la.append(ra["gold_tiers"].get(str(index), "none"))
                lb.append(rb["gold_tiers"].get(str(index), "none"))
        elif kind == "probe":
            if ra.get("gold_negation") is None or rb.get("gold_negation") is None:
                continue
            la.append(bool(ra["gold_negation"]))
            lb.append(bool(rb["gold_negation"]))
        else:  # segmentation: a boundary after syllable k?
            ta, tb = ra.get("gold_words"), rb.get("gold_words")
            if not ta or not tb:
                continue
            gaps = lambda ws: [g for w in ws for g in [False] * (len(w.split("_")) - 1) + [True]][:-1]  # noqa: E731
            ga, gb = gaps(ta), gaps(tb)
            if len(ga) != len(gb):
                continue
            la += ga
            lb += gb
    return la, lb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["protect", "probe", "segmentation"])
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    la, lb = paired_labels(args.kind, read_jsonl(args.a), read_jsonl(args.b))
    result = {"kind": args.kind, "n_units": len(la), "kappa": cohen_kappa(la, lb),
              "observed_agreement": sum(x == y for x, y in zip(la, lb)) / len(la),
              "annotator_a": sorted({r.get("annotator", "") for r in read_jsonl(args.a)}),
              "annotator_b": sorted({r.get("annotator", "") for r in read_jsonl(args.b)})}
    if args.kind == "protect":
        result["kappa_protected_vs_not"] = cohen_kappa([x != "none" for x in la], [y != "none" for y in lb])
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
