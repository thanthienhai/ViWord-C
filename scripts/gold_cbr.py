"""CBR against human word boundaries (item a: breaking the "CBR = 0 by design" circle).

ViWord-C decides on VnCoreNLP words, so its CBR measured on VnCoreNLP boundaries is 0 by
construction, and re-measuring with underthesea/pyvi only compares one automatic segmenter
with another. This script measures the CBR of every system, the teacher and the syllable
systems included, on boundaries a human annotated (the segmentation sheet of
make_annotation_sheets.py, or a VLSP-style file with words joined by "_"), next to:

* the same CBR on VnCoreNLP boundaries, for the same rows;
* the CBR of random syllable deletion at the same keep rate (chance);
* the agreement of VnCoreNLP with the human boundaries (word-boundary precision/recall/F1).

Gold file, one line per context: {"id": ..., "gold_words": ["học_sinh", "đi", "học", "."]}
(or "gold_text": "học_sinh đi học ."). The syllables must be the context's own syllables.

Usage:
  python scripts/gold_cbr.py --gold data/annot/segmentation_sheet.jsonl \
      --segmented data/segmented/vinli_test.jsonl \
      --compressed results/eval/vinli_Qwen2.5-7B-Instruct.compressed.jsonl \
                   results/eval/teacher_upper_vinli_Qwen2.5-7B-Instruct.compressed.jsonl \
      --out results/gold_cbr/vinli.md --json-out results/gold_cbr/vinli.json
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from viword.data import read_jsonl
from viword.diagnose import cbr_counts, chance_cbr_counts
from viword.segment import Word, align_compressed, doc_from_json, flatten


def gold_words(row: dict) -> list[Word]:
    tokens = row.get("gold_words") or row["gold_text"].split()
    return [Word(t.replace("_", " ")) for t in tokens]


def boundary_set(words: list[Word]) -> set[int]:
    """Syllable offsets at which a word ends (the last one included)."""
    out, i = set(), 0
    for w in words:
        i += len(w.syllables)
        out.add(i)
    return out


def boundary_agreement(gold: list[Word], auto: list[Word]) -> Counter:
    g, a = boundary_set(gold), boundary_set(auto)
    return Counter(tp=len(g & a), fp=len(a - g), fn=len(g - a))


def _same_syllables(gold: list[Word], auto: list[Word]) -> bool:
    lower = lambda ws: [s.lower() for w in ws for s in w.syllables]  # noqa: E731
    return lower(gold) == lower(auto)


def measure(gold_rows: list[dict], segmented: dict[str, list[Word]], compressed_rows: list[dict]) -> dict:
    gold = {r["id"]: gold_words(r) for r in gold_rows if r.get("gold_words") or r.get("gold_text")}
    mismatched = sorted(i for i in gold if i in segmented and not _same_syllables(gold[i], segmented[i]))
    usable = {i for i in gold if i in segmented and i not in mismatched}
    agreement = Counter()
    for i in usable:
        agreement += boundary_agreement(gold[i], segmented[i])
    per = defaultdict(lambda: {"gold": Counter(), "auto": Counter(), "chance": Counter(), "n": 0})
    for row in compressed_rows:
        i = row["id"]
        if i not in usable or row["method"] in {"none", "no_context"}:
            continue
        cell = per[row["method"], round(float(row["ratio"]), 3)]
        for kind, words in (("gold", gold[i]), ("auto", segmented[i])):
            mask, _ = align_compressed(words, row["compressed"])
            counts = cbr_counts(words, mask)
            cell[kind]["partial"] += counts["all", "partial"]
            cell[kind]["full"] += counts["all", "full"]
            if kind == "gold":
                chance = chance_cbr_counts(words, mask)
                cell["chance"]["partial"] += chance["partial"]
                cell["chance"]["survived"] += chance["survived"]
        cell["n"] += 1
    share = lambda a, b: a / b if b else float("nan")  # noqa: E731
    p = share(agreement["tp"], agreement["tp"] + agreement["fp"])
    r = share(agreement["tp"], agreement["tp"] + agreement["fn"])
    systems = []
    for (method, ratio), cell in sorted(per.items()):
        g, a, c = cell["gold"], cell["auto"], cell["chance"]
        systems.append({"method": method, "ratio": ratio, "n": cell["n"],
                        "cbr_gold": share(g["partial"], g["partial"] + g["full"]),
                        "words_gold": g["partial"] + g["full"],
                        "cbr_vncorenlp": share(a["partial"], a["partial"] + a["full"]),
                        "cbr_chance_gold": share(c["partial"], c["survived"])})
    return {"contexts": len(usable), "syllable_mismatch": mismatched[:20], "n_syllable_mismatch": len(mismatched),
            "vncorenlp_boundary": {"precision": p, "recall": r, "f1": share(2 * p * r, p + r), **agreement},
            "systems": systems}


def report(result: dict) -> str:
    b = result["vncorenlp_boundary"]
    lines = ["# CBR on human word boundaries", "",
             f"Contexts: {result['contexts']} (dropped {result['n_syllable_mismatch']} whose gold syllables differ "
             "from the segmented ones).",
             f"VnCoreNLP word boundaries vs human: P {b['precision']:.4f}, R {b['recall']:.4f}, F1 {b['f1']:.4f}.", "",
             "| method | ratio | n | CBR (human) | CBR (VnCoreNLP) | chance CBR (human) | words |",
             "|---|---|---|---|---|---|---|"]
    for s in result["systems"]:
        lines.append(f"| {s['method']} | {s['ratio']} | {s['n']} | {s['cbr_gold']:.4f} | {s['cbr_vncorenlp']:.4f} | "
                     f"{s['cbr_chance_gold']:.4f} | {s['words_gold']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--segmented", required=True)
    ap.add_argument("--compressed", nargs="+", required=True, help="evaluate.py <out>.compressed.jsonl files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    segmented = {r["id"]: flatten(doc_from_json(r["doc"])) for r in read_jsonl(args.segmented)}
    rows = [r for path in args.compressed for r in read_jsonl(path)]
    result = measure(read_jsonl(args.gold), segmented, rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report(result), encoding="utf-8")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(report(result))


if __name__ == "__main__":
    main()
