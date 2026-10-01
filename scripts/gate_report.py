"""Check the structural parts of the decision gates (DATN §2.3) from diagnose.py reports.

Only the parts that need no reader are decided here:
  G1 (first half): CBR of the syllable-level LLMLingua-2 >= 5% at ratio 1/3
  G2 (second half): >= 300 ViNLI pairs whose premise contains a negation
  G5: Spearman(c_T(w), n_syl(w)) < 0.9 for at least 2 target tokenizers
The reader-based parts (G1 scores, G2 probe NFR, G3, G4) come from evaluate.py/analyze.py.

Usage:
  python scripts/gate_report.py --belebele results/diagnose/belebele.json \
      --vinli results/diagnose/vinli.json --vinli-pairs data/tasks/vinli_test.jsonl
"""
import argparse
import json

from viword.data import read_jsonl
from viword.diagnose import has_negation
from viword.segment import doc_from_json, flatten


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--belebele", required=True)
    ap.add_argument("--vinli", required=True)
    ap.add_argument("--vinli-pairs", required=True, help="all test pairs, to count pairs per premise")
    ap.add_argument("--vinli-segmented", default="data/segmented/vinli_test_premises.jsonl")
    ap.add_argument("--vinli-premises", default="data/tasks/vinli_test_premises.jsonl")
    ap.add_argument("--syllable-method", default="llmlingua2_syl")
    args = ap.parse_args()

    print("## G1 (structural): CBR of the syllable-level compressor at 1/3 (threshold 5%)\n")
    for name, path in (("belebele", args.belebele), ("vinli", args.vinli)):
        m = load(path)["methods"][args.syllable_method]["0.333"]
        cbr = m["cbr"]["all"]
        print(f"- {name}: CBR = {cbr:.1%} (n = {m['cbr']['n_all']}, chance {m['chance_cbr_2syl']:.1%}) "
              f"-> {'PASS' if cbr >= 0.05 else 'FAIL'}")

    print("\n## G2 (count): ViNLI test pairs whose premise contains a negation (threshold 300)\n")
    docs = {r["id"]: r["doc"] for r in read_jsonl(args.vinli_segmented)}
    negated = {r["cluster"] for r in read_jsonl(args.vinli_premises)
               if has_negation(flatten(doc_from_json(docs[r["id"]])))}
    pairs = read_jsonl(args.vinli_pairs)
    n_pairs = sum(r["cluster"] in negated for r in pairs)
    print(f"- premises with negation: {len(negated)} of {len({r['cluster'] for r in pairs})}")
    print(f"- pairs with a negated premise: {n_pairs} of {len(pairs)} -> {'PASS' if n_pairs >= 300 else 'FAIL'}")

    print("\n## G5: Spearman(c_T(w), n_syl(w)) (threshold < 0.9 for >= 2 tokenizers)\n")
    profile = load(args.belebele)["token_cost"]
    below = 0
    for name, p in profile.items():
        below += p["spearman_cost_nsyl"] < 0.9 and name != "xlm-roberta-base"
        print(f"- {name}: fertility {p['fertility']:.2f}, CV_T {p['cv']:.2f}, "
              f"Spearman {p['spearman_cost_nsyl']:.2f}")
    print(f"-> {below} target tokenizers below 0.9 -> {'PASS' if below >= 2 else 'FAIL'}")


if __name__ == "__main__":
    main()
