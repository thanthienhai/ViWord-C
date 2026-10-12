"""Check the structural parts of the decision gates (DATN §2.3) from diagnose.py reports.

Only the parts that need no reader are decided here:
  G1 (first half): CBR of the syllable-level LLMLingua-2 >= 5% at ratio 1/3
  G2 (second half): >= 300 ViNLI test pairs whose premise contains a negation;
      this count uses only premise text/segmentation and pair IDs, never NLI labels
  G5: Spearman(c_T(w), n_syl(w)) < 0.9 for at least 2 target tokenizers
The reader-based parts (G1 scores, G2 probe NFR, G3, G4) come from evaluate.py/analyze.py.

Usage:
  python scripts/gate_report.py --belebele results/diagnose/belebele.json \
      --vinli results/diagnose/vinli.json --vinli-pairs data/tasks/vinli_test.jsonl \
      --probe-report results/probe_negation_eval.json \
      --llmlingua2-compare results/diagnose/llmlingua2_compare_belebele.json \
          results/diagnose/llmlingua2_compare_vinli.json
"""
import argparse
import json

from viword.data import read_jsonl
from viword.diagnose import has_negation
from viword.segment import doc_from_json, flatten

G2_TEST_PAIR_THRESHOLD = 300


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def choose_syllable_method(base_method: str, package_method: str,
                           comparison_reports: list[dict]) -> tuple[str, bool]:
    """Use the scored arm only when every supplied F2 comparison passes."""
    if not comparison_reports:
        return base_method, False
    passed = all(report.get("passed", False) for report in comparison_reports)
    return (base_method if passed else package_method), True


def summarize_post_training_cbr(reports: list[dict], ratio: str = "0.333") -> dict:
    """Pool CBR counts for every trained LLMLingua-2-vi seed and task."""
    rows, partial, total = [], 0.0, 0
    for report in reports:
        for method, by_ratio in report.get("methods", {}).items():
            if not method.startswith("llmlingua2vi_s") or ratio not in by_ratio:
                continue
            cbr = by_ratio[ratio]["cbr"]
            n = int(cbr["n_all"])
            value = float(cbr["all"])
            rows.append({"task": report.get("task", "?"), "method": method,
                         "cbr": value, "n": n})
            partial += value * n
            total += n
    if not rows:
        raise ValueError(f"no trained LLMLingua-2-vi methods at ratio {ratio}")
    pooled = partial / total if total else float("nan")
    return {"rows": rows, "n": total, "cbr": pooled, "passed": pooled >= 0.05}


def count_negated_test_pairs(segmented_rows: list[dict], premise_rows: list[dict],
                             pair_rows: list[dict]) -> dict:
    """Count G2 pairs from test text only.

    Callers should pass projections containing only ``id``/``doc`` and
    ``id``/``cluster``.  In particular, this function intentionally has no
    access to the ViNLI ``label`` field.
    """
    docs = {row["id"]: row["doc"] for row in segmented_rows}
    negated = {
        row["cluster"]
        for row in premise_rows
        if has_negation(flatten(doc_from_json(docs[row["id"]])))
    }
    n_pairs = sum(row["cluster"] in negated for row in pair_rows)
    n_clusters = {row["cluster"] for row in pair_rows}
    return {
        "premises_with_negation": len(negated),
        "premise_clusters": len(n_clusters),
        "pairs_with_negated_premise": n_pairs,
        "pairs": len(pair_rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--belebele", required=True)
    ap.add_argument("--vinli", required=True)
    ap.add_argument("--vinli-pairs", required=True,
                    help="ViNLI test pairs; only id/cluster are used for the G2 count")
    ap.add_argument("--vinli-segmented", default="data/segmented/vinli_test.jsonl",
                    help="segmented ViNLI test premises; only id/doc are used")
    ap.add_argument("--vinli-premises", default="data/tasks/vinli_test.jsonl",
                    help="ViNLI test premise text; only id/cluster are used, label is ignored")
    ap.add_argument("--probe-report", help="JSON report from evaluate_probe.py")
    ap.add_argument("--syllable-method", default="llmlingua2_syl")
    ap.add_argument("--package-method", default="llmlingua2")
    ap.add_argument("--llmlingua2-compare", nargs="+", default=[],
                    help="F2 JSON reports; if any fails, use --package-method for G1")
    ap.add_argument("--llmlingua2vi-reports", nargs="+", default=[],
                    help="post-training diagnose JSON reports for llmlingua2vi_s* CBR")
    args = ap.parse_args()

    comparison_reports = [load(path) for path in args.llmlingua2_compare]
    syllable_method, audited = choose_syllable_method(
        args.syllable_method, args.package_method, comparison_reports
    )
    if audited:
        status = "PASS" if syllable_method == args.syllable_method else "FAIL"
        print(f"## F2: scored/package LLMLingua-2 agreement -> {status}\n")
        for report in comparison_reports:
            print(f"- {report['task']}: {report['n_documents']} documents, "
                  f"{'PASS' if report.get('passed') else 'FAIL'}")
            for ratio, result in report.get("per_ratio", {}).items():
                scored_cbr = result.get("scored_cbr")
                package_cbr = result.get("package_cbr")
                scored_text = "n/a" if scored_cbr is None else f"{scored_cbr:.1%}"
                package_text = "n/a" if package_cbr is None else f"{package_cbr:.1%}"
                print(f"  - @ {ratio}: mask exact {result['mask_exact_rate']:.1%}, "
                      f"syllable agreement {result['syllable_agreement']:.1%}, "
                      f"CBR scored/package {scored_text}/{package_text}, "
                      f"ΔCBR {result['cbr_diff_points']:.2f} points")
        chosen_reason = "scored arm" if syllable_method == args.syllable_method else "package original (F2 failed)"
        print(f"- G1 syllable method: `{syllable_method}` ({chosen_reason})\n")
    else:
        print("## F2: no comparison report supplied; keeping configured scored arm\n")

    if args.probe_report:
        probe = load(args.probe_report)
        print("## G6: negation-probe precision (threshold 95%)\n")
        if probe.get("sheet") != "blind-v1":
            raise SystemExit(f"{args.probe_report} was not computed from a blind annotation sheet; "
                             "G6 cannot be decided from detector-prefilled labels")
        recall = "n/a" if probe.get("recall") is None else f"{probe['recall']:.1%}"
        print(f"- hand-checked premises: {probe['n_premises']} "
              f"({probe['predicted_probe_targets']} edited by the probe, {probe['n_unedited']} not); "
              f"TP/FP/FN: {probe['tp']}/{probe['fp']}/{probe['fn']}; "
              f"precision = {probe['precision']:.1%}, recall = {recall} "
              f"-> {'PASS' if probe.get('passed') else 'FAIL'}\n")
    else:
        print("## G6: negation-probe precision\n\n- not audited (run evaluate_probe.py)\n")

    print(f"## G1 (structural): CBR of `{syllable_method}` at 1/3 (threshold 5%)\n")
    for name, path in (("belebele", args.belebele), ("vinli", args.vinli)):
        m = load(path)["methods"][syllable_method]["0.333"]
        cbr = m["cbr"]["all"]
        chance = m.get("chance_cbr")
        if chance is None:
            chance = m.get("chance_cbr_2syl")
        chance_text = "n/a" if chance is None else f"{chance:.1%}"
        print(f"- {name}: CBR = {cbr:.1%} (n = {m['cbr']['n_all']}, chance {chance_text}) "
              f"-> {'PASS' if cbr >= 0.05 else 'FAIL'}")

    if args.llmlingua2vi_reports:
        post = summarize_post_training_cbr([load(path) for path in args.llmlingua2vi_reports])
        print("\n## G1 (post-training): CBR of LLMLingua-2-vi at 1/3 (threshold 5%)\n")
        for row in post["rows"]:
            print(f"- {row['task']} / {row['method']}: CBR = {row['cbr']:.1%} (n = {row['n']})")
        print(f"- pooled: CBR = {post['cbr']:.1%} (n = {post['n']}) "
              f"-> {'PASS' if post['passed'] else 'FAIL'}")

    print(f"\n## G2 (count, ViNLI test text only; threshold {G2_TEST_PAIR_THRESHOLD})\n")
    # Project before counting so the label field in the task JSONL cannot enter the
    # gate calculation.  The test split is used only for this structural half of G2.
    segmented = [{"id": r["id"], "doc": r["doc"]}
                 for r in read_jsonl(args.vinli_segmented)]
    premises = [{"id": r["id"], "cluster": r["cluster"]}
                for r in read_jsonl(args.vinli_premises)]
    pairs = [{"id": r["id"], "cluster": r["cluster"]}
             for r in read_jsonl(args.vinli_pairs)]
    g2 = count_negated_test_pairs(segmented, premises, pairs)
    print(f"- source: ViNLI test ({args.vinli_premises}, {args.vinli_segmented}, {args.vinli_pairs})")
    print("- inputs used: premise text/segmentation and IDs only; NLI labels are ignored")
    print(f"- premises with negation: {g2['premises_with_negation']} of {g2['premise_clusters']}")
    print(f"- pairs with a negated premise: {g2['pairs_with_negated_premise']} of {g2['pairs']} "
          f"-> {'PASS' if g2['pairs_with_negated_premise'] >= G2_TEST_PAIR_THRESHOLD else 'FAIL'}")

    print("\n## G5: Spearman(c_T(w), n_syl(w)) (threshold < 0.9 for >= 2 tokenizers)\n")
    profile = load(args.belebele)["token_cost"]
    below = 0
    for name, p in profile.items():
        below += p["spearman_cost_nsyl"] < 0.9
        print(f"- {name}: fertility {p['fertility']:.2f}, CV_T {p['cv']:.2f}, "
              f"Spearman {p['spearman_cost_nsyl']:.2f}")
    print(f"-> {below} target tokenizers below 0.9 -> {'PASS' if below >= 2 else 'FAIL'}")


if __name__ == "__main__":
    main()
