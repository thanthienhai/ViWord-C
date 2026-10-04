"""Tables and tests from evaluation rows (DATN §4.3).

  1. mean main metric per task × reader × method × ratio, with a cluster-bootstrap 95% CI;
  2. with --target/--reference: paired one-sided test target > reference per task × ratio,
     Holm-adjusted within each reader × reference. Give the strong simple baselines
     (truncation, lead) as references too: in LACC they tied the learned compressors;
  3. with --factorial: the 2×2 design (labels × decision unit, DATN §2.3), main effects and
     interaction with cluster-bootstrap CIs, so the H1 reading does not rest on eyeballing
     four means;
  4. ViNLI: NFR (label flips vs. the uncompressed prediction), overall, on examples the
     reader got right without compression, and on negated premises (H2); exact McNemar on
     flips, target vs each reference;
  5. VietNews: non-inferiority of target vs each reference on ROUGE-L (--margin in points).

Usage:
  python scripts/analyze.py --rows results/eval/*.jsonl --target viword_s0 \
      --reference llmlingua2vi_s0 truncation lead \
      --factorial llmlingua2vi_s0 llmlingua2vi_wordpool_s0 viword_syl_s0 viword_s0
"""
import argparse
from collections import defaultdict

import numpy as np

from viword.data import read_jsonl
from viword.eval import MAIN_METRIC
from viword.stats import cluster_bootstrap, holm, mcnemar_exact, non_inferiority, paired_test


def by_key(rows, *keys):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    return groups


def paired(rows_a, rows_b):
    """Align two systems on example id; returns (a_rows, b_rows) in the same order."""
    b = {r["id"]: r for r in rows_b}
    common = [r for r in rows_a if r["id"] in b]
    return common, [b[r["id"]] for r in common]


def ci(t):
    return f"{t['diff']:+.4f} [{t['ci_low']:+.4f}, {t['ci_high']:+.4f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--target", default=None)
    ap.add_argument("--reference", nargs="+", default=[],
                    help="one or more references; Holm is applied per reader x reference")
    ap.add_argument("--factorial", nargs=4, default=None,
                    metavar=("SYL_LABELS_SYL_UNIT", "SYL_LABELS_WORD_UNIT", "WORD_LABELS_SYL_UNIT",
                             "WORD_LABELS_WORD_UNIT"),
                    help="the four cells of the 2x2 design (DATN §2.3)")
    ap.add_argument("--margin", type=float, default=0.5, help="non-inferiority margin, ROUGE-L points")
    ap.add_argument("--n-boot", type=int, default=10_000)
    args = ap.parse_args()
    # results/eval/*.jsonl also matches evaluate.py's compression caches, which are not result rows
    rows = [r for path in args.rows if not path.endswith(".compressed.jsonl") for r in read_jsonl(path)]
    references = args.reference if args.target else []

    print("| task | reader | method | ratio | n | clusters | mean | 95% CI | tokens/budget |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (task, reader, method, ratio), group in sorted(by_key(rows, "task", "reader", "method", "ratio").items()):
        values = [r[MAIN_METRIC[task]] for r in group]
        clusters = [r["cluster"] for r in group]
        boot = cluster_bootstrap(values, clusters, args.n_boot)
        adherence = np.mean([r["n_tokens_compressed"] / max(r["budget"], 1) for r in group])
        print(f"| {task} | {reader} | {method} | {ratio} | {len(group)} | {len(set(clusters))} | "
              f"{np.mean(values):.4f} | [{np.percentile(boot, 2.5):.4f}, {np.percentile(boot, 97.5):.4f}] | "
              f"{adherence:.3f} |")

    systems = by_key(rows, "task", "reader", "method", "ratio")
    uncompressed = by_key([r for r in rows if r["method"] == "none"], "task", "reader")
    readers = sorted({r["reader"] for r in rows})

    for reference in references:
        print(f"\n## {args.target} vs {reference} (one-sided, Holm within reader)\n")
        for reader in readers:
            tests = {}
            for (task, rd, method, ratio), group in systems.items():
                ref = systems.get((task, rd, reference, ratio))
                if rd != reader or method != args.target or ref is None:
                    continue
                a, b = paired(group, ref)
                m = MAIN_METRIC[task]
                tests[task, ratio] = paired_test([r[m] for r in a], [r[m] for r in b],
                                                 [r["cluster"] for r in a], args.n_boot)
            if not tests:
                continue
            adjusted = holm({k: t["p_one_sided"] for k, t in tests.items()})
            print(f"reader {reader}")
            print("| task | ratio | diff [95% CI] | p | p Holm |\n|---|---|---|---|---|")
            for (task, ratio), t in sorted(tests.items()):
                print(f"| {task} | {ratio} | {ci(t)} | {t['p_one_sided']:.4f} | {adjusted[task, ratio]:.4f} |")

    if args.factorial:
        cell_names = args.factorial
        print("\n## 2x2 design: label unit x decision unit\n")
        print(f"Cells: SS = {cell_names[0]}, SW = {cell_names[1]}, WS = {cell_names[2]}, WW = {cell_names[3]} "
              "(first letter: labels, second: decision unit; S = syllable, W = word).")
        print("Per-example contrasts on the main metric, cluster-bootstrap 95% CI. unit = word − syllable "
              "decision (H1); label = word − syllable labels; interaction = unit effect with word labels − "
              "with syllable labels.\n")
        print("| task | reader | ratio | n | SS / SW / WS / WW | unit | label | interaction |")
        print("|---|---|---|---|---|---|---|---|")
        for task, reader, _, ratio in sorted(k for k in systems if k[2] == cell_names[0]):
            cells = [{r["id"]: r for r in systems.get((task, reader, m, ratio), [])} for m in cell_names]
            ids = sorted(set.intersection(*(set(c) for c in cells)))
            if not ids:
                continue
            m = MAIN_METRIC[task]
            v = np.array([[c[i][m] for i in ids] for c in cells])
            clusters = [cells[0][i]["cluster"] for i in ids]
            contrasts = [((v[1] - v[0]) + (v[3] - v[2])) / 2,  # unit
                         ((v[2] - v[0]) + (v[3] - v[1])) / 2,  # label
                         (v[3] - v[2]) - (v[1] - v[0])]  # interaction
            cols = [ci(paired_test(c, np.zeros_like(c), clusters, args.n_boot)) for c in contrasts]
            means = " / ".join(f"{x:.3f}" for x in v.mean(axis=1))
            print(f"| {task} | {reader} | {ratio} | {len(ids)} | {means} | {' | '.join(cols)} |")

    nli = [k for k in systems if k[0] == "vinli" and k[2] not in {"none", "no_context"}]
    if nli:
        # premises that the negation probe changed are the premises that contain a negation
        negated = {(r["reader"], r["id"]) for r in rows if r["task"] == "vinli" and r["method"] == "probe_negation"
                   and r["n_tokens_compressed"] < r["n_tokens_original"]}
        print("\n## ViNLI flip rate (NFR) vs. uncompressed prediction\n")
        print("Negated premises = pairs whose premise probe_negation changed (needs probe_negation rows). "
              "Gate G2 reads the probe's own row in the last column.\n")
        print("| reader | method | ratio | NFR | NFR on originally correct | NFR on negated premises (n) |")
        print("|---|---|---|---|---|---|")
        flips = {}
        for key in sorted(nli):
            task, reader, method, ratio = key
            a, base = paired(systems[key], uncompressed[task, reader])
            flipped = [x["prediction"] != y["prediction"] for x, y in zip(a, base)]
            correct = [f for f, y in zip(flipped, base) if y["accuracy"] == 1.0]
            neg = [f for f, x in zip(flipped, a) if (reader, x["id"]) in negated]
            flips[reader, method, ratio] = {x["id"]: f for x, f in zip(a, flipped)}
            print(f"| {reader} | {method} | {ratio} | {np.mean(flipped):.4f} | "
                  f"{np.mean(correct) if correct else float('nan'):.4f} | "
                  f"{np.mean(neg) if neg else float('nan'):.4f} ({len(neg)}) |")
        for reference in references:
            print(f"\nMcNemar on flips, {args.target} vs {reference}:")
            for (reader, method, ratio), fa in sorted(flips.items()):
                fb = flips.get((reader, reference, ratio))
                if method != args.target or fb is None:
                    continue
                ids = fa.keys() & fb.keys()
                only_a = sum(fa[i] and not fb[i] for i in ids)
                only_b = sum(fb[i] and not fa[i] for i in ids)
                print(f"  {reader} @ {ratio}: target-only flips {only_a}, reference-only flips {only_b}, "
                      f"p = {mcnemar_exact(only_a, only_b):.4f}")

    for reference in references:
        for (task, reader, method, ratio), group in sorted(systems.items()):
            ref = systems.get((task, reader, reference, ratio))
            if task != "vietnews" or method != args.target or ref is None:
                continue
            a, b = paired(group, ref)
            result = non_inferiority([r["rougeL"] for r in a], [r["rougeL"] for r in b],
                                     [r["cluster"] for r in a], args.margin / 100, args.n_boot)
            print(f"\nVietNews non-inferiority vs {reference}, {reader} @ {ratio}: "
                  f"diff {100 * result['diff']:+.2f} lower 95% {100 * result['lower_95']:+.2f} "
                  f"(margin −{args.margin}) -> {'non-inferior' if result['non_inferior'] else 'not shown'}")


if __name__ == "__main__":
    main()
