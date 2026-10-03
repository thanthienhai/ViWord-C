"""Tables and tests from evaluation rows (DATN §4.3).

  1. mean main metric per task × reader × method × ratio, with a cluster-bootstrap 95% CI;
  2. with --target/--reference: paired one-sided test target > reference per task × ratio,
     Holm-adjusted within each reader;
  3. ViNLI: NFR (label flips vs. the uncompressed prediction), overall and on examples the
     reader got right without compression; exact McNemar on flips, target vs reference;
  4. VietNews: non-inferiority of target vs reference on ROUGE-L (--margin in points).

Usage:
  python scripts/analyze.py --rows results/eval/*.jsonl --target viword_s0 --reference llmlingua2vi_s0
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--target", default=None)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--margin", type=float, default=0.5, help="non-inferiority margin, ROUGE-L points")
    ap.add_argument("--n-boot", type=int, default=10_000)
    args = ap.parse_args()
    rows = [r for path in args.rows for r in read_jsonl(path)]

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

    if args.target and args.reference:
        print(f"\n## {args.target} vs {args.reference} (one-sided, Holm within reader)\n")
        for reader in sorted({r["reader"] for r in rows}):
            tests = {}
            for (task, rd, method, ratio), group in systems.items():
                ref = systems.get((task, rd, args.reference, ratio))
                if rd != reader or method != args.target or ref is None:
                    continue
                a, b = paired(group, ref)
                m = MAIN_METRIC[task]
                tests[task, ratio] = paired_test([r[m] for r in a], [r[m] for r in b],
                                                 [r["cluster"] for r in a], args.n_boot)
            adjusted = holm({k: t["p_one_sided"] for k, t in tests.items()})
            print(f"reader {reader}")
            print("| task | ratio | diff | 95% CI | p | p Holm |\n|---|---|---|---|---|---|")
            for (task, ratio), t in sorted(tests.items()):
                print(f"| {task} | {ratio} | {t['diff']:+.4f} | [{t['ci_low']:+.4f}, {t['ci_high']:+.4f}] | "
                      f"{t['p_one_sided']:.4f} | {adjusted[task, ratio]:.4f} |")

    nli = [k for k in systems if k[0] == "vinli" and k[2] not in {"none", "no_context"}]
    if nli:
        print("\n## ViNLI flip rate (NFR) vs. uncompressed prediction\n")
        print("| reader | method | ratio | NFR | NFR on originally correct |\n|---|---|---|---|---|")
        flips = {}
        for key in sorted(nli):
            task, reader, method, ratio = key
            a, base = paired(systems[key], uncompressed[task, reader])
            flipped = [x["prediction"] != y["prediction"] for x, y in zip(a, base)]
            correct = [f for f, y in zip(flipped, base) if y["accuracy"] == 1.0]
            flips[reader, method, ratio] = {x["id"]: f for x, f in zip(a, flipped)}
            print(f"| {reader} | {method} | {ratio} | {np.mean(flipped):.4f} | "
                  f"{np.mean(correct) if correct else float('nan'):.4f} |")
        if args.target and args.reference:
            print(f"\nMcNemar on flips, {args.target} vs {args.reference}:")
            for (reader, method, ratio), fa in sorted(flips.items()):
                fb = flips.get((reader, args.reference, ratio))
                if method != args.target or fb is None:
                    continue
                ids = fa.keys() & fb.keys()
                only_a = sum(fa[i] and not fb[i] for i in ids)
                only_b = sum(fb[i] and not fa[i] for i in ids)
                print(f"  {reader} @ {ratio}: target-only flips {only_a}, reference-only flips {only_b}, "
                      f"p = {mcnemar_exact(only_a, only_b):.4f}")

    if args.target and args.reference:
        for (task, reader, method, ratio), group in sorted(systems.items()):
            ref = systems.get((task, reader, args.reference, ratio))
            if task != "vietnews" or method != args.target or ref is None:
                continue
            a, b = paired(group, ref)
            result = non_inferiority([r["rougeL"] for r in a], [r["rougeL"] for r in b],
                                     [r["cluster"] for r in a], args.margin / 100, args.n_boot)
            print(f"\nVietNews non-inferiority {reader} @ {ratio}: diff {100 * result['diff']:+.2f} "
                  f"lower 95% {100 * result['lower_95']:+.2f} (margin −{args.margin}) -> "
                  f"{'non-inferior' if result['non_inferior'] else 'not shown'}")


if __name__ == "__main__":
    main()
