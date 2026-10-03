"""Decide the reader-based parts of the decision gates (DATN §2.3) from evaluate.py rows.

Run on the DEV sets, never on test, so that no gate decision looks at test data.
Each gate is checked only if its methods are present in the rows:
  G1 (score half): unit=word pooling of LLMLingua-2 beats the syllable decision on >= 1 task x ratio
  G2 (NFR half):   ViNLI flip rate of probe_negation >= 5 points, on pairs whose premise it changed
  G3:              at 1/5, the best token-level compressor is within 5 points of the best
                   sentence-level baseline on a majority of tasks
  G4:              at 1/3, the teacher beats truncation on >= 2 tasks (200 dev examples)
"Beats" is the point estimate, as written in the gate; the cluster-bootstrap 95% CI is
printed next to it.

Usage:
  python scripts/gate_eval.py --rows results/gate/teacher_*.jsonl
  python scripts/gate_eval.py --rows results/gate/dev_*.jsonl
"""
import argparse
from collections import defaultdict

import numpy as np

from viword.data import read_jsonl
from viword.eval import MAIN_METRIC
from viword.stats import paired_test


def compare(systems, task, a, b, ratio, n_boot):
    """Paired a − b on the main metric, or None if one of the systems is missing."""
    rows_a, rows_b = systems.get((task, a, ratio)), systems.get((task, b, ratio))
    if not rows_a or not rows_b:
        return None
    by_id = {r["id"]: r for r in rows_b}
    common = [r for r in rows_a if r["id"] in by_id]
    m = MAIN_METRIC[task]
    return paired_test([r[m] for r in common], [by_id[r["id"]][m] for r in common],
                       [r["cluster"] for r in common], n_boot)


def fmt(t):
    return f"{t['diff']:+.4f} [{t['ci_low']:+.4f}, {t['ci_high']:+.4f}], n = {t['n']}"


def mean(systems, task, method, ratio):
    rows = systems.get((task, method, ratio))
    return float(np.mean([r[MAIN_METRIC[task]] for r in rows])) if rows else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--teacher", default="teacher")
    ap.add_argument("--syllable-method", default="llmlingua2_syl")
    ap.add_argument("--word-method", default="llmlingua2_wordpool")
    ap.add_argument("--token-methods", nargs="+",
                    default=["llmlingua2_syl", "llmlingua2_wordpool", "selective_context"])
    ap.add_argument("--sentence-methods", nargs="+", default=["tfidf_sent", "ppl_sent"])
    ap.add_argument("--n-boot", type=int, default=10_000)
    args = ap.parse_args()

    rows = [r for path in args.rows for r in read_jsonl(path)]
    readers = sorted({r["reader"] for r in rows})
    if len(readers) > 1:
        raise SystemExit(f"one reader per call, got {readers}")
    systems = defaultdict(list)
    for r in rows:
        systems[r["task"], r["method"], r["ratio"]].append(r)
    tasks = sorted({r["task"] for r in rows})
    methods = {r["method"] for r in rows}
    print(f"reader {readers[0]}; tasks {', '.join(tasks)}\n")

    if args.teacher in methods:
        print(f"## G4: {args.teacher} vs truncation at 1/3 (pass if it wins on >= 2 tasks)\n")
        wins = 0
        for task in tasks:
            t = compare(systems, task, args.teacher, "truncation", 0.333, args.n_boot)
            if t is None:
                continue
            wins += t["diff"] > 0
            print(f"- {task}: {MAIN_METRIC[task]} {mean(systems, task, args.teacher, 0.333):.4f} vs "
                  f"{mean(systems, task, 'truncation', 0.333):.4f}, diff {fmt(t)}")
        print(f"-> wins on {wins} tasks -> {'PASS' if wins >= 2 else 'FAIL'}\n")
        for method in sorted(m for m in methods if m.startswith(args.teacher)):
            ratios = [r["n_tokens_compressed"] / max(r["budget"], 1)
                      for r in rows if r["method"] == method and r["ratio"] == 0.333]
            print(f"  {method}: tokens/budget at 1/3 = {np.mean(ratios):.3f} "
                  f"(over budget: {np.mean([x > 1 for x in ratios]):.0%})")
        print()

    if args.word_method in methods and args.syllable_method in methods:
        print(f"## G1 (score): {args.word_method} vs {args.syllable_method} (pass if better on >= 1 task x ratio)\n")
        better = 0
        for task, ratio in sorted({(k[0], k[2]) for k in systems if k[1] == args.word_method}):
            t = compare(systems, task, args.word_method, args.syllable_method, ratio, args.n_boot)
            if t is not None:
                better += t["diff"] > 0
                print(f"- {task} @ {ratio}: diff {fmt(t)}")
        print(f"-> better on {better} task x ratio -> {'PASS' if better >= 1 else 'FAIL'}\n")

    if "probe_negation" in methods and "none" in methods and "vinli" in tasks:
        print("## G2 (NFR): flip rate of probe_negation on ViNLI pairs whose premise it changed (>= 5 points)\n")
        base = {r["id"]: r["prediction"] for r in systems["vinli", "none", 1.0]}
        probe = {r["id"]: r for (task, m, _), group in systems.items()
                 if task == "vinli" and m == "probe_negation" for r in group}
        changed = [r for r in probe.values() if r["n_tokens_compressed"] < r["n_tokens_original"]]
        nfr = np.mean([r["prediction"] != base[r["id"]] for r in changed]) if changed else float("nan")
        print(f"- changed pairs: {len(changed)} of {len(probe)}, NFR = {nfr:.1%} "
              f"-> {'PASS' if nfr >= 0.05 else 'FAIL'}\n")

    sentence = [m for m in args.sentence_methods if m in methods]
    token = [m for m in args.token_methods if m in methods]
    if sentence and token:
        print("## G3: best token-level vs best sentence-level at 1/5 (within 5 points on a majority of tasks)\n")
        ok, n = 0, 0
        for task in tasks:
            best_tok = max(((mean(systems, task, m, 0.2), m) for m in token if (task, m, 0.2) in systems),
                           default=None)
            best_sent = max(((mean(systems, task, m, 0.2), m) for m in sentence if (task, m, 0.2) in systems),
                            default=None)
            if best_tok is None or best_sent is None:
                continue
            n += 1
            gap = best_tok[0] - best_sent[0]
            ok += gap >= -0.05
            print(f"- {task}: {best_tok[1]} {best_tok[0]:.4f} vs {best_sent[1]} {best_sent[0]:.4f}, "
                  f"gap {gap:+.4f}")
        print(f"-> within 5 points on {ok} of {n} tasks -> {'PASS' if ok > n / 2 else 'FAIL'}")


if __name__ == "__main__":
    main()
