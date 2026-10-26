"""Report of a ViBidLQA evaluation (direction B): F1 (main), answer recall and EM by method and ratio, and paired
contrasts with a cluster bootstrap (cluster = gold article).

  python scripts/analyze_vibid.py --eval results/b/eval --proposed unitsel_util \
      --baselines lead,qsent,qunit,llmlingua2,llmlingua2w,llmlingua2w_q5,rerank,longllmlingua > results/b/dev.md

Every `<stem>*.jsonl` result file in --eval (not the .compressed ones) is read, for the stems of --modes. Methods
named `<x>_s<k>` are seeds of `<x>`: their rows are averaged per question before any mean or contrast. The best
baseline is chosen per mode and ratio by its mean F1, then the proposed method is compared with it (H1) and with no
compression (H1b). Rows written before answer recall existed get it from their prediction.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from viword.eval import answer_recall
from viword.stats import paired_test_stratified

SEED = re.compile(r"^(.+)_s([0-9]+)$")
REFERENCE = ("none", "no_context")
METRICS = ("f1", "recall", "em")


def name(method: str) -> str:
    s = SEED.match(method)
    return s.group(1) if s else method


def load(eval_dir: Path, stem: str, answers: dict) -> dict:
    """{(method, ratio): {id: {metric: value averaged over seeds, "cluster": ..., "n_seeds": k}}}."""
    raw = defaultdict(lambda: defaultdict(list))
    for path in sorted(eval_dir.glob(f"{stem}*.jsonl")):
        if path.name.endswith(".compressed.jsonl") or not re.fullmatch(rf"{re.escape(stem)}(_[a-z0-9]+)?\.jsonl",
                                                                        path.name):
            continue
        for r in map(json.loads, open(path, encoding="utf-8")):
            if "recall" not in r:
                r["recall"] = answer_recall(r["prediction"], answers[r["id"]])
            m = name(r["method"])
            raw[m, 1.0 if m in REFERENCE else round(r["ratio"], 3)][r["id"]].append(r)
    out = {}
    for key, by_id in raw.items():
        out[key] = {i: {**{k: float(np.mean([r[k] for r in rs])) for k in METRICS},
                        "cluster": rs[0]["cluster"], "n_seeds": len(rs)} for i, rs in by_id.items()}
    return out


def cell(g, m, ra):
    return g.get((m, 1.0 if m in REFERENCE else ra), {})


def contrast(g, a, ra_a, b, ra_b, metric="f1") -> str:
    x, y = cell(g, a, ra_a), cell(g, b, ra_b)
    keys = sorted(x.keys() & y.keys())
    if not keys:
        return "–"
    t = paired_test_stratified([x[k][metric] for k in keys], [y[k][metric] for k in keys],
                               [x[k]["cluster"] for k in keys], [0] * len(keys), 2000)
    return f"{100 * t['diff']:+.1f} [{100 * t['ci_low']:+.1f}, {100 * t['ci_high']:+.1f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--tasks", default="data/tasks")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--modes", default="random,retrieved")
    ap.add_argument("--proposed", default="unitsel_util")
    ap.add_argument("--baselines", default="lead,qsent,qunit,llmlingua2,llmlingua2w,llmlingua2w_q5,rerank,"
                                           "longllmlingua")
    args = ap.parse_args()
    eval_dir, baselines = Path(args.eval), args.baselines.split(",")
    stems = {"random": f"vibid_rag_random_{args.split}", "retrieved": f"vibid_rag_{args.split}"}
    answers = {r["id"]: r["answers"][0]
               for r in map(json.loads, open(f"{args.tasks}/vibid_{args.split}.jsonl", encoding="utf-8"))}
    oracle = load(eval_dir, f"vibid_rag_oracle_{args.split}", answers)
    for mode in args.modes.split(","):
        g = load(eval_dir, stems[mode], answers)
        if not g:
            continue
        ratios = sorted({ra for (_, ra) in g if ra != 1.0}, reverse=True)
        methods = list(dict.fromkeys(m for m, _ in sorted(g, key=lambda k: (k[0] not in REFERENCE, k[0]))))
        n = len({i for d in g.values() for i in d})
        print(f"\n## {mode} contexts, {args.split} ({n} questions): F1 / answer recall / EM\n")
        print("| method | seeds | " + " | ".join(f"@{ra}" for ra in ratios) + " |\n|---|---|"
              + "---|" * len(ratios))
        if oracle:
            d = cell(oracle, "none", 1.0)
            print("| oracle (gold units alone) | 1 | " + " / ".join(f"{np.mean([r[k] for r in d.values()]):.3f}"
                                                              for k in METRICS) + " |" + " |" * (len(ratios) - 1))
        for m in methods:
            ras = [1.0] if m in REFERENCE else ratios
            seeds = max(r["n_seeds"] for ra in ras for r in cell(g, m, ra).values()) if cell(g, m, ras[0]) else 0
            cells = [" / ".join(f"{np.mean([r[k] for r in cell(g, m, ra).values()]):.3f}" for k in METRICS)
                     if cell(g, m, ra) else "" for ra in ras]
            print(f"| {m} | {seeds} | " + " | ".join(cells + [""] * (len(ratios) - len(cells))) + " |")

        print(f"\n### {mode}: contrasts (points, 95% CI by gold article)\n")
        print("| contrast | metric | " + " | ".join(f"@{ra}" for ra in ratios) + " |\n|---|---|"
              + "---|" * len(ratios))
        for metric in ("f1", "recall"):
            print(f"| none − no_context | {metric} | {contrast(g, 'none', 1.0, 'no_context', 1.0, metric)} |"
                  + " |" * (len(ratios) - 1))
            best = {}
            for ra in ratios:
                present = [b for b in baselines if cell(g, b, ra)]
                best[ra] = max(present, key=lambda b: np.mean([r["f1"] for r in cell(g, b, ra).values()]),
                               default=None)
            if any(cell(g, args.proposed, ra) for ra in ratios):
                print(f"| {args.proposed} − best baseline (H1) | {metric} | "
                      + " | ".join(f"{contrast(g, args.proposed, ra, best[ra], ra, metric)} ({best[ra]})"
                                   if best[ra] else "–" for ra in ratios) + " |")
                print(f"| {args.proposed} − none (H1b) | {metric} | "
                      + " | ".join(contrast(g, args.proposed, ra, "none", 1.0, metric) for ra in ratios) + " |")
            for m in methods:
                if m not in REFERENCE and m != args.proposed:
                    print(f"| {m} − none | {metric} | "
                          + " | ".join(contrast(g, m, ra, "none", 1.0, metric) for ra in ratios) + " |")


if __name__ == "__main__":
    main()
