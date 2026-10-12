"""Gap between greedy selection (+ fill pass) and the exact 0/1 knapsack (DATN §3.5, E4).

For every example and ratio, both selections maximize Σ s(w) under Σ c_T(w) <= B with the
same scores and costs as `ScoredCompressor`; the report gives the mean and worst ratio
objective(greedy) / objective(DP) before the real-length trim. Hard priority is not part
of the DP objective, so run it on configurations without `protect=hard|force`.

Usage:
  python scripts/greedy_gap.py --task vinli --data data/tasks/vinli_dev.jsonl \
      --segmented data/segmented/vinli_dev.jsonl --tokenizer Qwen/Qwen2.5-7B-Instruct \
      --method scored:model=runs/viword_s0,unit=word,alpha=0.5 --limit 200 --out results/greedy_gap.json
"""
import argparse
import json
import os

import numpy as np

from viword.compressor import ScoredCompressor, unit_costs
from viword.data import load_contexts, load_task
from viword.eval import context_budget
from viword.methods import build_compressor
from viword.select import TokenCounter, exact_select, greedy_select


def objective_gap(scores, costs, budget: int, alpha: float) -> tuple[float, float]:
    """(Σ s of the greedy + fill selection, Σ s of the exact DP selection)."""
    greedy = greedy_select(scores, costs, budget, alpha=alpha)
    exact = exact_select(scores, costs, budget)
    return (float(sum(s for s, k in zip(scores, greedy) if k)),
            float(sum(s for s, k in zip(scores, exact) if k)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--segmented", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--method", required=True, help="a scored:/lexprior: spec")
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.5, 1 / 3, 0.2])
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    compressor = build_compressor(args.method)
    if not isinstance(compressor, ScoredCompressor):
        raise SystemExit("greedy_gap.py needs a scored:/lexprior: method")
    cfg = compressor.config
    if cfg.protect in {"hard", "force", "force_ctx"}:
        raise SystemExit("hard priority is not part of the DP objective; use protect=none|soft")
    counter = TokenCounter(AutoTokenizer.from_pretrained(args.tokenizer))
    contexts = load_contexts(load_task(args.task, args.data, args.limit), args.segmented)
    report = {"method": args.method, "task": args.task, "n_examples": len(contexts), "ratios": {}}
    for ratio in args.ratios:
        shares = []
        for ctx in contexts:
            words, scores, _, _ = compressor.unit_scores(ctx)
            _, budget = context_budget(ctx.text, ratio, counter)
            greedy, exact = objective_gap(scores, unit_costs(words, counter, cfg.unit, cfg.cost),
                                          budget, cfg.alpha)
            if exact > 0:
                shares.append(greedy / exact)
        report["ratios"][f"{ratio:.3f}"] = {"mean_greedy_over_dp": float(np.mean(shares)),
                                            "min_greedy_over_dp": float(np.min(shares)), "n": len(shares)}
        print(f"ratio {ratio:.3f}: greedy/DP objective mean {np.mean(shares):.4f}, worst {np.min(shares):.4f}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
