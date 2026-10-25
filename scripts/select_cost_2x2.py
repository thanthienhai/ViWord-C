"""Select the cost-aware ranking of the 2×2 cells on dev (item d), outside the registered H1 family.

The registered cells rank units by s(w) alone (α = 0). Word units are coarser and dearer than
syllables, so at 1/5 a whole-word drop can waste budget and (iv) − (i) would mix a semantic
gain with a granularity loss. Here every cell is re-run with the same cost-aware rule:

  a0p5   rank by s / c_T^0.5        a1   rank by s / c_T (greedy knapsack)
  dp     exact 0/1 knapsack by dynamic programming (max Σ s under Σ c_T ≤ B)

Rule (fixed before any dev run of it): one rule for all four cells, so the cells still differ
only in their unit; it maximizes the mean dev gain over α = 0 of the main metric (points),
averaged over the four cells, the dev tasks, the three ratios and the seeds; α = 0 itself is a
candidate (gain 0); ties go to the simpler rule (a0 < a0p5 < a1 < dp). The per-cell gains are
reported too. The selected rule gives the descriptive `*_ca` cells of the eval stage; the
registered H1 test stays on the α = 0 cells.

Rows: evaluate.py output of stage_select_cost_2x2, method names `<cell>_s<seed>` (α = 0) and
`<cell>_ca<tag>_s<seed>`.

Usage:
  python scripts/select_cost_2x2.py --rows results/cost2x2_dev/*.jsonl --out config/cost2x2_selected.json \
      --env config/cost2x2_selected.env --report results/cost2x2_selection.md
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from viword.data import read_jsonl
from viword.eval import MAIN_METRIC

CELLS = ("llmlingua2vi", "llmlingua2vi_wordpool", "viword_syl", "viword")
RULES = {"a0": "alpha=0", "a0p5": "alpha=0.5", "a1": "alpha=1", "dp": "exact=1"}
ORDER = list(RULES)
RATIOS = (0.5, 0.333, 0.2)
NAME_RE = re.compile(r"^(?P<cell>.+?)(?:_ca(?P<tag>a0p5|a1|dp))?_s(?P<seed>[0-9]+)$")


def _means(rows: list[dict]) -> dict:
    sums = defaultdict(list)
    for r in rows:
        m = NAME_RE.match(r["method"])
        if not m or m.group("cell") not in CELLS:
            continue
        key = (m.group("cell"), m.group("tag") or "a0", int(m.group("seed")), r["task"], round(float(r["ratio"]), 3))
        sums[key].append(r[MAIN_METRIC[r["task"]]])
    return {k: sum(v) / len(v) for k, v in sums.items()}


def select(rows: list[dict]) -> dict:
    means = _means(rows)
    base = {k: v for k, v in means.items() if k[1] == "a0"}
    if not base:
        raise ValueError("no α = 0 rows of the 2×2 cells")
    seeds = sorted({k[2] for k in base})
    tasks = sorted({k[3] for k in base})
    per_rule, per_cell = {"a0": 0.0}, {}
    for tag in ORDER[1:]:
        gains, cell_gains = [], defaultdict(list)
        for cell in CELLS:
            for seed in seeds:
                for task in tasks:
                    for ratio in RATIOS:
                        b, c = base.get((cell, "a0", seed, task, ratio)), means.get((cell, tag, seed, task, ratio))
                        if b is None or c is None:
                            raise ValueError(f"missing dev rows: {cell} {tag} seed {seed} {task} @{ratio}")
                        gains.append(100 * (c - b))
                        cell_gains[cell].append(100 * (c - b))
        per_rule[tag] = sum(gains) / len(gains)
        per_cell[tag] = {cell: sum(g) / len(g) for cell, g in cell_gains.items()}
    best = max(ORDER, key=lambda t: (round(per_rule[t], 9), -ORDER.index(t)))
    return {"schema": 1, "rule": "max mean dev gain over α=0 (points; 4 cells × tasks × ratios × seeds); "
                                 "tie: simpler rule (a0 < a0p5 < a1 < dp)",
            "tasks": tasks, "ratios": list(RATIOS), "seeds": seeds, "selected": best, "spec": RULES[best],
            "gain_points": per_rule, "gain_points_per_cell": per_cell}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--out", default="config/cost2x2_selected.json")
    ap.add_argument("--env", default="config/cost2x2_selected.env")
    ap.add_argument("--report", default="results/cost2x2_selection.md")
    args = ap.parse_args()
    rows = [r for path in args.rows if not path.endswith(".compressed.jsonl") for r in read_jsonl(path)]
    readers = {r.get("reader") for r in rows}
    if len(readers) != 1:
        raise SystemExit(f"select_cost_2x2: one dev reader expected, got {sorted(map(str, readers))}")
    decision = {**select(rows), "reader": readers.pop()}
    for path in (args.out, args.env, args.report):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.env).write_text(f"COST2X2_TAG={decision['selected']}\nCOST2X2_SPEC={decision['spec']}\n", encoding="utf-8")
    lines = ["# Cost-aware ranking of the 2×2 cells (dev)", "", f"Reader: `{decision['reader']}`; rule: {decision['rule']}",
             "", "| rule | mean gain (points) | " + " | ".join(CELLS) + " |", "|---|---:|" + "---:|" * len(CELLS)]
    for tag in ORDER:
        cells = decision["gain_points_per_cell"].get(tag, {c: 0.0 for c in CELLS})
        lines.append(f"| {tag} ({RULES[tag]}) | {decision['gain_points'][tag]:+.3f} | "
                     + " | ".join(f"{cells[c]:+.3f}" for c in CELLS) + " |")
    lines += ["", f"Selected: **{decision['selected']}** (`{decision['spec']}`)", ""]
    Path(args.report).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
