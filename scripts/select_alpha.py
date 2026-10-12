"""Select the H3 cost exponent α on dev only (DATN §3.5; 8-page version).

Grid: α ∈ {0.25, 0.5, 0.75, 1} for c_T(w) (`cost=tokens`) and, separately, for the
length control n_syl(w) (`cost=syllables`), each on top of the dev-selected π
configuration; α = 0 is that π arm itself (`viword_pi_s*`). Rows come from
`stage_select_alpha` on VietNews and ViNLI dev with the primary reader.

Rule (fixed before the test run): for each cost, α* maximizes the mean gain over α = 0
of the main metric (ROUGE-L or accuracy, in points), averaged over the two dev tasks,
the three ratios and the three seeds; α = 0 itself is a candidate (gain 0), and ties go
to the smaller α. α* is then fixed for every reader, task and tokenizer.

Usage:
  python scripts/select_alpha.py --rows results/alpha_dev/*.jsonl --out config/alpha_selected.json \
      --env config/alpha_selected.env --report results/alpha_selection.md
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from viword.data import read_jsonl
from viword.eval import MAIN_METRIC

TASKS = ("vietnews", "vinli")
RATIOS = (0.5, 0.333, 0.2)
CANDIDATE_RE = re.compile(r"^viword_h3_a(?P<alpha>[0-9p]+)_c(?P<cost>tokens|syllables)_s(?P<seed>[0-9]+)$")
BASE_RE = re.compile(r"^viword_pi_s(?P<seed>[0-9]+)$")


def _mean_by(rows, method, task, ratio):
    values = [r[MAIN_METRIC[task]] for r in rows
              if r["method"] == method and r["task"] == task and round(float(r["ratio"]), 3) == ratio]
    return sum(values) / len(values) if values else None


def select(rows: list[dict]) -> dict:
    seeds = sorted({int(m.group("seed")) for r in rows if (m := BASE_RE.match(r["method"]))})
    if not seeds:
        raise ValueError("no viword_pi_s* (α = 0) rows")
    grid = defaultdict(set)
    for r in rows:
        if m := CANDIDATE_RE.match(r["method"]):
            grid[m.group("cost"), float(m.group("alpha").replace("p", "."))].add(int(m.group("seed")))
    if not grid:
        raise ValueError("no viword_h3_* candidate rows")
    candidates = []
    for (cost, alpha), cand_seeds in sorted(grid.items()):
        if sorted(cand_seeds) != seeds:
            raise ValueError(f"incomplete seeds for cost={cost} α={alpha}: {sorted(cand_seeds)} vs {seeds}")
        gains = []
        for task in TASKS:
            for ratio in RATIOS:
                for s in seeds:
                    tag = f"{alpha:g}".replace(".", "p")
                    cand = _mean_by(rows, f"viword_h3_a{tag}_c{cost}_s{s}", task, ratio)
                    base = _mean_by(rows, f"viword_pi_s{s}", task, ratio)
                    if cand is None or base is None:
                        raise ValueError(f"missing dev rows: {task} @{ratio} seed {s} (cost={cost}, α={alpha})")
                    gains.append(100 * (cand - base))
        candidates.append({"cost": cost, "alpha": alpha, "gain_points": sum(gains) / len(gains)})
    selected = {}
    for cost in ("tokens", "syllables"):
        pool = [c for c in candidates if c["cost"] == cost] + [{"cost": cost, "alpha": 0.0, "gain_points": 0.0}]
        best = max(pool, key=lambda c: (round(c["gain_points"], 9), -c["alpha"]))
        selected[cost] = best["alpha"]
    return {"schema": 1, "rule": "max mean dev gain over α=0 (points; tasks × ratios × seeds); tie: smaller α",
            "tasks": list(TASKS), "ratios": list(RATIOS), "seeds": seeds,
            "alpha_tokens": selected["tokens"], "alpha_syllables": selected["syllables"],
            "candidates": candidates}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--out", default="config/alpha_selected.json")
    ap.add_argument("--env", default="config/alpha_selected.env")
    ap.add_argument("--report", default="results/alpha_selection.md")
    args = ap.parse_args()
    rows = [r for path in args.rows if not path.endswith(".compressed.jsonl") for r in read_jsonl(path)]
    readers = {r.get("reader") for r in rows}
    if len(readers) != 1:
        raise SystemExit(f"select_alpha: one dev reader expected, got {sorted(map(str, readers))}")
    decision = {**select(rows), "reader": readers.pop()}
    for path in (args.out, args.env, args.report):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.env).write_text(f"ALPHA={decision['alpha_tokens']}\nALPHA_SYL={decision['alpha_syllables']}\n",
                              encoding="utf-8")
    lines = ["# α selection on dev (H3)", "", f"Reader: `{decision['reader']}`; rule: {decision['rule']}", "",
             "| cost | α | mean gain over α=0 (points) |", "|---|---:|---:|"]
    lines += [f"| {c['cost']} | {c['alpha']:g} | {c['gain_points']:+.3f} |" for c in decision["candidates"]]
    lines += ["", f"α* (c_T) = {decision['alpha_tokens']:g}; α* (n_syl control) = {decision['alpha_syllables']:g}", ""]
    Path(args.report).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
