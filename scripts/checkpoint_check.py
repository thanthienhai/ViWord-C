"""Reader sanity check for trained checkpoints (DATN §3.3 / task C3).

The check is deliberately separate from checkpoint selection by BCE loss. At the
registered 1/3 budget on the 200-example DEV slice of each gate task, every ViWord-C
checkpoint (`viword_s*`) must beat **truncation** and the **lexical-prior control**
(`lexprior`, "đối chứng tra từ" of DATN §4.2: the teacher keep rate of the word type,
no context). A checkpoint passes a baseline when it beats it on a majority of the dev
tasks, the same majority rule as gate G4: VietNews alone has a strong lead bias (the
teacher itself loses to truncation there), so a single-task requirement would test the
news genre, not the encoder. LLMLingua-2-vi checkpoints are reported for reference only:
they are a control arm and must not gate the evaluation. ``stage_eval`` consumes the
JSON PASS marker and refuses to run without it.

Two more baselines are reported on the same rows and never gate (added 2026-10-22, before
any official run): the published LLMLingua-2 (`llmlingua2`, trained on English labels), and
the teacher's own compression cut to the budget (`teacher`, the G4 input of
results/teacher_dev). They answer whether distillation beats the package at all and
whether the teacher itself does, i.e. how much room the encoder has.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from viword.data import read_jsonl
from viword.eval import MAIN_METRIC


BASELINES = ("truncation", "lexprior")
REFERENCE_BASELINES = ("llmlingua2", "teacher")  # reported, never gating
GATED_PREFIX = "viword_s"
REFERENCE_PREFIXES = ("llmlingua2vi_s",)


def _mean(rows: list[dict], metric: str) -> float | None:
    values = [row[metric] for row in rows if metric in row]
    return float(np.mean(values)) if values else None


def summarize(rows: list[dict], ratio: float = 0.333,
              expected_per_task: int = 200) -> dict:
    """Summarize checkpoint-vs-baseline means and return a PASS decision."""
    systems: dict[tuple[str, str, float], list[dict]] = defaultdict(list)
    for row in rows:
        systems[row["task"], row["method"], round(float(row["ratio"]), 3)].append(row)

    tasks = sorted({row["task"] for row in rows})
    methods = {row["method"] for row in rows}
    gated = sorted(m for m in methods if m.startswith(GATED_PREFIX))
    reference = sorted(m for m in methods if m.startswith(REFERENCE_PREFIXES))
    target_ratio = round(ratio, 3)
    comparisons = []
    for task in tasks:
        metric = MAIN_METRIC[task]
        baseline_rows = {b: systems.get((task, b, target_ratio), []) for b in BASELINES + REFERENCE_BASELINES}
        baseline_scores = {b: _mean(rows_b, metric) for b, rows_b in baseline_rows.items()}
        for method in gated + reference:
            checkpoint_rows = systems.get((task, method, target_ratio), [])
            score = _mean(checkpoint_rows, metric)
            comparison = {
                "task": task,
                "method": method,
                "gated": method in gated,
                "metric": metric,
                "n": len({row["id"] for row in checkpoint_rows}),
                "score": score,
            }
            for b in BASELINES + REFERENCE_BASELINES:
                comparison[f"n_{b}"] = len({row["id"] for row in baseline_rows[b]})
                comparison[b] = baseline_scores[b]
                comparison[f"beats_{b}"] = (score is not None and baseline_scores[b] is not None
                                            and score > baseline_scores[b])
            comparison["complete"] = all(comparison[k] == expected_per_task
                                         for k in ["n"] + [f"n_{b}" for b in BASELINES])
            comparisons.append(comparison)

    checkpoints = []
    for method in gated + reference:
        own = [c for c in comparisons if c["method"] == method]
        wins = {b: sum(c[f"beats_{b}"] for c in own) for b in BASELINES}
        reference_wins = {b: sum(c[f"beats_{b}"] for c in own) for b in REFERENCE_BASELINES}
        complete = bool(own) and len(own) == len(tasks) and all(c["complete"] for c in own)
        passed = complete and all(wins[b] > len(tasks) / 2 for b in BASELINES)
        checkpoints.append({"method": method, "gated": method in gated, "complete": complete,
                            "wins": wins, "reference_wins": reference_wins, "n_tasks": len(tasks),
                            "passed": passed})
    passed = bool(gated) and all(c["passed"] for c in checkpoints if c["gated"])
    references = []  # per task: the teacher's own compression against the published LLMLingua-2
    for task in tasks:
        own = next((c for c in comparisons if c["task"] == task), None)
        if own is None:
            continue
        teacher, package = own["teacher"], own["llmlingua2"]
        references.append({"task": task, "metric": own["metric"], "truncation": own["truncation"],
                           "llmlingua2": package, "teacher": teacher,
                           "n_llmlingua2": own["n_llmlingua2"], "n_teacher": own["n_teacher"],
                           "teacher_minus_llmlingua2": None if teacher is None or package is None
                           else teacher - package})
    return {
        "ratio": target_ratio,
        "expected_per_task": expected_per_task,
        "baseline_methods": list(BASELINES),
        "reference_baselines": list(REFERENCE_BASELINES),
        "rule": "every viword_s* beats each baseline on a majority of dev tasks",
        "checkpoint_methods": gated,
        "reference_methods": reference,
        "comparisons": comparisons,
        "checkpoints": checkpoints,
        "references": references,
        "passed": passed,
    }


def _score_text(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def render_report(summary: dict) -> str:
    lines = [
        "# Checkpoint reader sanity check",
        "",
        f"- split: DEV; examples per task: {summary['expected_per_task']}; "
        f"ratio: {summary['ratio']:.3f}; reader: {summary.get('reader', 'unspecified')}",
        f"- rule: {summary['rule']}; LLMLingua-2-vi rows, the published LLMLingua-2 and the teacher "
        "are reference only",
        "",
        "| Task | Checkpoint | Metric | Checkpoint | Truncation | Lexprior | LLMLingua-2 (ref.) | "
        "Teacher (ref.) | n |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for c in summary["comparisons"]:
        lines.append(
            f"| {c['task']} | `{c['method']}` | {c['metric']} | {_score_text(c['score'])} | "
            f"{_score_text(c['truncation'])} | {_score_text(c['lexprior'])} | "
            f"{_score_text(c.get('llmlingua2'))} | {_score_text(c.get('teacher'))} | {c['n']} |"
        )
    lines += ["", "| Checkpoint | gated | wins vs truncation | wins vs lexprior | wins vs LLMLingua-2 (ref.) "
              "| Result |", "|---|---|---|---|---|---|"]
    for c in summary["checkpoints"]:
        result = ("PASS" if c["passed"] else "FAIL") if c["gated"] else "reference"
        package_wins = c.get("reference_wins", {}).get("llmlingua2")
        package_text = "n/a" if package_wins is None else f"{package_wins}/{c['n_tasks']}"
        lines.append(f"| `{c['method']}` | {'yes' if c['gated'] else 'no'} | "
                     f"{c['wins']['truncation']}/{c['n_tasks']} | {c['wins']['lexprior']}/{c['n_tasks']} | "
                     f"{package_text} | {result} |")
    if summary.get("references"):
        lines += ["", "## Reference, not gating: the teacher against the published LLMLingua-2", "",
                  "The teacher row is the teacher's own compression cut to the budget (G4's `trim=1`). If it does "
                  "not beat LLMLingua-2, a student of this teacher has little room to.", "",
                  "| Task | Metric | Truncation | LLMLingua-2 | Teacher | Teacher − LLMLingua-2 | n LLMLingua-2 / teacher |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for r in summary["references"]:
            diff = r["teacher_minus_llmlingua2"]
            lines.append(f"| {r['task']} | {r['metric']} | {_score_text(r['truncation'])} | "
                         f"{_score_text(r['llmlingua2'])} | {_score_text(r['teacher'])} | "
                         f"{'n/a' if diff is None else f'{diff:+.4f}'} | {r['n_llmlingua2']} / {r['n_teacher']} |")
    lines.extend(["", f"## {'PASS' if summary['passed'] else 'FAIL'}", ""])
    return "\n".join(lines)


def verify(path: str, reader: str | None = None) -> None:
    with open(path, encoding="utf-8") as f:
        summary = json.load(f)
    if reader and summary.get("reader") != reader:
        raise SystemExit(
            f"checkpoint sanity check reader mismatch: report={summary.get('reader')!r}, "
            f"current={reader!r}; eval is blocked"
        )
    if not summary.get("passed", False):
        raise SystemExit(f"checkpoint sanity check is FAIL: {path}; eval is blocked")
    print(f"checkpoint sanity check: PASS ({path})")


def main() -> None:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--rows", nargs="+", help="reader-evaluation JSONL files from DEV")
    mode.add_argument("--verify", help="verify a previously written PASS JSON report")
    ap.add_argument("--reader", help="reader identity recorded in/checked against the report")
    ap.add_argument("--ratio", type=float, default=0.333)
    ap.add_argument("--expected-per-task", type=int, default=200)
    ap.add_argument("--out", help="Markdown report path")
    ap.add_argument("--json-out", help="machine-readable report path")
    args = ap.parse_args()

    if args.verify:
        verify(args.verify, args.reader)
        return
    if not args.out or not args.json_out:
        ap.error("--rows requires both --out and --json-out")

    rows = [row for path in args.rows for row in read_jsonl(path)]
    summary = summarize(rows, args.ratio, args.expected_per_task)
    summary["reader"] = args.reader
    report = render_report(summary)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report, encoding="utf-8")
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8")
    print(report, end="")
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
