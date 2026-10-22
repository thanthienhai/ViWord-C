"""Compute the preregistered minimum detectable effects on DEV only (DATN §4.3/J4).

The pilot uses the same seed × cluster bootstrap as the three-seed analysis.  H1 is
the registered H1 contrast (ViWord-C minus LLMLingua-2-vi, DATN §2.3/§4.3).  H2 reports
the NFR contrast on ViNLI and the ROUGE-L contrast on VietNews for the dev-selected π
configuration.

The dev pilot has fewer source clusters than the test sets, so every MDE is also
projected to the test size: MDE_test = MDE_dev · sqrt(clusters_dev / clusters_test)
(the standard error of a cluster mean shrinks with the square root of the number of
clusters).  The gate compares the non-inferiority margin with the *test* MDE, because
the margin is applied on test.  With --exclude the test size is that of the primary analysis, which
leaves out the test rows readers scored before the test run (DATN §4.3, amendment of 2026-10-22).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

# Keep the standalone CLI usable outside run_pipeline.sh, which normally exports
# PYTHONPATH=. before invoking repository scripts.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze import aggregate_seed_rows, paired, row_key
from viword.data import read_jsonl
from viword.heldout import excluded_ids, load_exclusions
from viword.segment_meta import sha256_path

H1_TARGET, H1_REFERENCE = "viword", "llmlingua2vi"
from viword.stats import minimum_detectable_effect

H1_TASKS = ("vimmrc", "belebele")
RATIOS = (0.5, 0.333, 0.2)


def _ratio(value: float) -> float:
    return round(float(value), 3)


def _group(rows: list[dict]) -> dict[tuple[str, str, float], list[dict]]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["task"], row["method"], _ratio(row["ratio"])].append(row)
    return groups


def _seed_sd(diff: list[float], rows: list[dict]) -> tuple[float, float | None]:
    by_seed = defaultdict(list)
    for value, row in zip(diff, rows):
        by_seed[row["_seed"]].append(value)
    means = [float(np.mean(values)) for values in by_seed.values()]
    return float(np.mean(means)), float(np.std(means, ddof=1)) if len(means) > 1 else None


def _project(result: dict, test_clusters: dict[str, int]) -> dict:
    """Add the MDE projected to the test-set cluster count of the task."""
    n_test = test_clusters.get(result["task"])
    if not n_test:
        raise ValueError(f"no test cluster count for {result['task']}; pass --test-data {result['task']}=PATH")
    result["test_clusters"] = n_test
    result["mde_test"] = result["mde"] * (result["n_clusters"] / n_test) ** 0.5
    return result


def _effect(label: str, task: str, ratio: float, left: list[dict], right: list[dict],
            metric: str, *, scale: float, n_boot: int, margin_points: float | None = None) -> dict:
    left, right = paired(left, right)
    if not left:
        raise ValueError(f"no paired rows for {label} {task} @{ratio}")
    if len(left) != len(right):
        raise ValueError(f"misaligned rows for {label} {task} @{ratio}")
    diff = [(float(a[metric]) - float(b[metric])) * scale for a, b in zip(left, right)]
    clusters = [row["cluster"] for row in left]
    strata = [row["_seed"] for row in left]
    mde = minimum_detectable_effect(diff, clusters, n_boot=n_boot, strata=strata)
    observed, seed_sd = _seed_sd(diff, left)
    result = {
        "label": label,
        "task": task,
        "ratio": ratio,
        "metric": metric,
        "scale": scale,
        "n": len(diff),
        "n_clusters": len(set(clusters)),
        "observed_effect": observed,
        "mde": mde,
        "seed_sd": seed_sd,
    }
    if margin_points is not None:
        result["margin_points"] = margin_points
        result["margin_gt_seed_sd"] = seed_sd is not None and margin_points > seed_sd
    return result


def _nfr_effect(task: str, ratio: float, pi_rows: list[dict], base_rows: list[dict],
                none_rows: list[dict], n_boot: int) -> dict:
    pi = {row_key(row): row for row in pi_rows}
    base = {row_key(row): row for row in base_rows}
    none = {row["id"]: row for row in none_rows}
    keys = sorted(set(pi) & set(base))
    if not keys:
        raise ValueError(f"no paired NFR rows for vinli @{ratio}")
    left, right = [], []
    for key in keys:
        if pi[key]["id"] not in none:
            raise ValueError(f"missing uncompressed ViNLI row for {pi[key]['id']}")
        uncompressed = none[pi[key]["id"]]
        left.append({**pi[key], "_mde_value": float(pi[key]["prediction"] != uncompressed["prediction"])})
        right.append({**base[key], "_mde_value": float(base[key]["prediction"] != uncompressed["prediction"])})
    diff = [a["_mde_value"] - b["_mde_value"] for a, b in zip(left, right)]
    clusters = [row["cluster"] for row in left]
    strata = [row["_seed"] for row in left]
    mde = minimum_detectable_effect(diff, clusters, n_boot=n_boot, strata=strata)
    observed, seed_sd = _seed_sd(diff, left)
    return {
        "label": "π NFR − π=none NFR, both vs uncompressed",
        "task": task,
        "ratio": ratio,
        "metric": "nfr",
        "scale": 1.0,
        "n": len(diff),
        "n_clusters": len(set(clusters)),
        "observed_effect": observed,
        "mde": mde,
        "seed_sd": seed_sd,
    }


def count_test_clusters(specs: list[str], exclusions: dict | None = None) -> dict[str, int]:
    """Parse TASK=PATH specs and count distinct source clusters in each test file, without the rows
    readers scored before the test run when `exclusions` is given (the primary analysis drops them)."""
    counts = {}
    for spec in specs:
        task, _, path = spec.partition("=")
        if not task or not path:
            raise ValueError(f"invalid --test-data {spec!r}; expected TASK=PATH")
        skip = excluded_ids(exclusions, path) if exclusions else set()
        counts[task] = len({row["cluster"] for row in read_jsonl(path) if row["id"] not in skip})
    return counts


def compute(rows: list[dict], seeds: list[int], reader: str | None = None,
            n_boot: int = 2_000, margin_points: float = 0.5,
            test_clusters: dict[str, int] | None = None) -> dict:
    readers = sorted({row.get("reader") for row in rows if row.get("reader") is not None})
    if reader is not None:
        rows = [row for row in rows if row.get("reader") == reader]
        readers = [reader]
    if len(readers) != 1:
        raise ValueError(f"MDE requires exactly one dev reader, got {readers}")
    rows = aggregate_seed_rows(rows, seeds)
    systems = _group(rows)
    reader = readers[0]

    test_clusters = test_clusters or {}
    available_h1 = [
        task for task in H1_TASKS
        if all((task, method, ratio) in systems
               for method in (H1_TARGET, H1_REFERENCE) for ratio in RATIOS)
    ]
    missing_h1 = [task for task in H1_TASKS if task not in available_h1]
    if not available_h1:
        raise ValueError("no H1 task has complete DEV rows; expected ViMMRC and/or Belebele")
    h1 = []
    for task in available_h1:
        for ratio in RATIOS:
            h1.append(_project(_effect(
                "ViWord-C − LLMLingua-2-vi (registered H1 contrast)", task, ratio,
                systems[task, H1_TARGET, ratio], systems[task, H1_REFERENCE, ratio],
                "accuracy", scale=1.0, n_boot=n_boot,
            ), test_clusters))

    h2_rouge = []
    for ratio in RATIOS:
        row = _project(_effect(
            "π − π=none", "vietnews", ratio,
            systems["vietnews", "viword_pi", ratio], systems["vietnews", "viword", ratio],
            "rougeL", scale=100.0, n_boot=n_boot, margin_points=margin_points,
        ), test_clusters)
        row["margin_gt_mde"] = margin_points > row["mde_test"]
        h2_rouge.append(row)

    h2_nfr = []
    for ratio in RATIOS:
        h2_nfr.append(_project(_nfr_effect(
            "vinli", ratio,
            systems["vinli", "viword_pi", ratio], systems["vinli", "viword", ratio],
            systems["vinli", "none", 1.0], n_boot,
        ), test_clusters))

    rouge_checks = {
        "margin_points": margin_points,
        "max_mde_points": max(row["mde_test"] for row in h2_rouge),
        "max_seed_sd_points": max(row["seed_sd"] for row in h2_rouge if row["seed_sd"] is not None),
        "margin_gt_all_mde": all(row["margin_gt_mde"] for row in h2_rouge),
        "margin_gt_all_seed_sd": all(row["margin_gt_seed_sd"] for row in h2_rouge),
    }
    return {
        "schema": 1,
        "reader": reader,
        "dev_only": True,
        "seeds": list(seeds),
        "n_boot": n_boot,
        "h1_dev_tasks": available_h1,
        "h1_missing_dev_tasks": missing_h1,
        "h1_contrast": f"{H1_TARGET} − {H1_REFERENCE}",
        "test_clusters": test_clusters,
        "h1": h1,
        "h2": {"rougeL": h2_rouge, "nfr": h2_nfr},
        "rouge_margin_check": rouge_checks,
        "passed": rouge_checks["margin_gt_all_mde"] and rouge_checks["margin_gt_all_seed_sd"],
    }


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def render_report(result: dict) -> str:
    lines = [
        "# MDE pilot on DEV",
        "",
        f"Reader: `{result['reader']}`; seeds: {', '.join(map(str, result['seeds']))}; "
        f"bootstrap: {result['n_boot']} replicates.",
        "Only DEV rows are used. Each replicate resamples seeds, then source clusters within each seed.",
        f"H1 DEV coverage: {', '.join(result['h1_dev_tasks'])}. "
        f"Not available as DEV: {', '.join(result['h1_missing_dev_tasks']) or 'none'}; no TEST rows are substituted.",
        "",
        "## H1",
        "",
        "| task | ratio | contrast | observed | MDE dev | MDE test | seed SD | n | clusters dev/test |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in result["h1"]:
        lines.append(
            f"| {row['task']} | {row['ratio']} | {row['label']} | {row['observed_effect']:.4f} | "
            f"{row['mde']:.4f} | {row['mde_test']:.4f} | {_fmt(row['seed_sd'])} | {row['n']} | "
            f"{row['n_clusters']}/{row['test_clusters']} |"
        )
    lines += [
        "",
        "## H2 — ROUGE-L and NFR",
        "",
        f"ROUGE-L values and MDE are in percentage points; the preregistered non-inferiority margin "
        f"is {result['rouge_margin_check']['margin_points']} points and is compared with the test-size MDE.",
        "",
        "| task | ratio | contrast | observed | MDE dev | MDE test | seed SD | margin > MDE test | margin > seed SD |",
        "|---|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for row in result["h2"]["rougeL"]:
        lines.append(
            f"| {row['task']} | {row['ratio']} | {row['label']} | {row['observed_effect']:.4f} | "
            f"{row['mde']:.4f} | {row['mde_test']:.4f} | {_fmt(row['seed_sd'])} | "
            f"{'PASS' if row['margin_gt_mde'] else 'FAIL'} | "
            f"{'PASS' if row['margin_gt_seed_sd'] else 'FAIL'} |"
        )
    lines += [
        "",
        "| task | ratio | contrast | observed | MDE dev | MDE test | seed SD |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in result["h2"]["nfr"]:
        lines.append(
            f"| {row['task']} | {row['ratio']} | {row['label']} | {row['observed_effect']:.4f} | "
            f"{row['mde']:.4f} | {row['mde_test']:.4f} | {_fmt(row['seed_sd'])} |"
        )
    check = result["rouge_margin_check"]
    lines += [
        "",
        f"ROUGE-L margin check: {check['margin_points']} > max test MDE {check['max_mde_points']:.4f} points = "
        f"{'PASS' if check['margin_gt_all_mde'] else 'FAIL'}; "
        f"{check['margin_points']} > max seed SD {check['max_seed_sd_points']:.4f} points = "
        f"{'PASS' if check['margin_gt_all_seed_sd'] else 'FAIL'}.",
        "",
        f"## Gate: {'PASS' if result['passed'] else 'FAIL'}",
        "",
    ]
    return "\n".join(lines)


def write_outputs(result: dict, out_json: str, report: str) -> None:
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(report).write_text(render_report(result), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", nargs="+", help="DEV evaluate.py JSONL rows")
    parser.add_argument("--out", default="results/mde_dev/report.json")
    parser.add_argument("--report", default="results/mde_dev/report.md")
    parser.add_argument("--reader")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--n-boot", type=int, default=2_000)
    parser.add_argument("--margin-points", type=float, default=0.5)
    parser.add_argument("--test-data", nargs="+", default=[],
                        help="TASK=PATH of the test files, to project the MDE to the test size")
    parser.add_argument("--exclude", default=None,
                        help="config/test_exclusions.json: count test clusters without the rows readers scored "
                             "before the test run, as the primary analysis does; --verify then requires a "
                             "report computed with this file")
    parser.add_argument("--verify", help="verify an existing JSON report before test evaluation")
    args = parser.parse_args()
    if args.verify:
        report = json.loads(Path(args.verify).read_text(encoding="utf-8"))
        if not report.get("dev_only") or not report.get("passed"):
            raise SystemExit(f"mde: refusing test evaluation; MDE report is not PASS: {args.verify}")
        if args.exclude and report.get("test_exclusions_sha256") != sha256_path(args.exclude):
            raise SystemExit(f"mde: report was not projected to the test rows left by {args.exclude}; "
                             "recompute it (run_pipeline.sh mde)")
        reported = report.get("rouge_margin_check", {}).get("margin_points")
        if reported is None or abs(float(reported) - args.margin_points) > 1e-9:
            raise SystemExit(f"mde: report was computed for margin {reported}, but eval uses "
                             f"--margin-points {args.margin_points}; recompute the MDE report")
        if report.get("h1_contrast") != f"{H1_TARGET} − {H1_REFERENCE}":
            raise SystemExit("mde: report predates the registered H1 contrast; recompute it")
        if args.reader and report.get("reader") != args.reader:
            raise SystemExit(f"mde: report was computed with reader {report.get('reader')!r}, "
                             f"expected the primary reader {args.reader!r}")
        print(f"MDE PASS: {args.verify}")
        return
    if not args.rows:
        parser.error("--rows is required unless --verify is used")
    if len(args.seeds) != 3:
        parser.error("MDE requires exactly three --seeds")
    try:
        rows = [row for path in args.rows if not path.endswith(".compressed.jsonl") for row in read_jsonl(path)]
        exclusions = load_exclusions(args.exclude) if args.exclude else None
        result = compute(rows, args.seeds, args.reader, args.n_boot, args.margin_points,
                         count_test_clusters(args.test_data, exclusions))
        if args.exclude:
            result["test_exclusions_sha256"] = sha256_path(args.exclude)
        write_outputs(result, args.out, args.report)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"mde: {exc}") from exc
    print(render_report(result))
    if not result["passed"]:
        raise SystemExit("mde: FAIL; increase the DEV pilot or record a preregistered margin change before eval")


if __name__ == "__main__":
    main()
