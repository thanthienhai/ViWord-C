"""Select the linguistic-protection configuration on dev (DATN §2.3).

The input consists of ``scripts/evaluate.py`` rows for ViNLI and VietNews dev.
For every ``viword_pi_*`` candidate, this script:

* compares VietNews ROUGE-L with the same-seed ``viword_s*`` (π = none);
* computes ViNLI NFR as prediction flips against ``none`` at ratio 1.0;
* keeps candidates whose ROUGE-L loss is at most 0.5 points;
* selects the lowest-NFR candidate, with narrower protection as the tie-break.

It writes both a machine-readable JSON decision and a shell environment file.  The
pipeline sources the latter before test evaluation, so a stale/default DELTA or TIERS
cannot silently replace the dev-selected configuration.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from viword.data import read_jsonl


MARGIN_POINTS = 0.5
TASKS = ("vinli", "vietnews")
RATIOS = (0.5, 0.333, 0.2)
CANDIDATE_RE = re.compile(
    r"^viword_pi_(?P<mode>soft|hard)_(?P<tier>T1(?:_T2(?:_T3)?)?)"
    r"_d(?P<delta>[0-9]+(?:p[0-9]+)?)_s(?P<seed>.+)$"
)


def parse_candidate(method: str) -> dict | None:
    match = CANDIDATE_RE.fullmatch(method)
    if not match:
        return None
    delta = float(match["delta"].replace("p", "."))
    tier_tag = match["tier"]
    tiers = tuple(tier_tag.split("_"))
    return {
        "method": method,
        "mode": match["mode"],
        "tier_tag": tier_tag,
        "tiers": tiers,
        "delta": delta,
        "seed": match["seed"],
    }


def _ratio(value) -> float:
    return round(float(value), 3)


def _index(rows: list[dict]) -> dict[tuple[str, str, float, str], dict]:
    indexed = {}
    for row in rows:
        key = (row["task"], row["method"], _ratio(row["ratio"]), row["id"])
        if key in indexed:
            raise ValueError(f"duplicate result row: {key}")
        indexed[key] = row
    return indexed


def _rows_for(index, task: str, method: str, ratio: float) -> dict[str, dict]:
    return {
        example_id: row
        for (row_task, row_method, row_ratio, example_id), row in index.items()
        if row_task == task and row_method == method and row_ratio == _ratio(ratio)
    }


def _require_common(left: dict[str, dict], right: dict[str, dict], label: str) -> list[str]:
    ids = sorted(set(left) & set(right))
    if not ids:
        raise ValueError(f"no paired rows for {label}")
    if set(left) != set(right):
        missing_left = sorted(set(right) - set(left))[:3]
        missing_right = sorted(set(left) - set(right))[:3]
        raise ValueError(
            f"misaligned rows for {label}; missing candidate={missing_left}, missing reference={missing_right}"
        )
    return ids


def _candidate_metrics(candidate: dict, index: dict) -> dict:
    method = candidate["method"]
    seed = candidate["seed"]
    base_method = f"viword_s{seed}"
    none_method = "none"
    rouge_losses = []
    nfr_flips = []
    n_rouge_pairs = 0
    n_nfr_pairs = 0

    for ratio in RATIOS:
        cand_news = _rows_for(index, "vietnews", method, ratio)
        base_news = _rows_for(index, "vietnews", base_method, ratio)
        ids = _require_common(cand_news, base_news, f"{method} vs {base_method} VietNews @{ratio}")
        candidate_mean = sum(cand_news[i]["rougeL"] for i in ids) / len(ids)
        baseline_mean = sum(base_news[i]["rougeL"] for i in ids) / len(ids)
        # Store loss in ROUGE-L points, matching the DATN margin 0.5 (not 0.005).
        rouge_losses.append(100.0 * (baseline_mean - candidate_mean))
        n_rouge_pairs += len(ids)

        cand_nli = _rows_for(index, "vinli", method, ratio)
        uncompressed = _rows_for(index, "vinli", none_method, 1.0)
        ids = _require_common(cand_nli, uncompressed, f"{method} vs none ViNLI @{ratio}")
        nfr_flips.extend(cand_nli[i]["prediction"] != uncompressed[i]["prediction"] for i in ids)
        n_nfr_pairs += len(ids)

    if not rouge_losses or not nfr_flips:
        raise ValueError(f"candidate has no complete dev metrics: {method}")
    return {
        **candidate,
        "tiers": list(candidate["tiers"]),
        "rouge_loss_by_ratio": dict(zip(RATIOS, rouge_losses)),
        "rouge_loss_points": sum(rouge_losses) / len(rouge_losses),
        "nfr": sum(nfr_flips) / len(nfr_flips),
        "n_rouge_pairs": n_rouge_pairs,
        "n_nfr_pairs": n_nfr_pairs,
        "feasible": max(rouge_losses) <= MARGIN_POINTS + 1e-12,
    }


def _aggregate_candidates(candidates: list[dict], index: dict) -> dict:
    """Evaluate one grid point and aggregate its metrics across all seeds."""
    per_seed = [_candidate_metrics(candidate, index) for candidate in candidates]
    n_rouge = sum(row["n_rouge_pairs"] for row in per_seed)
    n_nfr = sum(row["n_nfr_pairs"] for row in per_seed)
    rouge_loss = sum(row["rouge_loss_points"] * row["n_rouge_pairs"] for row in per_seed) / n_rouge
    # DATN §2.3: ROUGE-L may not drop more than the margin at ANY ratio (the test-time
    # non-inferiority check is per ratio), so feasibility uses the worst ratio, averaged
    # over seeds, not the mean over ratios.
    loss_by_ratio = {ratio: sum(row["rouge_loss_by_ratio"][ratio] for row in per_seed) / len(per_seed)
                     for ratio in RATIOS}
    nfr = sum(row["nfr"] * row["n_nfr_pairs"] for row in per_seed) / n_nfr
    first = per_seed[0]
    return {
        "mode": first["mode"],
        "tiers": first["tiers"],
        "tier_tag": first["tier_tag"],
        "delta": first["delta"],
        "seeds": sorted(row["seed"] for row in per_seed),
        "rouge_loss_points": rouge_loss,
        "rouge_loss_by_ratio": loss_by_ratio,
        "max_rouge_loss_points": max(loss_by_ratio.values()),
        "nfr": nfr,
        "n_rouge_pairs": n_rouge,
        "n_nfr_pairs": n_nfr,
        "feasible": max(loss_by_ratio.values()) <= MARGIN_POINTS + 1e-12,
    }


def select(rows: list[dict], margin_points: float = MARGIN_POINTS) -> dict:
    readers = sorted({row.get("reader") for row in rows if row.get("reader") is not None})
    if len(readers) > 1:
        raise ValueError(f"selection must use one reader, got {readers}")
    index = _index(rows)
    by_config = defaultdict(list)
    for method in sorted({row["method"] for row in rows}):
        parsed = parse_candidate(method)
        if parsed is not None:
            key = parsed["mode"], parsed["tier_tag"], parsed["delta"]
            by_config[key].append(parsed)
    if not by_config:
        raise ValueError("no viword_pi_* candidate rows found")
    expected_seeds = {candidate["seed"] for group in by_config.values() for candidate in group}
    candidates = []
    for group in by_config.values():
        observed_seeds = {candidate["seed"] for candidate in group}
        if observed_seeds != expected_seeds:
            raise ValueError(
                f"incomplete seed grid for {group[0]['mode']} {group[0]['tier_tag']} {group[0]['delta']}: "
                f"expected {sorted(expected_seeds)}, got {sorted(observed_seeds)}"
            )
        candidates.append(_aggregate_candidates(group, index))
    for candidate in candidates:
        candidate["feasible"] = candidate["max_rouge_loss_points"] <= margin_points + 1e-12

    feasible = [row for row in candidates if row["feasible"]]
    if not feasible:
        raise ValueError(f"no π configuration satisfies ROUGE-L loss <= {margin_points} points at every ratio")

    def sort_key(row):
        # DATN tie-break: protect fewer words. The tier grid is nested (T1 ⊂ T1+T2 ⊂
        # T1+T2+T3) and soft/hard share P for the same tiers, so fewer tiers is exactly
        # fewer protected words. Remaining terms make the choice deterministic.
        return (row["nfr"], len(row["tiers"]), 0 if row["mode"] == "soft" else 1,
                row["delta"], row["tier_tag"])

    selected = min(feasible, key=sort_key)
    return {
        "schema": 1,
        "reader": readers[0] if readers else None,
        "selection_rule": ("minimize ViNLI NFR subject to VietNews ROUGE-L loss <= margin at every ratio; "
                           "tie: fewer protected words (= fewer tiers in the nested grid)"),
        "rouge_reference": "viword_s* (same seed, π=none)",
        "nfr_reference": "none (uncompressed, ratio=1.0)",
        "rouge_margin_points": margin_points,
        "tasks": list(TASKS),
        "ratios": list(RATIOS),
        "selected": {
            "mode": selected["mode"],
            "tiers": list(selected["tiers"]),
            "delta": selected["delta"],
            "seeds": selected["seeds"],
            "rouge_loss_points": selected["rouge_loss_points"],
            "max_rouge_loss_points": selected["max_rouge_loss_points"],
            "nfr": selected["nfr"],
            "n_rouge_pairs": selected["n_rouge_pairs"],
            "n_nfr_pairs": selected["n_nfr_pairs"],
        },
        "candidates": [
            {key: value for key, value in row.items() if key not in {"method", "seed"}}
            for row in sorted(candidates, key=sort_key)
        ],
    }


def render_report(decision: dict) -> str:
    selected = decision["selected"]
    lines = [
        "# π configuration selection on dev",
        "",
        f"Reader: `{decision['reader'] or 'not recorded'}`  ",
        f"Rule: {decision['selection_rule']}  ",
        f"ROUGE-L reference: `{decision['rouge_reference']}`; margin: **{decision['rouge_margin_points']:.3f} points**  ",
        f"NFR reference: `{decision['nfr_reference']}`",
        "",
        "| mode | tiers | δ | ROUGE-L loss, mean (points) | ROUGE-L loss, worst ratio | NFR | ROUGE pairs | NFR pairs | feasible |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in decision["candidates"]:
        marker = " **← selected**" if (
            row["mode"] == selected["mode"] and row["tiers"] == selected["tiers"]
            and row["delta"] == selected["delta"]
        ) else ""
        lines.append(
            f"| {row['mode']} | {'+'.join(row['tiers'])} | {row['delta']:.3f} | "
            f"{row['rouge_loss_points']:+.3f} | {row['max_rouge_loss_points']:+.3f} | {row['nfr']:.4f} | "
            f"{row['n_rouge_pairs']} | "
            f"{row['n_nfr_pairs']} | {'yes' if row['feasible'] else 'no'}{marker} |"
        )
    lines += [
        "",
        "Only the selected configuration may be used for the π arm on test; rerun this stage if dev data, reader, "
        "checkpoint, or protection rules change.",
        "",
    ]
    return "\n".join(lines)


def write_outputs(decision: dict, out_json: str, out_report: str, out_env: str) -> None:
    selected = decision["selected"]
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_report).parent.mkdir(parents=True, exist_ok=True)
    Path(out_env).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(out_report).write_text(render_report(decision), encoding="utf-8")
    tiers = "+".join(selected["tiers"])
    env = (
        "# Generated by scripts/select_pi.py; commit this file before stage eval.\n"
        f"PI_MODE={selected['mode']}\n"
        f"PI_TIERS='{tiers}'\n"
        f"PI_DELTA={selected['delta']:.3f}\n"
    )
    Path(out_env).write_text(env, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", nargs="+", required=True)
    parser.add_argument("--out", default="config/pi_selected.json")
    parser.add_argument("--report", default="results/pi_selection.md")
    parser.add_argument("--env", default="config/pi_selected.env")
    parser.add_argument("--margin", type=float, default=MARGIN_POINTS)
    args = parser.parse_args()
    try:
        rows = [row for path in args.rows for row in read_jsonl(path)]
        decision = select(rows, margin_points=args.margin)
        write_outputs(decision, args.out, args.report, args.env)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"select_pi: {exc}") from exc
    print(render_report(decision))


if __name__ == "__main__":
    main()
