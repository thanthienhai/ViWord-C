"""Tables and tests from evaluation rows (DATN §4.3).

  1. mean main metric per task × reader × method × ratio, with a cluster-bootstrap 95% CI;
  2. with --target/--reference: paired one-sided test target > reference per task × ratio;
     Holm is applied only to the registered family cells (H1 defaults to ViMMRC +
     Belebele × {1/2, 1/3, 1/5} for the primary reader). Give the strong simple baselines
     (truncation, lead) as references too: in LACC they tied the learned compressors;
  3. with --factorial: the 2×2 design (labels × decision unit, DATN §2.3), main effects and
     interaction with cluster-bootstrap CIs, so the H1 reading does not rest on eyeballing
     four means;
  4. ViNLI: NFR (label flips vs. the uncompressed prediction), overall, on examples the
     reader got right without compression, and on negated premises (H2). Target vs each
     reference: one-sided paired bootstrap on the flip difference over premises (and seeds),
     Holm within the registered NFR family; one-sided exact McNemar per seed is printed as a
     descriptive check only, because pooled pairs share premises and seeds;
  5. VietNews: non-inferiority of target vs each reference on ROUGE-L (--margin in points).

Usage:
  python scripts/analyze.py --rows results/eval/*.jsonl --target viword \
      --reference llmlingua2vi truncation lead \
      --seed-mode aggregate --seeds 0 1 2 \
      --primary-reader Qwen/Qwen2.5-7B-Instruct \
      --hypothesis H1 \
      --holm-family 'H1:tasks=vimmrc+belebele,ratios=0.5+0.333+0.2,reader=primary,test=metric' \
      --factorial llmlingua2vi llmlingua2vi_wordpool viword_syl viword

  H2: --target viword_pi --reference viword llmlingua2vi_force --hypothesis H2
      (default family H2:tasks=vinli,ratios=0.5+0.333+0.2,reader=primary,test=nfr)
"""
import argparse
from collections import defaultdict
import re

import numpy as np

from viword.data import read_jsonl
from viword.eval import BUDGET_DEVIATION_THRESHOLD, MAIN_METRIC, budget_deviates, tokens_per_budget
from viword.stats import (cluster_bootstrap, cluster_bootstrap_slope, flip_rate_test, holm, mcnemar_exact_one_sided,
                          non_inferiority, non_inferiority_stratified, paired_test,
                          paired_test_stratified, stratified_cluster_bootstrap)

# Registered families (DATN §4.3): Holm runs within one hypothesis family only.
# test=metric: paired main-metric tests; test=nfr: ViNLI flip-rate tests.
DEFAULT_HOLM_FAMILIES = {
    "H1": "H1:tasks=vimmrc+belebele,ratios=0.5+0.333+0.2,reader=primary,test=metric",
    "H2": "H2:tasks=vinli,ratios=0.5+0.333+0.2,reader=primary,test=nfr",
}
DEFAULT_HOLM_FAMILY = DEFAULT_HOLM_FAMILIES["H1"]


def parse_holm_family(spec: str) -> dict:
    """Parse NAME:tasks=a+b,ratios=0.5+0.333,reader=primary."""
    name, separator, fields_text = spec.partition(":")
    if not separator or not name:
        raise ValueError(f"invalid Holm family {spec!r}; expected NAME:key=value,...")
    fields = {}
    for field in fields_text.split(","):
        key, separator, value = field.partition("=")
        if not separator or not value:
            raise ValueError(f"invalid Holm family field {field!r} in {spec!r}")
        fields[key] = value
    tasks = tuple(task for task in fields.get("tasks", "").split("+") if task)
    ratios = tuple(round(float(ratio), 3) for ratio in fields.get("ratios", "").split("+") if ratio)
    reader = fields.get("reader", "primary")
    test = fields.get("test", "metric")
    if not tasks or not ratios:
        raise ValueError(f"Holm family {name!r} needs non-empty tasks and ratios")
    if test not in {"metric", "nfr"}:
        raise ValueError(f"Holm family {name!r}: test must be metric or nfr, got {test!r}")
    return {"name": name, "tasks": tasks, "ratios": ratios, "reader": reader, "test": test}


def holm_family_members(family: dict, reader: str, primary_reader: str | None) -> set[tuple[str, float]]:
    """Return task/ratio cells in a family for one reader."""
    selected_reader = primary_reader if family["reader"] == "primary" else family["reader"]
    if selected_reader not in {"all", reader}:
        return set()
    return {(task, ratio) for task in family["tasks"] for ratio in family["ratios"]}


def adjust_for_holm_family(
    tests: dict,
    family: dict | None,
    reader: str,
    primary_reader: str | None,
) -> tuple[dict, set[tuple[str, float]]]:
    """Apply Holm only to p-values whose task/ratio cell is in one family."""
    family_cells = holm_family_members(family, reader, primary_reader) if family else set()
    if not family_cells:
        return {}, family_cells
    adjusted = holm({
        key: test["p_one_sided"]
        for key, test in tests.items()
        if key in family_cells
    })
    return adjusted, family_cells


SEED_METHOD_RE = re.compile(r"^(?P<base>.+)_s(?P<seed>[0-9]+)$")


def split_seed_method(method: str) -> tuple[str, int | None]:
    """Return the seed-independent name and seed encoded by stage_eval method names."""
    match = SEED_METHOD_RE.fullmatch(method)
    return (match.group("base"), int(match.group("seed"))) if match else (method, None)


def aggregate_seed_rows(rows: list[dict], expected_seeds: list[int]) -> list[dict]:
    """Normalize *_s0/*_s1/*_s2 methods and require complete seed coverage."""
    normalized = []
    coverage = defaultdict(set)
    for row in rows:
        method, seed = split_seed_method(row["method"])
        copied = dict(row)
        copied["method"] = method
        copied["_seed"] = seed
        normalized.append(copied)
        if seed is not None:
            coverage[(row["task"], row["reader"], method, row["ratio"])].add(seed)
    expected = set(expected_seeds)
    incomplete = {
        key: sorted(seeds)
        for key, seeds in coverage.items()
        if seeds != expected
    }
    if incomplete:
        examples = "; ".join(f"{key}: {seeds}" for key, seeds in list(incomplete.items())[:4])
        raise ValueError(f"incomplete seed coverage; expected {sorted(expected)} ({examples})")
    return normalized


def row_key(row: dict):
    """Use (id, seed) for aggregate rows, while preserving old id-only rows."""
    return (row["id"], row.get("_seed")) if "_seed" in row else row["id"]


def by_key(rows, *keys):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    return groups


def paired(rows_a, rows_b):
    """Align two systems on example id; returns (a_rows, b_rows) in the same order."""
    if not any("_seed" in r for r in rows_a + rows_b):
        b = {r["id"]: r for r in rows_b}
        common = [r for r in rows_a if r["id"] in b]
        return common, [b[r["id"]] for r in common]
    by_seed = {(r["id"], r.get("_seed")): r for r in rows_b if r.get("_seed") is not None}
    shared = {r["id"]: r for r in rows_b if r.get("_seed") is None}
    common, matched = [], []
    for row in rows_a:
        other = by_seed.get((row["id"], row.get("_seed")))
        if other is None:
            other = shared.get(row["id"])
        if other is not None:
            common.append(row)
            matched.append(other)
    return common, matched


def ci(t):
    return f"{t['diff']:+.4f} [{t['ci_low']:+.4f}, {t['ci_high']:+.4f}]"


def row_budget_ratio(row):
    """Read explicit adherence, with compatibility for old result rows."""
    if "tokens_per_budget" in row:
        return row["tokens_per_budget"]
    return tokens_per_budget(row["n_tokens_compressed"], row["budget"])


def mean_budget_ratio(rows):
    values = [row_budget_ratio(row) for row in rows]
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def budget_pair(left, right):
    """Format target/reference adherence and flag a >5-point mismatch."""
    left_ratio, right_ratio = mean_budget_ratio(left), mean_budget_ratio(right)
    if left_ratio is None or right_ratio is None:
        return "-"
    gap = abs(left_ratio - right_ratio)
    flag = " ⚠ >5%" if gap > BUDGET_DEVIATION_THRESHOLD else ""
    return f"{left_ratio:.3f}/{right_ratio:.3f} (Δ={gap:.3f}){flag}"


EMPTY_ANSWER_LIMIT = 0.02  # tasks.md I5: parser failures must stay below 2%


def empty_share(rows) -> float | None:
    """Share of rows whose parsed prediction is empty (parser could not read the answer)."""
    values = [row.get("prediction") for row in rows if "prediction" in row]
    return sum(v == "" for v in values) / len(values) if values else None


def seed_summary(rows, metric: str, aggregate: bool) -> tuple[float, float | None]:
    """Return mean and sample SD of per-seed means for a result group."""
    if not aggregate:
        return float(np.mean([row[metric] for row in rows])), None
    per_seed = by_key(rows, "_seed")
    means = [np.mean([row[metric] for row in group]) for group in per_seed.values()]
    return float(np.mean(means)), float(np.std(means, ddof=1)) if len(means) > 1 else None


def bootstrap_mean(rows, metric: str, args) -> np.ndarray:
    values = [row[metric] for row in rows]
    clusters = [row["cluster"] for row in rows]
    if args.seed_mode == "aggregate":
        return stratified_cluster_bootstrap(values, clusters, [row["_seed"] for row in rows], args.n_boot)
    return cluster_bootstrap(values, clusters, args.n_boot)


def paired_result(a, b, metric: str, args) -> dict:
    values_a = [row[metric] for row in a]
    values_b = [row[metric] for row in b]
    clusters = [row["cluster"] for row in a]
    if args.seed_mode == "aggregate":
        return paired_test_stratified(values_a, values_b, clusters,
                                      [row["_seed"] for row in a], args.n_boot)
    return paired_test(values_a, values_b, clusters, args.n_boot)


def non_inferiority_result(a, b, metric: str, margin: float, args) -> dict:
    values_a = [row[metric] for row in a]
    values_b = [row[metric] for row in b]
    clusters = [row["cluster"] for row in a]
    if args.seed_mode == "aggregate":
        return non_inferiority_stratified(values_a, values_b, clusters,
                                          [row["_seed"] for row in a], margin, args.n_boot)
    return non_inferiority(values_a, values_b, clusters, margin, args.n_boot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--target", default=None)
    ap.add_argument("--reference", nargs="+", default=[],
                    help="one or more references; Holm uses the registered family cells")
    ap.add_argument("--primary-reader", default=None,
                    help="reader used by the registered primary hypothesis family")
    ap.add_argument("--holm-family", action="append", default=None,
                    help="NAME:tasks=task+task,ratios=r+r,reader=primary|all|READER,test=metric|nfr; "
                         "repeat per hypothesis")
    ap.add_argument("--hypothesis", default="H1",
                    help="which registered family this run tests (H1, H2, ...); only that family is "
                         "Holm-adjusted, every other p-value is printed raw")
    ap.add_argument("--seed-mode", choices=("independent", "aggregate"), default="independent",
                    help="analyze each method independently, or aggregate *_s0/*_s1/*_s2 by seed")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                    help="expected seed IDs for --seed-mode aggregate (default: 0 1 2)")
    ap.add_argument("--factorial", nargs=4, default=None,
                    metavar=("SYL_LABELS_SYL_UNIT", "SYL_LABELS_WORD_UNIT", "WORD_LABELS_SYL_UNIT",
                             "WORD_LABELS_WORD_UNIT"),
                    help="the four cells of the 2x2 design (DATN §2.3)")
    ap.add_argument("--margin", type=float, default=0.5, help="non-inferiority margin, ROUGE-L points")
    ap.add_argument("--cbr-regression", default=None, metavar="SYLLABLE_SYSTEM",
                    help="H1 mechanism (DATN §2.3): regress the per-example gain target − SYLLABLE_SYSTEM "
                         "on the per-example CBR of SYLLABLE_SYSTEM")
    ap.add_argument("--n-boot", type=int, default=10_000)
    args = ap.parse_args()
    # results/eval/*.jsonl also matches evaluate.py's compression caches, which are not result rows
    rows = [r for path in args.rows if not path.endswith(".compressed.jsonl") for r in read_jsonl(path)]
    if args.seed_mode == "aggregate":
        if len(args.seeds) != 3:
            ap.error("--seed-mode aggregate requires exactly three --seeds")
        try:
            rows = aggregate_seed_rows(rows, args.seeds)
        except ValueError as exc:
            ap.error(str(exc))
    references = args.reference if args.target else []
    readers = sorted({r["reader"] for r in rows})
    family_specs = list(args.holm_family or [])
    if not any(spec.partition(":")[0] == args.hypothesis for spec in family_specs):
        if args.hypothesis not in DEFAULT_HOLM_FAMILIES:
            ap.error(f"no --holm-family registered for --hypothesis {args.hypothesis}")
        family_specs.append(DEFAULT_HOLM_FAMILIES[args.hypothesis])
    try:
        families = {family["name"]: family for family in map(parse_holm_family, family_specs)}
    except ValueError as exc:
        ap.error(str(exc))
    active_family = families[args.hypothesis]
    primary_reader = args.primary_reader or (readers[0] if len(readers) == 1 else None)
    if references and any(family["reader"] == "primary" for family in families.values()) and primary_reader is None:
        ap.error("--primary-reader is required when rows contain multiple readers")
    if references and primary_reader is not None and primary_reader not in readers:
        ap.error(f"--primary-reader {primary_reader!r} not found in rows; readers={readers}")

    print("| task | reader | method | ratio | n | clusters | mean ± seed SD | 95% CI | tokens/budget | >5% "
          "| empty answers |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for (task, reader, method, ratio), group in sorted(by_key(rows, "task", "reader", "method", "ratio").items()):
        metric = MAIN_METRIC[task]
        mean, seed_sd = seed_summary(group, metric, args.seed_mode == "aggregate")
        boot = bootstrap_mean(group, metric, args)
        clusters = [r["cluster"] for r in group]
        adherence = mean_budget_ratio(group)
        adherence_text = f"{adherence:.3f}" if adherence is not None else "-"
        # `none`/`no_context` have no budget to adhere to (B is the full context, or nothing is sent)
        adherence_flag = "⚠" if method not in {"none", "no_context"} and budget_deviates(adherence) else ""
        mean_text = f"{mean:.4f}" if seed_sd is None else f"{mean:.4f} ± {seed_sd:.4f}"
        empty = empty_share(group)
        empty_text = "-" if empty is None else f"{empty:.1%}" + (" ⚠" if empty > EMPTY_ANSWER_LIMIT else "")
        print(f"| {task} | {reader} | {method} | {ratio} | {len(group)} | {len(set(clusters))} | "
              f"{mean_text} | [{np.percentile(boot, 2.5):.4f}, {np.percentile(boot, 97.5):.4f}] | "
              f"{adherence_text} | {adherence_flag} | {empty_text} |")

    systems = by_key(rows, "task", "reader", "method", "ratio")
    uncompressed = by_key([r for r in rows if r["method"] == "none"], "task", "reader")
    print(f"\nHypothesis under test: {args.hypothesis}. Registered Holm families:")
    for family in families.values():
        resolved = primary_reader if family["reader"] == "primary" else family["reader"]
        cells = ", ".join(f"{task}@{ratio:g}" for task in family["tasks"] for ratio in family["ratios"])
        active = " (active)" if family is active_family else ""
        print(f"- {family['name']}{active}: test={family['test']}; reader={resolved}; cells={cells}")

    for reference in references:
        print(f"\n## {args.target} vs {reference} (one-sided, registered Holm family)\n")
        for reader in readers:
            tests = {}
            for (task, rd, method, ratio), group in systems.items():
                ref = systems.get((task, rd, reference, ratio))
                if rd != reader or method != args.target or ref is None:
                    continue
                a, b = paired(group, ref)
                m = MAIN_METRIC[task]
                tests[task, ratio] = paired_result(a, b, m, args)
            if not tests:
                continue
            family = active_family if active_family["test"] == "metric" else None
            adjusted, family_cells = adjust_for_holm_family(tests, family, reader, primary_reader)
            print(f"reader {reader}")
            print("| task | ratio | diff [95% CI] | p | p Holm | tok/budget target/reference |\n"
                  "|---|---|---|---|---|---|")
            for (task, ratio), t in sorted(tests.items()):
                target_rows = systems[task, reader, args.target, ratio]
                reference_rows = systems[task, reader, reference, ratio]
                adjusted_text = f"{adjusted[task, ratio]:.4f}" if (task, ratio) in adjusted else "—"
                print(f"| {task} | {ratio} | {ci(t)} | {t['p_one_sided']:.4f} | "
                      f"{adjusted_text} | {budget_pair(target_rows, reference_rows)} |")

    if args.factorial:
        cell_names = args.factorial
        print("\n## 2x2 design: label unit x decision unit\n")
        print(f"Cells: SS = {cell_names[0]}, SW = {cell_names[1]}, WS = {cell_names[2]}, WW = {cell_names[3]} "
              "(first letter: labels, second: decision unit; S = syllable, W = word).")
        print("Per-example contrasts on the main metric, cluster-bootstrap 95% CI. unit = word − syllable "
              "decision (H1); label = word − syllable labels; interaction = unit effect with word labels − "
              "with syllable labels.\n")
        print("| task | reader | ratio | n | SS / SW / WS / WW | unit | label | interaction | tok/budget SS/SW/WS/WW |")
        print("|---|---|---|---|---|---|---|---|---|")
        for task, reader, _, ratio in sorted(k for k in systems if k[2] == cell_names[0]):
            cells = [{row_key(r): r for r in systems.get((task, reader, m, ratio), [])} for m in cell_names]
            ids = sorted(set.intersection(*(set(c) for c in cells)))
            if not ids:
                continue
            m = MAIN_METRIC[task]
            v = np.array([[c[i][m] for i in ids] for c in cells])
            clusters = [cells[0][i]["cluster"] for i in ids]
            strata = [cells[0][i].get("_seed") for i in ids]
            contrasts = [((v[1] - v[0]) + (v[3] - v[2])) / 2,  # unit
                         ((v[2] - v[0]) + (v[3] - v[1])) / 2,  # label
                         (v[3] - v[2]) - (v[1] - v[0])]  # interaction
            if args.seed_mode == "aggregate":
                cols = [ci(paired_test_stratified(c, np.zeros_like(c), clusters, strata, args.n_boot))
                        for c in contrasts]
            else:
                cols = [ci(paired_test(c, np.zeros_like(c), clusters, args.n_boot)) for c in contrasts]
            means = " / ".join(
                (f"{mean:.3f}" if sd is None else f"{mean:.3f} ± {sd:.3f}")
                for mean, sd in (seed_summary(list(c.values()), m, args.seed_mode == "aggregate")
                                 for c in cells)
            )
            budgets = [mean_budget_ratio(c.values()) for c in cells]
            budget_text = " / ".join("-" if x is None else f"{x:.3f}" for x in budgets)
            if all(x is not None for x in budgets) and max(budgets) - min(budgets) > BUDGET_DEVIATION_THRESHOLD:
                budget_text += " ⚠ >5%"
            print(f"| {task} | {reader} | {ratio} | {len(ids)} | {means} | {' | '.join(cols)} | {budget_text} |")

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
            flips[reader, method, ratio] = {row_key(x): f for x, f in zip(a, flipped)}
            print(f"| {reader} | {method} | {ratio} | {np.mean(flipped):.4f} | "
                  f"{np.mean(correct) if correct else float('nan'):.4f} | "
                  f"{np.mean(neg) if neg else float('nan'):.4f} ({len(neg)}) |")
        for reference in references:
            print(f"\n### NFR: {args.target} < {reference} (one-sided paired bootstrap over premises"
                  f"{' and seeds' if args.seed_mode == 'aggregate' else ''})\n")
            for reader in readers:
                tests, mcnemar_text = {}, {}
                for (rd, method, ratio), fa in sorted(flips.items()):
                    fb = flips.get((rd, reference, ratio))
                    if rd != reader or method != args.target or fb is None:
                        continue
                    keys = sorted(fa.keys() & fb.keys(), key=str)
                    rows_a = {row_key(x): x for x in systems["vinli", reader, method, ratio]}
                    clusters = [rows_a[k]["cluster"] for k in keys]
                    aggregate = args.seed_mode == "aggregate"
                    strata = [rows_a[k]["_seed"] for k in keys] if aggregate else None
                    tests["vinli", ratio] = flip_rate_test([fa[k] for k in keys], [fb[k] for k in keys],
                                                           clusters, strata, args.n_boot)
                    per_seed = []
                    for s in (sorted({k[1] for k in keys}) if aggregate else [None]):
                        sub = [k for k in keys if s is None or k[1] == s]
                        only_a = sum(bool(fa[k] and not fb[k]) for k in sub)
                        only_b = sum(bool(fb[k] and not fa[k]) for k in sub)
                        label = "" if s is None else f"s{s} "
                        per_seed.append(f"{label}{only_a}/{only_b} p={mcnemar_exact_one_sided(only_a, only_b):.3f}")
                    mcnemar_text["vinli", ratio] = "; ".join(per_seed)
                if not tests:
                    continue
                family = active_family if active_family["test"] == "nfr" else None
                adjusted, _ = adjust_for_holm_family(tests, family, reader, primary_reader)
                print(f"reader {reader}")
                print("| ratio | NFR diff [95% CI] | p | p Holm | McNemar per seed, target-only/ref-only "
                      "(descriptive) |\n|---|---|---|---|---|")
                for (task, ratio), t in sorted(tests.items()):
                    adjusted_text = f"{adjusted[task, ratio]:.4f}" if (task, ratio) in adjusted else "—"
                    print(f"| {ratio} | {ci(t)} | {t['p_one_sided']:.4f} | {adjusted_text} | "
                          f"{mcnemar_text[task, ratio]} |")

    for reference in references:
        for (task, reader, method, ratio), group in sorted(systems.items()):
            ref = systems.get((task, reader, reference, ratio))
            if task != "vietnews" or method != args.target or ref is None:
                continue
            a, b = paired(group, ref)
            result = non_inferiority_result(a, b, "rougeL", args.margin / 100, args)
            print(f"\nVietNews non-inferiority vs {reference}, {reader} @ {ratio}: "
                  f"diff {100 * result['diff']:+.2f} lower 95% {100 * result['lower_95']:+.2f} "
                  f"(margin −{args.margin}) -> {'non-inferior' if result['non_inferior'] else 'not shown'}; "
                  f"tok/budget target/reference {budget_pair(a, b)}")

    if args.cbr_regression and args.target:
        print(f"\n## H1 mechanism: gain {args.target} − {args.cbr_regression} regressed on the per-example CBR "
              f"of {args.cbr_regression}\n")
        print("Slope > 0 means the word decision helps more where the syllable system breaks more words. "
              "Cluster bootstrap 95% CI; examples whose compression keeps no multi-syllable word are dropped.\n")
        print("| task | reader | ratio | slope [95% CI] | n | clusters |\n|---|---|---|---|---|---|")
        for (task, reader, method, ratio), group in sorted(systems.items()):
            ref = systems.get((task, reader, args.cbr_regression, ratio))
            if method != args.target or ref is None:
                continue
            a, b = paired(group, ref)
            m = MAIN_METRIC[task]
            pairs = [(y.get("cbr"), x[m] - y[m], x["cluster"]) for x, y in zip(a, b) if y.get("cbr") is not None]
            if len({c for _, _, c in pairs}) < 2:
                continue
            t = cluster_bootstrap_slope([p[0] for p in pairs], [p[1] for p in pairs],
                                        [p[2] for p in pairs], args.n_boot)
            print(f"| {task} | {reader} | {ratio} | {t['slope']:+.4f} [{t['ci_low']:+.4f}, {t['ci_high']:+.4f}] | "
                  f"{t['n']} | {t['n_clusters']} |")


if __name__ == "__main__":
    main()
