"""Statistics for the comparisons in DATN §4.3.

All bootstrap procedures resample *clusters* (source documents), because several examples
can share one source (ViNLI premises, ViMMRC/Belebele passages). Resampling examples
instead would give confidence intervals that are too narrow.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


def _cluster_sums(values: np.ndarray, clusters: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return one sum and one row-count weight per source cluster."""
    if len(values) != len(clusters):
        raise ValueError(f"values/clusters length mismatch: {len(values)} != {len(clusters)}")
    if not clusters:
        raise ValueError("cluster bootstrap needs at least 2 clusters; got no rows")
    index = defaultdict(list)
    for i, c in enumerate(clusters):
        index[c].append(i)
    groups = list(index.values())
    sums = np.array([values[g].sum() for g in groups])
    sizes = np.array([len(g) for g in groups], dtype=float)
    return sums, sizes


def cluster_bootstrap(values, clusters: list[str], n_boot: int = 10_000, seed: int = 0) -> np.ndarray:
    """Bootstrap a row-level mean while resampling source clusters.

    A replicate draws source-cluster IDs with replacement, not individual rows. Every
    selected cluster contributes all of its rows, so the numerator and denominator use
    the cluster's sum and row-count weight. This preserves the example-level estimand
    while accounting for within-premise/passage/article dependence.
    """
    values = np.asarray(values, dtype=float)
    sums, sizes = _cluster_sums(values, clusters)
    if len(sums) < 2:
        raise ValueError("cluster bootstrap needs at least 2 clusters; check the cluster ids")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    return sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)


def stratified_cluster_bootstrap(values, clusters: list[str], strata: list[str | int | None],
                                 n_boot: int = 10_000, seed: int = 0) -> np.ndarray:
    """Bootstrap a mean over a crossed seed × source-cluster design (DATN §4.3).

    The three training seeds are evaluated on the *same* examples, so a source cluster
    (premise, passage, article) appears under every seed and its difficulty is shared
    across seeds. Each replicate therefore draws seeds with replacement and **one** set
    of cluster ids with replacement, and applies that same cluster draw to every drawn
    seed. Drawing clusters independently per seed would treat the seeds as independent
    samples of sources and underestimate the standard error (the LACC bootstrap lesson).
    All rows of a drawn (seed, cluster) cell are kept, so the estimand stays the
    row-level mean.
    """
    values = np.asarray(values, dtype=float)
    if len(values) != len(clusters) or len(values) != len(strata):
        raise ValueError("values, clusters and strata must have the same length")
    if not strata:
        raise ValueError("stratified bootstrap needs at least one row")
    stratum_ids = list(dict.fromkeys(strata))
    cluster_ids = list(dict.fromkeys(clusters))
    if len(cluster_ids) < 2:
        raise ValueError("cluster bootstrap needs at least 2 clusters; check the cluster ids")
    s_index = {s: k for k, s in enumerate(stratum_ids)}
    c_index = {c: k for k, c in enumerate(cluster_ids)}
    sums = np.zeros((len(stratum_ids), len(cluster_ids)))
    sizes = np.zeros((len(stratum_ids), len(cluster_ids)))
    for v, c, s in zip(values, clusters, strata):
        sums[s_index[s], c_index[c]] += v
        sizes[s_index[s], c_index[c]] += 1

    rng = np.random.default_rng(seed)
    n_s, n_c = len(stratum_ids), len(cluster_ids)
    out = np.empty(n_boot, dtype=float)
    chunk = max(1, min(n_boot, 2_000_000 // max(n_c, 1)))
    for start in range(0, n_boot, chunk):
        stop = min(n_boot, start + chunk)
        k = stop - start
        # multiplicity of every seed and every cluster in each replicate
        seed_w = np.zeros((k, n_s))
        np.add.at(seed_w, (np.repeat(np.arange(k), n_s), rng.integers(0, n_s, size=k * n_s)), 1)
        clus_w = np.zeros((k, n_c))
        np.add.at(clus_w, (np.repeat(np.arange(k), n_c), rng.integers(0, n_c, size=k * n_c)), 1)
        numerator = np.einsum("bs,sc,bc->b", seed_w, sums, clus_w)
        denominator = np.einsum("bs,sc,bc->b", seed_w, sizes, clus_w)
        out[start:stop] = numerator / denominator
    return out


def paired_test(a, b, clusters: list[str], n_boot: int = 10_000, seed: int = 0) -> dict:
    """Paired comparison of system a vs b on the same examples.

    Returns the mean difference a − b, its 95% CI, and a one-sided bootstrap p-value for
    H1: a > b (fraction of bootstrap means ≤ 0).
    """
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    boot = cluster_bootstrap(diff, clusters, n_boot, seed)
    return {
        "diff": float(diff.mean()),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "p_one_sided": float((boot <= 0).mean()),
        "n": len(diff),
        "n_clusters": len(set(clusters)),
    }


def paired_test_stratified(a, b, clusters: list[str], strata: list[str | int | None],
                           n_boot: int = 10_000, seed: int = 0) -> dict:
    """Paired test with the seed-then-cluster bootstrap used for the 3-seed report."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    boot = stratified_cluster_bootstrap(diff, clusters, strata, n_boot, seed)
    return {
        "diff": float(diff.mean()),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "p_one_sided": float((boot <= 0).mean()),
        "n": len(diff),
        "n_clusters": len(set(clusters)),
    }


def non_inferiority(a, b, clusters: list[str], margin: float, n_boot: int = 10_000,
                    seed: int = 0) -> dict:
    """a is non-inferior to b if the lower 95% (one-sided) bound of a − b is above −margin."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    boot = cluster_bootstrap(diff, clusters, n_boot, seed)
    lower = float(np.percentile(boot, 5))
    return {"diff": float(diff.mean()), "lower_95": lower, "margin": margin,
            "non_inferior": lower > -margin}


def non_inferiority_stratified(a, b, clusters: list[str], strata: list[str | int | None],
                               margin: float, n_boot: int = 10_000, seed: int = 0) -> dict:
    """Non-inferiority test with the seed-then-cluster bootstrap."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    boot = stratified_cluster_bootstrap(diff, clusters, strata, n_boot, seed)
    lower = float(np.percentile(boot, 5))
    return {"diff": float(diff.mean()), "lower_95": lower, "margin": margin,
            "non_inferior": lower > -margin}


def minimum_detectable_effect(diff, clusters: list[str], alpha: float = 0.05, power: float = 0.8,
                              n_boot: int = 2_000, seed: int = 0,
                              strata: list[str | int | None] | None = None) -> float:
    """Normal-approximation MDE from a dev bootstrap.

    When ``strata`` is supplied, the bootstrap samples strata (the three training
    seeds) and then source clusters within each stratum, matching the main analysis.
    Without it, the original cluster-only bootstrap is retained for compatibility.
    """
    boot = (stratified_cluster_bootstrap(diff, clusters, strata, n_boot, seed)
            if strata is not None else cluster_bootstrap(diff, clusters, n_boot, seed))
    se = float(np.std(boot))
    z = {0.05: 1.645, 0.025: 1.960, 0.01: 2.326}[alpha] + {0.8: 0.842, 0.9: 1.282}[power]
    return z * se


def cluster_bootstrap_slope(x, y, clusters: list[str], n_boot: int = 10_000, seed: int = 0) -> dict:
    """OLS slope of y on x with a cluster-bootstrap 95% CI (DATN §2.3, H1 mechanism:
    per-example score gain regressed on the per-example CBR of the syllable system).
    Rows of every drawn cluster are kept together, so examples of one source (and the
    seeds of one example) are not treated as independent."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) != len(y) or len(x) != len(clusters):
        raise ValueError("x, y and clusters must have the same length")

    def slope(xs, ys):
        vx = np.var(xs)
        return float(np.cov(xs, ys, bias=True)[0, 1] / vx) if vx > 0 else float("nan")

    groups = defaultdict(list)
    for i, c in enumerate(clusters):
        groups[c].append(i)
    members = list(groups.values())
    if len(members) < 2:
        raise ValueError("cluster bootstrap needs at least 2 clusters; check the cluster ids")
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
        boot.append(slope(x[idx], y[idx]))
    boot = np.array(boot)
    boot = boot[~np.isnan(boot)]
    return {"slope": slope(x, y), "ci_low": float(np.percentile(boot, 2.5)),
            "ci_high": float(np.percentile(boot, 97.5)), "n": len(x), "n_clusters": len(members)}


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm–Bonferroni adjusted p-values (family-wise error control)."""
    order = sorted(p_values, key=p_values.get)
    adjusted, running = {}, 0.0
    for rank, key in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * p_values[key]))
        adjusted[key] = running
    return adjusted


def mcnemar_exact(flips_a_only: int, flips_b_only: int) -> float:
    """Two-sided exact McNemar test on discordant pairs.

    Valid only for independent pairs (one seed, one hypothesis per premise). With
    several hypotheses per ViNLI premise or several seeds, use `flip_rate_test`."""
    n, k = flips_a_only + flips_b_only, min(flips_a_only, flips_b_only)
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def mcnemar_exact_one_sided(flips_a_only: int, flips_b_only: int) -> float:
    """One-sided exact McNemar for H: system a flips less often than b.
    p = P(X <= flips_a_only), X ~ Binomial(discordant pairs, 1/2)."""
    n = flips_a_only + flips_b_only
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(flips_a_only + 1)) / 2 ** n


def flip_rate_test(flips_a, flips_b, clusters: list[str], strata: list | None = None,
                   n_boot: int = 10_000, seed: int = 0) -> dict:
    """Paired one-sided test that system a has a lower flip rate (NFR) than b.

    The per-example difference flip_a − flip_b is bootstrapped over source clusters
    (premises), and over seeds when `strata` is given, so pairs sharing a premise or
    a ViNLI example evaluated under three seeds are not counted as independent
    (an exact McNemar on pooled discordant pairs would be). p = share of bootstrap
    means ≥ 0."""
    diff = np.asarray(flips_a, dtype=float) - np.asarray(flips_b, dtype=float)
    boot = (stratified_cluster_bootstrap(diff, clusters, strata, n_boot, seed)
            if strata is not None else cluster_bootstrap(diff, clusters, n_boot, seed))
    return {
        "diff": float(diff.mean()),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "p_one_sided": float((boot >= 0).mean()),
        "n": len(diff),
        "n_clusters": len(set(clusters)),
    }
