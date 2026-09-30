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
    index = defaultdict(list)
    for i, c in enumerate(clusters):
        index[c].append(i)
    groups = list(index.values())
    sums = np.array([values[g].sum() for g in groups])
    sizes = np.array([len(g) for g in groups], dtype=float)
    return sums, sizes


def cluster_bootstrap(values, clusters: list[str], n_boot: int = 10_000, seed: int = 0) -> np.ndarray:
    """Bootstrap distribution of the mean of `values`, resampling clusters."""
    values = np.asarray(values, dtype=float)
    sums, sizes = _cluster_sums(values, clusters)
    if len(sums) < 2:
        raise ValueError("cluster bootstrap needs at least 2 clusters; check the cluster ids")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    return sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)


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


def non_inferiority(a, b, clusters: list[str], margin: float, n_boot: int = 10_000,
                    seed: int = 0) -> dict:
    """a is non-inferior to b if the lower 95% (one-sided) bound of a − b is above −margin."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    boot = cluster_bootstrap(diff, clusters, n_boot, seed)
    lower = float(np.percentile(boot, 5))
    return {"diff": float(diff.mean()), "lower_95": lower, "margin": margin,
            "non_inferior": lower > -margin}


def minimum_detectable_effect(diff, clusters: list[str], alpha: float = 0.05, power: float = 0.8,
                              n_boot: int = 2_000, seed: int = 0) -> float:
    """Normal approximation: MDE = (z_{1-α} + z_{power}) · SE, SE from the cluster bootstrap.
    Run it on a pilot (dev) difference to decide the test-set size before testing."""
    se = float(np.std(cluster_bootstrap(diff, clusters, n_boot, seed)))
    z = {0.05: 1.645, 0.025: 1.960, 0.01: 2.326}[alpha] + {0.8: 0.842, 0.9: 1.282}[power]
    return z * se


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm–Bonferroni adjusted p-values (family-wise error control)."""
    order = sorted(p_values, key=p_values.get)
    adjusted, running = {}, 0.0
    for rank, key in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * p_values[key]))
        adjusted[key] = running
    return adjusted


def mcnemar_exact(flips_a_only: int, flips_b_only: int) -> float:
    """Two-sided exact McNemar test on discordant pairs (used for NFR, DATN §4.3)."""
    n, k = flips_a_only + flips_b_only, min(flips_a_only, flips_b_only)
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)
