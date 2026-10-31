"""Confidence intervals and paired tests over the same evaluation questions.

* CIs: percentile bootstrap over questions, and a database-clustered bootstrap that resamples
  whole Spider databases. Questions sharing a schema aren't independent, and the claim being made
  is generalisation to unseen schemas, so the clustered CI is the more honest one (and wider).
* p-values: paired sign-flip permutation test. Under H0 (no difference) each question's paired
  difference is equally likely to have either sign, so flipping signs gives a proper null
  distribution. The clustered version flips whole databases at once. With binary outcomes and
  one seed, the question-level test is a Monte Carlo exact McNemar test.
* contrast(): any linear combination of systems, e.g. a - b (paired difference) or the 2x2
  interaction (a - b) - (c - d), with the same CIs and tests.
"""
from __future__ import annotations

import numpy as np

N_BOOT = 10_000


def _cluster_arrays(ids, values, clusters):
    keys = sorted({clusters[i] for i in ids})
    index = {k: j for j, k in enumerate(keys)}
    sums, counts = np.zeros(len(keys)), np.zeros(len(keys))
    for i, v in zip(ids, values):
        j = index[clusters[i]]
        sums[j] += v
        counts[j] += 1
    return sums, counts


def _boot_means(v: np.ndarray, rng, n_boot: int) -> np.ndarray:
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    return v[idx].mean(axis=1)


def _cluster_boot_means(sums, counts, rng, n_boot: int) -> np.ndarray:
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    return sums[idx].sum(axis=1) / counts[idx].sum(axis=1)


def _ci(means, alpha):
    return [float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))]


def bootstrap_ci(values, n_boot: int = N_BOOT, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """(mean, lower, upper) percentile bootstrap CI of the mean over questions."""
    v = np.asarray(list(values), dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    lo, hi = _ci(_boot_means(v, np.random.default_rng(seed), n_boot), alpha)
    return float(v.mean()), lo, hi


def accuracy_ci(per_q: dict, clusters: dict | None = None, n_boot=N_BOOT, alpha=0.05, seed=0) -> dict:
    ids = sorted(per_q)
    v = np.array([per_q[i] for i in ids], dtype=float)
    rng = np.random.default_rng(seed)
    out = {"n": len(ids), "mean": float(v.mean()), "ci": _ci(_boot_means(v, rng, n_boot), alpha)}
    if clusters:
        s, c = _cluster_arrays(ids, v, clusters)
        out["ci_clustered"] = _ci(_cluster_boot_means(s, c, rng, n_boot), alpha)
        out["n_clusters"] = len(s)
    return out


def _sign_flip_p(values: np.ndarray, rng, n_perm: int) -> float:
    """Two-sided p for H0: mean(values) = 0, flipping the sign of each entry independently."""
    obs = abs(values.mean())
    if obs == 0:
        return 1.0
    signs = rng.integers(0, 2, size=(n_perm, len(values)), dtype=np.int8) * 2 - 1
    null = np.abs(signs @ values) / len(values)
    return float((np.sum(null >= obs - 1e-12) + 1) / (n_perm + 1))


def _cluster_sign_flip_p(sums: np.ndarray, total_n: float, rng, n_perm: int) -> float:
    obs = abs(sums.sum() / total_n)
    if obs == 0:
        return 1.0
    signs = rng.integers(0, 2, size=(n_perm, len(sums)), dtype=np.int8) * 2 - 1
    null = np.abs(signs @ sums) / total_n
    return float((np.sum(null >= obs - 1e-12) + 1) / (n_perm + 1))


def contrast(systems: list[dict], weights: list[float], clusters: dict | None = None,
             n_boot: int = N_BOOT, alpha: float = 0.05, seed: int = 0) -> dict:
    """Per-question linear contrast sum_k w_k * score_k(q) over questions all systems answered."""
    ids = sorted(set(systems[0]).intersection(*systems[1:]))
    if not ids:
        return {"n": 0}
    d = np.array([sum(w * s[i] for w, s in zip(weights, systems)) for i in ids], dtype=float)
    rng = np.random.default_rng(seed)
    out = {"n": len(ids), "diff": float(d.mean()), "ci": _ci(_boot_means(d, rng, n_boot), alpha),
           "p_value": _sign_flip_p(d, rng, n_boot),
           "a_better": int((d > 0).sum()), "b_better": int((d < 0).sum())}
    if clusters:
        s, c = _cluster_arrays(ids, d, clusters)
        out["ci_clustered"] = _ci(_cluster_boot_means(s, c, rng, n_boot), alpha)
        out["p_value_clustered"] = _cluster_sign_flip_p(s, c.sum(), rng, n_boot)
        out["n_clusters"] = len(s)
    return out


def paired_test(a: dict, b: dict, clusters: dict | None = None, **kw) -> dict:
    """A vs B on the same questions. a, b: {question_id: score in [0, 1]}."""
    ids = sorted(set(a) & set(b))
    out = contrast([a, b], [1, -1], clusters, **kw)
    if ids:
        out["a"] = float(np.mean([a[i] for i in ids]))
        out["b"] = float(np.mean([b[i] for i in ids]))
    return out


def seed_average(per_seed: list[dict]) -> dict:
    """Average per-question scores across replicates (questions present in every replicate)."""
    if not per_seed:
        return {}
    ids = set(per_seed[0]).intersection(*per_seed[1:])
    return {i: float(np.mean([s[i] for s in per_seed])) for i in ids}


def mean_std(xs) -> tuple[float | None, float | None]:
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    return float(np.mean(xs)), (float(np.std(xs, ddof=1)) if len(xs) > 1 else None)
