"""Statistical helpers for RQ1-RQ3 significance testing.

Primary test: paired difference CI — NOT single-metric CI overlap.

Usage:
    from scripts.stats import paired_diff_ci, prob_of_improvement

    point, lo, hi = paired_diff_ci(rl_scores, pso_scores)
    # Significant RL improvement if lo > 0
    # Significant PSO improvement if hi < 0

    poi = prob_of_improvement(rl_scores, pso_scores)
    # 1.0 means RL > PSO on every sample; 0.5 means indistinguishable
"""
from __future__ import annotations

import numpy as np


def paired_diff_ci(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    reps: int = 5_000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Bootstrap CI on mean(scores_a - scores_b).

    Parameters
    ----------
    scores_a, scores_b : array-like, shape (n,)
        Paired scores — same seed/layout order.
    reps : int
        Bootstrap resamples (5000 recommended).
    ci : float
        Confidence level, default 0.95.
    rng : np.random.Generator or None
        For reproducibility; uses default_rng(0) if None.

    Returns
    -------
    point : float   mean(d) where d = scores_a - scores_b
    ci_lo : float   lower percentile of bootstrap distribution
    ci_hi : float   upper percentile of bootstrap distribution
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"scores_a and scores_b must match shape; got {a.shape} vs {b.shape}")
    d = a - b
    point = float(np.mean(d))
    rng_ = rng if rng is not None else np.random.default_rng(0)
    n = len(d)
    boot = np.array([np.mean(d[rng_.integers(0, n, size=n)]) for _ in range(reps)])
    alpha = 1.0 - ci
    return point, float(np.percentile(boot, 100 * alpha / 2)), float(np.percentile(boot, 100 * (1 - alpha / 2)))


def prob_of_improvement(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> float:
    """P(a > b) estimated via Mann-Whitney U statistic.

    Follows Agarwal et al. 2021 Section 3: U_a / (n_a * n_b).
    Compatible with 1-D score arrays (n_seeds,).
    """
    from scipy.stats import mannwhitneyu
    a = np.asarray(scores_a, dtype=float).ravel()
    b = np.asarray(scores_b, dtype=float).ravel()
    U, _ = mannwhitneyu(a, b, alternative="greater")
    return float(U / (len(a) * len(b)))


if __name__ == "__main__":
    import json, sys
    # CLI: python scripts/stats.py results_a.json results_b.json final_cov
    if len(sys.argv) < 4:
        print("Usage: stats.py <file_a.json> <file_b.json> <metric>")
        sys.exit(1)
    a_data = json.loads(open(sys.argv[1]).read())
    b_data = json.loads(open(sys.argv[2]).read())
    metric = sys.argv[3]
    a_scores = np.array(a_data[f"{metric}_per_layout"])
    b_scores = np.array(b_data[f"{metric}_per_layout"])
    point, lo, hi = paired_diff_ci(a_scores, b_scores)
    poi = prob_of_improvement(a_scores, b_scores)
    print(f"d = {point:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  P(a>b)={poi:.3f}")
    if lo > 0:
        print("Conclusion: a significantly outperforms b (CI excludes 0)")
    elif hi < 0:
        print("Conclusion: b significantly outperforms a (CI excludes 0)")
    else:
        print("Conclusion: inconclusive (CI contains 0)")
