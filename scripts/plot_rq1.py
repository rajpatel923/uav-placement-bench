"""Figure 1: RQ1 representation comparison.

Bar chart: final_cov IQM ± 95% CI for all 5 obs_modes, best RL algo.
Secondary panel: training wall-clock per obs_mode (cost).

Usage:
    python scripts/plot_rq1.py --results results --regime static --out figures/
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
from stats import paired_diff_ci


OBS_MODES = ["flat", "sorted", "cluster_summary", "spatial_grid", "graph"]


def _load_per_layout(results_dir: Path, regime: str, obs_mode: str,
                     metric: str) -> tuple[np.ndarray, list]:
    """Load per-layout final_cov across all seeds for one obs_mode."""
    folder = results_dir / regime
    scores, times = [], []
    if not folder.exists():
        return np.array([]), []
    for algo_dir in sorted(folder.iterdir()):
        for f in sorted(algo_dir.glob("seed*.json")):
            d = json.loads(f.read_text())
            if d.get("obs_mode") == obs_mode:
                v = d.get(f"{metric}_per_layout")
                if v:
                    scores.append(v)
                t = d.get("train_time_s")
                if t:
                    times.append(t)
    if not scores:
        return np.array([]), times
    # shape: (n_seeds, n_layouts)
    min_len = min(len(s) for s in scores)
    arr = np.array([s[:min_len] for s in scores])
    return arr, times


def iqm(scores: np.ndarray) -> float:
    """Inter-quartile mean: mean of scores in [25th, 75th] percentile."""
    q25, q75 = np.percentile(scores, 25), np.percentile(scores, 75)
    subset = scores[(scores >= q25) & (scores <= q75)]
    return float(np.mean(subset)) if len(subset) > 0 else float(np.mean(scores))


def bootstrap_iqm_ci(scores: np.ndarray, reps: int = 2000, ci: float = 0.95,
                     rng: np.random.Generator | None = None) -> tuple[float, float]:
    rng_ = rng or np.random.default_rng(0)
    n = len(scores)
    q25_fix, q75_fix = np.percentile(scores, 25), np.percentile(scores, 75)
    boot_iqms = []
    for _ in range(reps):
        s = scores[rng_.integers(0, n, size=n)]
        subset = s[(s >= q25_fix) & (s <= q75_fix)]
        boot_iqms.append(float(np.mean(subset)) if len(subset) > 0 else float(np.mean(s)))
    alpha = 1.0 - ci
    return (float(np.percentile(boot_iqms, 100 * alpha / 2)),
            float(np.percentile(boot_iqms, 100 * (1 - alpha / 2))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--regime", default="static")
    p.add_argument("--metric", default="final_cov")
    p.add_argument("--out", default="figures")
    args = p.parse_args()
    results = Path(args.results); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(OBS_MODES)))

    iqm_vals, ci_los, ci_his, mean_times, arrays = [], [], [], [], []
    for obs_mode in OBS_MODES:
        arr, times = _load_per_layout(results, args.regime, obs_mode, args.metric)
        arrays.append(arr)
        if arr.size == 0:
            iqm_vals.append(float("nan")); ci_los.append(0); ci_his.append(0)
            mean_times.append(float("nan"))
            continue
        flat = arr.ravel()
        v = iqm(flat)
        lo, hi = bootstrap_iqm_ci(flat)
        iqm_vals.append(v); ci_los.append(v - lo); ci_his.append(hi - v)
        mean_times.append(float(np.mean(times)) / 60.0 if times else float("nan"))

    x = np.arange(len(OBS_MODES))
    bars = ax1.bar(x, iqm_vals, color=colors,
                   yerr=[ci_los, ci_his], capsize=5, error_kw={"linewidth": 1.5})
    ax1.set_xticks(x); ax1.set_xticklabels(OBS_MODES, rotation=15, ha="right")
    ax1.set_ylabel(f"{args.metric} IQM"); ax1.set_ylim(0, 1)
    ax1.set_title(f"(a) RQ1: Representation comparison — {args.regime}")
    ax1.axhline(y=0, color="k", linewidth=0.5)

    # Annotate significance vs. flat baseline using paired_diff_ci
    baseline_arr = arrays[OBS_MODES.index("flat")]
    if baseline_arr.size >= 2:
        for i, (obs_mode, arr) in enumerate(zip(OBS_MODES, arrays)):
            if obs_mode == "flat" or arr.size < 2:
                continue
            n = min(arr.shape[0], baseline_arr.shape[0])
            pt, lo, hi = paired_diff_ci(
                arr[:n].mean(axis=1), baseline_arr[:n].mean(axis=1)
            )
            if lo > 0 or hi < 0:  # CI excludes 0 → significant
                y = (iqm_vals[i] if not np.isnan(iqm_vals[i]) else 0) + ci_his[i] + 0.02
                ax1.text(i, y, "*", ha="center", va="bottom", fontsize=12)

    ax2.bar(x, mean_times, color=colors)
    ax2.set_xticks(x); ax2.set_xticklabels(OBS_MODES, rotation=15, ha="right")
    ax2.set_ylabel("Training wall-clock (min)"); ax2.set_title("(b) Training cost per obs_mode")

    plt.tight_layout()
    out_path = out / f"rq1_{args.regime}.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
