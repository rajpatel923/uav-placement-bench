"""Aggregate results with rliable: IQM, mean, and stratified-bootstrap 95% CIs.

Example:
    python scripts/aggregate.py --regime static --metric final_cov
Outputs a table to stdout, a CSV, and an interval plot PNG in results/.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from rliable import library as rly  # noqa: E402
from rliable import metrics  # noqa: E402


def load(results_dir: Path, regime: str, metric: str):
    """Returns {algo: array (n_seeds, n_layouts)}; each eval layout is treated as a task."""
    scores = defaultdict(list)
    for f in sorted((results_dir / regime).glob("*/seed*.json")):
        d = json.loads(f.read_text())
        scores[d["algo"]].append(d[f"{metric}_per_layout"])
    return {k: np.array(v) for k, v in scores.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regime", default="static")
    p.add_argument("--metric", choices=["final_cov", "avg_cov"], default="final_cov")
    p.add_argument("--results", default="results")
    p.add_argument("--reps", type=int, default=5000, help="bootstrap repetitions")
    args = p.parse_args()

    rdir = Path(args.results)
    scores = load(rdir, args.regime, args.metric)
    if not scores:
        raise SystemExit(f"No results in {rdir / args.regime}")

    agg = lambda x: np.array([metrics.aggregate_iqm(x), metrics.aggregate_mean(x)])
    point, ci = rly.get_interval_estimates(scores, agg, reps=args.reps)

    order = sorted(scores, key=lambda a: -point[a][0])
    lines = ["algo,n_seeds,iqm,iqm_lo,iqm_hi,mean,mean_lo,mean_hi"]
    print(f"\n{args.regime} | metric={args.metric} | 95% stratified bootstrap CIs ({args.reps} reps)")
    print(f"{'algo':8s} {'seeds':>5s}  {'IQM':>6s} [{'lo':>6s}, {'hi':>6s}]   {'Mean':>6s} [{'lo':>6s}, {'hi':>6s}]")
    for a in order:
        (iqm, mean), ((iqm_lo, mean_lo), (iqm_hi, mean_hi)) = point[a], ci[a]
        print(f"{a:8s} {scores[a].shape[0]:5d}  {iqm:6.3f} [{iqm_lo:6.3f}, {iqm_hi:6.3f}]   "
              f"{mean:6.3f} [{mean_lo:6.3f}, {mean_hi:6.3f}]")
        lines.append(f"{a},{scores[a].shape[0]},{iqm:.4f},{iqm_lo:.4f},{iqm_hi:.4f},{mean:.4f},{mean_lo:.4f},{mean_hi:.4f}")

    csv_path = rdir / f"summary_{args.regime}_{args.metric}.csv"
    csv_path.write_text("\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(6, 0.5 + 0.45 * len(order)))
    y = np.arange(len(order))
    for i, a in enumerate(order):
        lo, hi = ci[a][0][0], ci[a][1][0]
        ax.plot([lo, hi], [i, i], lw=6, alpha=0.5)
        ax.plot(point[a][0], i, "k|", ms=14)
    ax.set_yticks(y, order)
    ax.set_xlabel(f"IQM {args.metric} (95% CI)")
    ax.set_title(f"Regime: {args.regime}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    png = rdir / f"iqm_{args.regime}_{args.metric}.png"
    fig.savefig(png, dpi=150)
    print(f"\nSaved {csv_path} and {png}")


if __name__ == "__main__":
    main()
