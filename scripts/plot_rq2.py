"""Figure 2: RQ2 complexity threshold heatmap.

Reads results_rq2/{config}/{method}/seed*.json,
computes paired diff CI (RL vs PSO) per config,
plots a 3×2 annotated heatmap.

Usage:
    python scripts/plot_rq2.py --results results_rq2 --out figures/
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from scripts.stats import paired_diff_ci, prob_of_improvement


CONFIGS_ORDERED = [
    ("C1", "static\n(0 m/s, no fail)"),
    ("C2", "mobile-low\n(5 m/s, no fail)"),
    ("C3", "mobile-high\n(15 m/s, no fail)"),
    ("C4", "static\n(0 m/s, +fail)"),
    ("C5", "mobile-low\n(5 m/s, +fail)"),
    ("C6", "mobile-high\n(15 m/s, +fail)"),
]


def load_scores(results_dir: Path, config: str, method: str, metric: str) -> np.ndarray:
    folder = results_dir / config / method
    scores = []
    for f in sorted(folder.glob("seed*.json")):
        d = json.loads(f.read_text())
        scores.append(d[f"{metric}_per_layout"])
    if not scores:
        return np.array([])
    return np.concatenate(scores)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results_rq2")
    p.add_argument("--out", default="figures")
    p.add_argument("--metric", default="final_cov")
    p.add_argument("--rl-method", default="rl_flat",
                   help="RL method key (e.g. rl_flat, rl_graph)")
    args = p.parse_args()
    results = Path(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    points, ci_los, ci_his, pois = [], [], [], []
    labels = []

    for cfg_name, cfg_label in CONFIGS_ORDERED:
        rl = load_scores(results, cfg_name, args.rl_method, args.metric)
        pso = load_scores(results, cfg_name, "pso", args.metric)
        if len(rl) == 0 or len(pso) == 0:
            points.append(float("nan")); ci_los.append(float("nan"))
            ci_his.append(float("nan")); pois.append(float("nan"))
        else:
            n = min(len(rl), len(pso))
            pt, lo, hi = paired_diff_ci(rl[:n], pso[:n])
            poi = prob_of_improvement(rl[:n], pso[:n])
            points.append(pt); ci_los.append(lo); ci_his.append(hi); pois.append(poi)
        labels.append(cfg_label)

    # 3×2 heatmap: rows = failure condition, cols = mobility level
    fig, axes = plt.subplots(1, 1, figsize=(10, 5))
    data = np.array(points).reshape(2, 3)   # (fail, mobility)
    ci_lo_arr = np.array(ci_los).reshape(2, 3)
    ci_hi_arr = np.array(ci_his).reshape(2, 3)
    poi_arr = np.array(pois).reshape(2, 3)

    # Custom colormap: red=PSO wins, grey=inconclusive, green=RL wins
    cmap = mcolors.LinearSegmentedColormap.from_list("rwg", ["#d62728", "#cccccc", "#2ca02c"])
    finite_vals = data[np.isfinite(data)]
    vmax = max(abs(finite_vals).max(), 0.05) if len(finite_vals) > 0 else 0.05
    im = axes.imshow(data, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

    row_labels = ["No failure", "With failure"]
    col_labels = ["0 m/s", "5 m/s", "15 m/s"]
    axes.set_xticks(range(3)); axes.set_xticklabels(col_labels, fontsize=11)
    axes.set_yticks(range(2)); axes.set_yticklabels(row_labels, fontsize=11)
    axes.set_xlabel("User speed", fontsize=12)
    axes.set_title("RL − PSO mean(Δfinal_cov) with 95% CI", fontsize=13)

    for r in range(2):
        for c in range(3):
            pt = data[r, c]
            lo = ci_lo_arr[r, c]
            hi = ci_hi_arr[r, c]
            poi = poi_arr[r, c]
            if np.isnan(pt):
                txt = "N/A"
            else:
                sig = "*" if lo > 0 or hi < 0 else ""
                txt = f"{pt:+.3f}{sig}\n[{lo:+.3f},{hi:+.3f}]\nP={poi:.2f}"
            axes.text(c, r, txt, ha="center", va="center", fontsize=8,
                      color="white" if abs(pt) > vmax * 0.6 else "black")

    plt.colorbar(im, ax=axes, label="mean(RL − PSO)")
    plt.tight_layout()
    out_path = out / "rq2_heatmap.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
