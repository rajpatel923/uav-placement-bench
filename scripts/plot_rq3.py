"""Figure 3: RQ3 generalization results.

Panel (a): RQ3-A line chart — final_cov vs distribution shift config per obs_mode.
Panel (b): RQ3-B bar chart — GNN coverage at fleet sizes 4/6/8.
Panel (c): Cost table — training time and inference latency per obs_mode.

Usage:
    python scripts/plot_rq3.py --rq3a results_rq3/rq3a --rq3b results_rq3/rq3b \\
        --timing results --out figures/
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


SHIFT_LABELS = ["train", "users-50", "users-150", "area-2000", "area-4000"]
OBS_MODES = ["flat", "sorted", "cluster_summary", "spatial_grid", "graph"]
FLEET_LABELS = ["fleet-4", "fleet-6", "fleet-8"]


def _load_mean(results_dir: Path, config: str, method: str, metric: str) -> float:
    folder = results_dir / config / method
    vals = []
    for f in sorted(folder.glob("seed*.json")):
        d = json.loads(f.read_text())
        v = d.get(f"{metric}_per_layout")
        if v is not None:
            vals.extend(v)
    return float(np.mean(vals)) if vals else float("nan")


def _load_timing(timing_dir: Path, obs_mode: str) -> dict:
    times = []
    for f in sorted(Path(timing_dir).glob(f"*/{obs_mode}/timing_seed*.json")):
        try:
            times.append(json.loads(f.read_text()))
        except Exception:
            pass
    if not times:
        return {"train_time_s": float("nan"), "inference_latency_ms": float("nan")}
    return {
        "train_time_s": float(np.mean([t["train_time_s"] for t in times])),
        "inference_latency_ms": float(np.mean([t["inference_latency_ms"] for t in times])),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rq3a", default="results_rq3/rq3a")
    p.add_argument("--rq3b", default="results_rq3/rq3b")
    p.add_argument("--timing", default="results",
                   help="root results dir with timing_seed*.json files")
    p.add_argument("--metric", default="final_cov")
    p.add_argument("--out", default="figures")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rq3a = Path(args.rq3a)
    rq3b = Path(args.rq3b)

    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
    ax_a, ax_b, ax_c = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

    # ── Panel (a): RQ3-A ─────────────────────────────────────────────────────
    colors = plt.cm.tab10(np.linspace(0, 1, len(OBS_MODES)))
    for obs_mode, color in zip(OBS_MODES, colors):
        vals = [_load_mean(rq3a, cfg, obs_mode, args.metric) for cfg in SHIFT_LABELS]
        ax_a.plot(range(len(SHIFT_LABELS)), vals, marker="o", label=obs_mode, color=color)
    ax_a.set_xticks(range(len(SHIFT_LABELS)))
    ax_a.set_xticklabels(SHIFT_LABELS, rotation=20, ha="right", fontsize=8)
    ax_a.set_ylabel(args.metric)
    ax_a.set_title("(a) RQ3-A: Distribution Shift")
    ax_a.legend(fontsize=7)
    ax_a.set_ylim(0, 1)

    # ── Panel (b): RQ3-B ─────────────────────────────────────────────────────
    vals_b = [_load_mean(rq3b, cfg, "graph", args.metric) for cfg in FLEET_LABELS]
    bars = ax_b.bar(FLEET_LABELS, vals_b, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax_b.set_ylabel(args.metric)
    ax_b.set_title("(b) RQ3-B: Fleet Size (GNN only)")
    ax_b.set_ylim(0, 1)
    for bar, val in zip(bars, vals_b):
        if not np.isnan(val):
            ax_b.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                      f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    # ── Panel (c): Cost table ─────────────────────────────────────────────────
    ax_c.axis("off")
    rows = []
    for obs_mode in OBS_MODES:
        t = _load_timing(args.timing, obs_mode)
        rows.append([
            obs_mode,
            f"{t['train_time_s'] / 60:.1f} min" if not np.isnan(t["train_time_s"]) else "N/A",
            f"{t['inference_latency_ms']:.2f} ms" if not np.isnan(t["inference_latency_ms"]) else "N/A",
        ])
    table = ax_c.table(
        cellText=rows,
        colLabels=["obs_mode", "train time", "latency"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax_c.set_title("(c) Deployment Cost")

    plt.suptitle("RQ3: Generalization and Deployment Cost", fontsize=13, y=1.02)
    out_path = out / "rq3_generalization.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
