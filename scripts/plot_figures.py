"""Generate publication-quality PDF bar charts from benchmark results.

Reads ``results/{regime}/{algo}/seed*.json`` files and produces:
  - ``docs/figures/fig1_static_coverage.pdf``
  - ``docs/figures/fig2_mobile_coverage.pdf``

Algos with no data are silently skipped so the script works on a partial results
directory (e.g. after a smoke run).  Requires a full 10-seed sweep for the bars
to reflect the true IQM; smoke-test bars are labelled as such only if the caller
supplies ``--smoke-label``.

Example::

    python scripts/plot_figures.py --results results
    python scripts/plot_figures.py --results results --smoke-label
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# Canonical algo ordering (worst → best, roughly) and display names
# ---------------------------------------------------------------------------
ALGO_ORDER: list[str] = [
    "random",
    "hover",
    "lyu_spiral",
    "kmeans",
    "pso",
    "dqn",
    "a2c",
    "ppo",
    "sac",
]

ALGO_LABELS: dict[str, str] = {
    "random": "Random",
    "hover": "Hover",
    "lyu_spiral": "Lyu Spiral",
    "kmeans": "k-Means",
    "pso": "PSO",
    "dqn": "DQN",
    "a2c": "A2C",
    "ppo": "PPO",
    "sac": "SAC",
}

# Colour palette — distinct, colour-blind-friendly-ish
ALGO_COLOURS: dict[str, str] = {
    "random": "#9E9E9E",
    "hover": "#607D8B",
    "lyu_spiral": "#795548",
    "kmeans": "#2196F3",
    "pso": "#03A9F4",
    "dqn": "#FF9800",
    "a2c": "#FF5722",
    "ppo": "#4CAF50",
    "sac": "#9C27B0",
}

DEFAULT_COLOUR = "#BDBDBD"


# ---------------------------------------------------------------------------
# IQM + bootstrap CI
# ---------------------------------------------------------------------------

def iqm_bootstrap_ci(
    scores_2d: np.ndarray,
    reps: int = 5000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (iqm, ci_lo, ci_hi) for a (n_seeds, n_layouts) score matrix.

    Bootstraps within the fixed interquartile subset — thresholds are fixed
    at the population level (Agarwal et al. 2021 rliable convention).
    """
    flat = scores_2d.ravel()
    q25, q75 = np.percentile(flat, 25), np.percentile(flat, 75)
    iq = flat[(flat >= q25) & (flat <= q75)]
    if len(iq) == 0:
        iq = flat
    point_iqm = float(iq.mean())

    rng = np.random.default_rng(seed)
    boot_iqms = np.array([
        float(rng.choice(iq, size=len(iq), replace=True).mean())
        for _ in range(reps)
    ])
    alpha = (1.0 - ci) / 2.0
    ci_lo = float(np.percentile(boot_iqms, 100 * alpha))
    ci_hi = float(np.percentile(boot_iqms, 100 * (1 - alpha)))
    return point_iqm, ci_lo, ci_hi


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

KNOWN_ALGOS: set[str] = {
    "kmeans", "pso", "random", "lyu_spiral", "hover", "ppo", "sac", "a2c", "dqn"
}


def load_scores(results_dir: Path, regime: str, metric: str) -> dict[str, np.ndarray]:
    """Return ``{algo: np.ndarray of shape (n_seeds, n_layouts)}``."""
    regime_dir = results_dir / regime
    if not regime_dir.exists():
        return {}

    raw: dict[str, list[list[float]]] = defaultdict(list)
    for f in sorted(regime_dir.glob("*/seed*.json")):
        try:
            d = json.loads(f.read_text())
            algo = d.get("algo", f.parent.name)
            if algo not in KNOWN_ALGOS:
                print(
                    f"[plot_figures] WARNING: unrecognised algo '{algo}' in {f}, skipping",
                    file=sys.stderr,
                )
                continue
            per_layout = d.get(f"{metric}_per_layout")
            if per_layout is not None:
                raw[algo].append(per_layout)
        except Exception:
            continue  # skip unreadable files gracefully

    return {k: np.array(v) for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def make_bar_figure(
    scores: dict[str, np.ndarray],
    regime: str,
    metric: str,
    fig_number: int,
    smoke_label: bool = False,
) -> plt.Figure:
    """Return a matplotlib Figure with IQM bar chart + 95% CI error bars."""
    # Keep only algos that have data, in canonical order
    present = [a for a in ALGO_ORDER if a in scores]
    # Append any algos not in ALGO_ORDER (future additions)
    extra = sorted(a for a in scores if a not in ALGO_ORDER)
    present += extra

    if not present:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"Fig {fig_number}: {regime.capitalize()} — {metric}")
        return fig

    iqms, lo_errs, hi_errs = [], [], []
    for algo in present:
        mat = scores[algo]
        iqm, ci_lo, ci_hi = iqm_bootstrap_ci(mat)
        iqms.append(iqm)
        lo_errs.append(iqm - ci_lo)
        hi_errs.append(ci_hi - iqm)

    n = len(present)
    fig, ax = plt.subplots(figsize=(max(5, n * 0.8 + 1.5), 4.5))

    x = np.arange(n)
    colours = [ALGO_COLOURS.get(a, DEFAULT_COLOUR) for a in present]
    bars = ax.bar(
        x,
        iqms,
        color=colours,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    ax.errorbar(
        x,
        iqms,
        yerr=[lo_errs, hi_errs],
        fmt="none",
        ecolor="black",
        elinewidth=1.2,
        capsize=4,
        capthick=1.2,
        zorder=4,
    )

    # Annotate bars with seed count
    for bar, algo in zip(bars, present):
        n_seeds = scores[algo].shape[0]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (hi_errs[present.index(algo)] if hi_errs else 0) + 0.005,
            f"n={n_seeds}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#555555",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([ALGO_LABELS.get(a, a) for a in present], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(f"IQM {metric.replace('_', ' ')} (95% CI)", fontsize=9)
    ax.set_ylim(bottom=0.0)

    title = f"Fig {fig_number}: {regime.capitalize()} regime — {metric.replace('_', ' ').title()}"
    if smoke_label:
        title += "  [SMOKE — not final results]"
    ax.set_title(title, fontsize=10, pad=8)

    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Plot IQM bar charts from benchmark results.")
    p.add_argument("--results", default="results", help="Root results directory (default: results)")
    p.add_argument("--metric", default="final_cov", choices=["final_cov", "avg_cov"])
    p.add_argument("--out", default="docs/figures", help="Output directory for PDFs")
    p.add_argument("--smoke-label", action="store_true",
                   help="Add 'SMOKE — not final results' label to figures")
    args = p.parse_args()

    results_dir = Path(args.results)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    regime_configs = [
        ("static", 1, "fig1_static_coverage.pdf"),
        ("mobile", 2, "fig2_mobile_coverage.pdf"),
    ]

    any_data = False
    for regime, fig_num, filename in regime_configs:
        scores = load_scores(results_dir, regime, args.metric)
        if not scores:
            print(f"[plot_figures] No data for regime={regime}, skipping {filename}")
            continue

        any_data = True
        n_seeds_total = sum(v.shape[0] for v in scores.values())
        algos_found = list(scores.keys())
        print(f"[plot_figures] {regime}: {len(algos_found)} algos, {n_seeds_total} seed files total — "
              f"{algos_found}")

        fig = make_bar_figure(scores, regime, args.metric, fig_num, smoke_label=args.smoke_label)
        pdf_path = out_dir / filename
        fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot_figures] Saved {pdf_path}")

    if not any_data:
        print(f"[plot_figures] WARNING: no result files found under {results_dir}/")
        sys.exit(0)


if __name__ == "__main__":
    main()
