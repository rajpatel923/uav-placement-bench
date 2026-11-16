#!/usr/bin/env python
"""Verify Al-Hourani channel constants and reproduce radius-altitude figure.

Compare optimal elevation angles against published Table I of [1].
Plot coverage radius vs altitude for all four environments (compare to Fig. 3 of [1]).

Usage: python scripts/verify_channel.py [--pl-max 100] [--out docs/channel_verification.png]
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from uavbench.channel import AlHouraniChannel, ENVIRONMENTS

PUBLISHED_THETA_OPT = {
    "suburban": 20.34,
    "urban": 42.44,
    "dense_urban": 54.62,
    "highrise_urban": 75.52,
}


def check_elevation_angles() -> bool:
    print("\n=== Optimal elevation angles vs Al-Hourani WCL 2014 Table I ===")
    all_ok = True
    for name, expected in PUBLISHED_THETA_OPT.items():
        ch = AlHouraniChannel(name)
        got = ch.optimal_elevation_deg()
        diff = abs(got - expected)
        status = "OK" if diff < 0.01 else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(f"  {name:20s}: published {expected:.2f}°  computed {got:.4f}°  diff {diff:.4f}°  [{status}]")
    return all_ok


def plot_radius_altitude(pl_max: float, out_path: str) -> None:
    altitudes = np.linspace(10, 500, 300)
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in ENVIRONMENTS:
        ch = AlHouraniChannel(name)
        radii = np.array([ch.coverage_radius(float(h), pl_max) for h in altitudes])
        ax.plot(altitudes, radii, label=name)
        h_opt, _ = ch.optimal_altitude(pl_max)
        ax.axvline(h_opt, linestyle="--", alpha=0.25)
    ax.set_xlabel("UAV altitude h (m)")
    ax.set_ylabel(f"Coverage radius R (m) [PL ≤ {pl_max:.0f} dB]")
    ax.set_title("Al-Hourani Coverage Radius vs Altitude\n(Reproduce shape of WCL 2014 Fig. 3)")
    ax.legend()
    ax.grid(True, alpha=0.4)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure → {out_path}")
    print("Compare shape visually against Al-Hourani WCL 2014 Fig. 3.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pl-max", type=float, default=100.0)
    p.add_argument("--out", default="docs/channel_verification.png")
    args = p.parse_args()
    ok = check_elevation_angles()
    plot_radius_altitude(args.pl_max, args.out)
    if not ok:
        print("\nWARNING: Some elevation angles deviate more than 0.01°. Check channel.py constants.")
        sys.exit(1)


if __name__ == "__main__":
    main()
