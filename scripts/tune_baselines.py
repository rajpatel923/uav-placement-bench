#!/usr/bin/env python
"""Tune PSO and k-means hyperparameters on the validation split.

Every trial uses tuning seeds 100–102 and 5 val layouts. Writes results to results_val/.
Budget = number from EXPERIMENT_PROTOCOL.md Section 5 (same for all methods).

NEVER change TUNING_SEEDS or the split offset here. See docs/EXPERIMENT_PROTOCOL.md.
"""
from __future__ import annotations
import argparse, itertools, json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from uavbench.env import make_env
from uavbench.baselines import TargetTrackingPolicy, PLANNERS
from uavbench.evaluation import run_episode, summarize, save_result, layout_seeds
from scipy.optimize import linear_sum_assignment

TUNING_SEEDS = [100, 101, 102]
N_EVAL_LAYOUTS = 5  # small during tuning; use 20 for final reported results

PSO_CONFIGS = [
    {"n_particles": n, "iters": it, "w": w, "c1": 1.5, "c2": 1.5}
    for n, it, w in itertools.product([20, 30, 50], [40, 60, 100], [0.5, 0.7, 0.9])
]

KMEANS_CONFIGS = [
    {"edge_quantile": q} for q in [0.7, 0.8, 0.9, 0.95]
]


class _KwargsPolicy:
    """Thin wrapper that forwards planner_kwargs into the planner call."""

    def __init__(self, algo: str, seed: int, **planner_kwargs):
        self.planner = PLANNERS[algo]
        self.rng = np.random.default_rng(seed)
        self.planner_kwargs = planner_kwargs
        self.targets = None
        self._t = 0

    def reset(self, env):
        self._t = 0
        self._plan(env)

    def _plan(self, env):
        targets = self.planner(env, self.rng, **self.planner_kwargs)
        cost = np.linalg.norm(env.uavs[:, None, :] - targets[None, :, :], axis=-1)
        rows, cols = linear_sum_assignment(cost)
        self.targets = targets[cols[np.argsort(rows)]]

    def act(self, env) -> np.ndarray:
        self._t += 1
        delta = (self.targets - env.uavs) / env._step_scale
        return np.clip(delta, -1.0, 1.0).ravel().astype(np.float32)


def eval_policy(algo: str, regime: str, seed: int, out_dir: str, **planner_kwargs) -> float:
    env = make_env(regime=regime)
    policy = _KwargsPolicy(algo, seed=seed, **planner_kwargs)

    cov_runs = []
    for ls in layout_seeds("val", N_EVAL_LAYOUTS):
        obs, _ = env.reset(seed=ls)
        policy.reset(env.unwrapped)
        cov_runs.append(run_episode(env, lambda obs, base: policy.act(base), ls))
    result = summarize(cov_runs)
    result["planner_kwargs"] = planner_kwargs
    save_result(out_dir, regime, f"{algo}_tuning", seed, result)
    return result["final_cov"]


def tune(algo: str, regime: str, configs: list[dict], budget: int, out_dir: str):
    print(f"\n--- Tuning {algo} on {regime} ({min(len(configs), budget)} trials) ---")
    best_score, best_cfg = -1.0, None
    for trial, cfg in enumerate(configs[:budget]):
        scores = [eval_policy(algo, regime, s, out_dir, **cfg) for s in TUNING_SEEDS]
        mean = float(np.mean(scores))
        print(f"  Trial {trial+1:3d}: {cfg}  ->  final_cov={mean:.4f} "
              f"[{min(scores):.4f}, {max(scores):.4f}]")
        if mean > best_score:
            best_score, best_cfg = mean, cfg
    print(f"  Best: {best_cfg}  ->  {best_score:.4f}")
    out = Path(out_dir) / f"best_{algo}_{regime}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"algo": algo, "regime": regime, "hp": best_cfg,
                               "val_final_cov": best_score}, indent=2))
    return best_cfg, best_score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regime", default="static", choices=["static", "mobile"])
    p.add_argument("--budget", type=int, default=20)
    p.add_argument("--out", default="results_val")
    args = p.parse_args()
    tune("pso", args.regime, PSO_CONFIGS, args.budget, args.out)
    tune("kmeans", args.regime, KMEANS_CONFIGS, min(4, args.budget), args.out)


if __name__ == "__main__":
    main()
