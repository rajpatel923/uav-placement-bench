"""RQ3 generalization evaluation.

RQ3-A: Distribution shift — evaluate all saved policies on shifted n_users / area_size.
RQ3-B: Structural shift — evaluate GNN policy (trained on n_uavs=4) on 6 and 8 UAVs.

Usage (smoke):
    python scripts/eval_generalization.py --model-dir /tmp/smoke/static --obs-mode flat \\
        --out /tmp/rq3 --seeds 0 1 --n-eval 5

Usage (full, after sweeping all obs_modes in RQ1):
    python scripts/eval_generalization.py --model-dir results/static \\
        --obs-mode flat sorted cluster_summary spatial_grid graph \\
        --out results_rq3 --seeds $(seq -s' ' 0 9) --n-eval 20
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from stable_baselines3 import PPO, SAC

from uavbench import make_env
from uavbench.evaluation import layout_seeds, run_episode, save_result, summarize

# RQ3-A distribution shift configs
RQ3A_CONFIGS = [
    {"label": "train",      "n_users": 100, "area": 3000.0},
    {"label": "users-50",   "n_users": 50,  "area": 3000.0},
    {"label": "users-150",  "n_users": 150, "area": 3000.0},
    {"label": "area-2000",  "n_users": 100, "area": 2000.0},
    {"label": "area-4000",  "n_users": 100, "area": 4000.0},
]

# RQ3-B fleet size configs (GNN only)
RQ3B_CONFIGS = [
    {"label": "fleet-4", "n_uavs": 4},
    {"label": "fleet-6", "n_uavs": 6},
    {"label": "fleet-8", "n_uavs": 8},
]


def _load_model(model_dir: Path, obs_mode: str, seed: int):
    """Load SB3 model checkpoint.

    Tries two directory layouts in order:
      1. {model_dir}/{obs_mode}/models/seed{N}.zip  — obs-mode-keyed layout (RQ3 sweep)
      2. {model_dir}/{algo}/models/seed{N}.zip       — algo-keyed layout (train_rl.py default)

    For each path, tries PPO then SAC loaders.
    """
    from stable_baselines3 import A2C, DQN

    algo_names = ["ppo", "sac", "a2c", "dqn"]
    algo_classes = [PPO, SAC, A2C, DQN]

    # Layout 1: obs_mode-keyed (primary expectation for RQ3)
    obs_mode_path = model_dir / obs_mode / "models" / f"seed{seed}.zip"
    candidate_paths = [obs_mode_path]
    # Layout 2: algo-keyed (train_rl.py saves here)
    algo_paths = [model_dir / algo / "models" / f"seed{seed}.zip" for algo in algo_names]
    candidate_paths += algo_paths

    for p in candidate_paths:
        if not p.exists():
            continue
        if p != obs_mode_path:
            warnings.warn(
                f"obs-mode-keyed checkpoint not found at {obs_mode_path}; "
                f"falling back to {p}. Verify this model matches obs_mode={obs_mode!r}.",
                UserWarning,
                stacklevel=2,
            )
        for algo_cls in algo_classes:
            try:
                return algo_cls.load(p)
            except Exception:
                continue
    return None


def eval_rq3a(model_dir, obs_modes, seeds, n_eval, split, out):
    """RQ3-A: zero-shot distribution shift."""
    for obs_mode in obs_modes:
        for seed in seeds:
            model = _load_model(Path(model_dir), obs_mode, seed)
            if model is None:
                print(f"  SKIP {obs_mode} seed={seed} — checkpoint not found")
                continue
            for cfg in RQ3A_CONFIGS:
                label = cfg["label"]
                n_users = cfg["n_users"]
                area = cfg["area"]
                try:
                    env = make_env("static", obs_mode=obs_mode,
                                   n_users=n_users, area=area)
                    env.reset(seed=10000)  # trigger obs shape check
                except Exception as e:
                    # MLP obs dim mismatch (n_users changed) — expected limitation
                    res = {"final_cov": float("nan"), "avg_cov": float("nan"),
                           "error": str(e), "note": "obs_dim_mismatch"}
                    save_result(out, f"rq3a/{label}", obs_mode, seed, res)
                    print(f"  rq3a/{label} {obs_mode} seed={seed}: {e}")
                    continue

                def act(obs, _):
                    a, _ = model.predict(obs, deterministic=True)
                    return a

                try:
                    covs = [run_episode(env, act, s) for s in layout_seeds(split, n_eval)]
                    res = summarize(covs)
                except Exception as e:
                    res = {"final_cov": float("nan"), "avg_cov": float("nan"), "error": str(e)}
                res.update({"obs_mode": obs_mode, "config": label,
                             "n_users": n_users, "area": area})
                save_result(out, f"rq3a/{label}", obs_mode, seed, res)
                print(f"  rq3a/{label} {obs_mode:14s} seed={seed} final_cov={res['final_cov']:.3f}")


def eval_rq3b(model_dir, seeds, n_eval, split, out):
    """RQ3-B: structural shift — GNN only."""
    obs_mode = "graph"
    for seed in seeds:
        model = _load_model(Path(model_dir), obs_mode, seed)
        if model is None:
            print(f"  SKIP graph seed={seed} — checkpoint not found")
            continue
        for cfg in RQ3B_CONFIGS:
            label, n_uavs = cfg["label"], cfg["n_uavs"]
            try:
                env = make_env("static", obs_mode=obs_mode, n_uavs=n_uavs)

                def act(obs, _):
                    a, _ = model.predict(obs, deterministic=True)
                    return a

                covs = [run_episode(env, act, s) for s in layout_seeds(split, n_eval)]
                res = summarize(covs)
            except Exception as e:
                res = {"final_cov": float("nan"), "avg_cov": float("nan"), "error": str(e)}
            res.update({"obs_mode": obs_mode, "config": label, "n_uavs": n_uavs})
            save_result(out, f"rq3b/{label}", obs_mode, seed, res)
            print(f"  rq3b/{label} graph seed={seed} final_cov={res['final_cov']:.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True,
                   help="directory containing {obs_mode}/models/seed*.zip")
    p.add_argument("--obs-mode", nargs="+", default=["flat"],
                   choices=["flat", "sorted", "cluster_summary", "spatial_grid", "graph"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--n-eval", type=int, default=20)
    p.add_argument("--split", choices=["test", "val"], default="val")
    p.add_argument("--out", default="results_rq3")
    p.add_argument("--rq3b", action="store_true", help="also run RQ3-B fleet-size evaluation")
    args = p.parse_args()

    print("=== RQ3-A: distribution shift ===")
    eval_rq3a(args.model_dir, args.obs_mode, args.seeds, args.n_eval, args.split, args.out)
    if args.rq3b:
        print("\n=== RQ3-B: fleet size (GNN only) ===")
        eval_rq3b(args.model_dir, args.seeds, args.n_eval, args.split, args.out)


if __name__ == "__main__":
    main()
