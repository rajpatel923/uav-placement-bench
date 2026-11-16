#!/usr/bin/env python
"""Tune SB3 RL hyperparameters on the validation split.

WARNING: This is compute-heavy. SAC × 20 trials × 3 seeds × 200k steps ≈ hours on CPU.
Use --timesteps 50000 for a preliminary sweep, then confirm best config at 200k.
Budget must equal the baseline budget (see docs/EXPERIMENT_PROTOCOL.md Section 5).

NEVER use the test split (seeds 10000+i) here. Only --split val.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from stable_baselines3 import PPO, SAC, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from uavbench.env import make_env
from uavbench.evaluation import run_episode, summarize, save_result, layout_seeds

TUNING_SEEDS = [100, 101, 102]
N_EVAL_LAYOUTS = 5
ALGOS = {"ppo": PPO, "sac": SAC, "a2c": A2C, "dqn": DQN}
ON_POLICY = {"ppo", "a2c"}
DISCRETE = {"dqn"}

GRIDS: dict[str, list[dict]] = {
    "ppo": [{"learning_rate": lr, "ent_coef": ec}
            for lr in [3e-4, 1e-4, 3e-3] for ec in [0.0, 0.01]],
    "a2c": [{"learning_rate": lr, "ent_coef": ec}
            for lr in [7e-4, 3e-4, 1e-3] for ec in [0.0, 0.01]],
    "sac": [{"learning_rate": lr, "buffer_size": bs}
            for lr in [3e-4, 1e-4] for bs in [100_000, 50_000]],
    "dqn": [{"learning_rate": lr, "exploration_fraction": ef}
            for lr in [1e-4, 3e-4] for ef in [0.1, 0.2]],
}


def eval_policy(algo_name: str, regime: str, seed: int, timesteps: int,
                hp: dict, out_dir: str, obs_mode: str = "flat", device: str = "cpu") -> float:
    is_discrete = algo_name in DISCRETE
    is_on_policy = algo_name in ON_POLICY
    env_fn = lambda: make_env(regime=regime, discrete=is_discrete, obs_mode=obs_mode)
    n_envs = 4 if is_on_policy else 1
    env = make_vec_env(env_fn, n_envs=n_envs, seed=seed)
    model = ALGOS[algo_name]("MlpPolicy", env, verbose=0, seed=seed, device=device, **hp)
    model.learn(total_timesteps=timesteps)
    eval_env = make_env(regime=regime, discrete=is_discrete, obs_mode=obs_mode)
    cov_runs = []
    for ls in layout_seeds("val", N_EVAL_LAYOUTS):
        def act_fn(obs, base):
            action, _ = model.predict(obs, deterministic=True)
            return action
        cov_runs.append(run_episode(eval_env, act_fn, ls))
    result = summarize(cov_runs)
    result["hp"] = hp
    result["obs_mode"] = obs_mode
    save_result(out_dir, regime, f"{algo_name}_tuning", seed, result)
    return result["final_cov"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", required=True, choices=list(ALGOS))
    p.add_argument("--regime", default="static", choices=["static", "mobile"])
    p.add_argument("--budget", type=int, default=20)
    p.add_argument("--timesteps", type=int, default=50_000,
                   help="Use 50k for initial search; confirm best config at 200k")
    p.add_argument("--obs-mode", default="flat",
                   choices=["flat", "sorted", "cluster_summary"])
    p.add_argument("--out", default="results_val")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    grid = GRIDS[args.algo][:args.budget]
    print(f"\n--- Tuning {args.algo} on {args.regime} "
          f"({len(grid)} trials × {len(TUNING_SEEDS)} seeds, {args.timesteps} steps each) ---")
    best_score, best_hp = -1.0, None
    for trial, hp in enumerate(grid):
        scores = [eval_policy(args.algo, args.regime, s, args.timesteps, hp,
                               args.out, args.obs_mode, args.device) for s in TUNING_SEEDS]
        mean = float(np.mean(scores))
        print(f"  Trial {trial+1:3d}: {hp}  ->  final_cov={mean:.4f} "
              f"[{min(scores):.4f}, {max(scores):.4f}]")
        if mean > best_score:
            best_score, best_hp = mean, hp
    print(f"\n  Best {args.algo}: {best_hp}  ->  {best_score:.4f}")
    out_path = Path(args.out) / f"best_{args.algo}_{args.regime}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "algo": args.algo, "regime": args.regime,
        "hp": best_hp, "val_final_cov": best_score,
        "obs_mode": args.obs_mode, "timesteps": args.timesteps,
    }, indent=2))


if __name__ == "__main__":
    main()
