"""Shared evaluation protocol so RL agents and baselines are scored identically.

Layout splits (seeds passed to env.reset):
  * "test": EVAL_SEED_OFFSET + i  -> the ONLY layouts used for reported results.
  * "val":  VAL_SEED_OFFSET + i   -> use these for ALL hyperparameter tuning.
Training resets come from the run seed via SB3 and never use these offsets.
Training performance and evaluation performance are kept separate, as
recommended by Agarwal et al. 2021. Do not change these constants without a
logged decision in docs/DECISIONS.md and a full re-run.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EVAL_SEED_OFFSET = 10_000   # test split
VAL_SEED_OFFSET = 20_000    # validation split (tuning only)
SPLIT_OFFSETS = {"test": EVAL_SEED_OFFSET, "val": VAL_SEED_OFFSET}
DEFAULT_OUT = {"test": "results", "val": "results_val"}
FINAL_WINDOW = 10  # "final coverage" = mean coverage over the last N steps


def layout_seeds(split: str, n: int) -> list[int]:
    """Reset seeds for the n layouts of a split ('test' or 'val')."""
    return [SPLIT_OFFSETS[split] + i for i in range(n)]


def run_episode(env, act_fn, layout_seed: int):
    """act_fn(obs, base_env) -> action. Returns per-step coverage array."""
    obs, info = env.reset(seed=layout_seed)
    base = env.unwrapped
    covs = []
    done = False
    while not done:
        action = act_fn(obs, base)
        obs, _, terminated, truncated, info = env.step(action)
        covs.append(info["coverage"])
        done = terminated or truncated
    return np.array(covs)


def summarize(cov_runs: list[np.ndarray]) -> dict:
    final = [float(np.mean(c[-FINAL_WINDOW:])) for c in cov_runs]
    avg = [float(np.mean(c)) for c in cov_runs]
    return {
        "final_cov": float(np.mean(final)),
        "avg_cov": float(np.mean(avg)),
        "final_cov_per_layout": final,
        "avg_cov_per_layout": avg,
    }


def save_result(out_dir: str | Path, regime: str, algo: str, seed: int, payload: dict):
    path = Path(out_dir) / regime / algo / f"seed{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"regime": regime, "algo": algo, "seed": seed, **payload}
    path.write_text(json.dumps(payload, indent=2))
    return path
