"""RQ2 complexity sweep across 6 configs × 4 methods × 10 seeds.

Usage (smoke):
    python scripts/sweep_rq2.py --seeds 0 1 --n-eval 5 --out /tmp/rq2 --obs-mode flat --timesteps 20000 --configs C1 C4

Usage (full, run after RQ1 to pick best obs_mode):
    python scripts/sweep_rq2.py --seeds $(seq -s' ' 0 9) --n-eval 20 --out results_rq2 --obs-mode graph

Methods run per config: pso, warm_start_pso, mpc, rl (trained on that config).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from stable_baselines3 import PPO

from uavbench import make_env
from uavbench.baselines import TargetTrackingPolicy
from uavbench.evaluation import layout_seeds, run_episode, save_result, summarize

CONFIGS = {
    "C1": dict(user_speed=0.0,  single_failure=False, mobility="static", label="static"),
    "C2": dict(user_speed=5.0,  single_failure=False, mobility="mobile", label="mobile-low"),
    "C3": dict(user_speed=15.0, single_failure=False, mobility="mobile", label="mobile-high"),
    "C4": dict(user_speed=0.0,  single_failure=True,  mobility="static", label="static+fail"),
    "C5": dict(user_speed=5.0,  single_failure=True,  mobility="mobile", label="mobile-low+fail"),
    "C6": dict(user_speed=15.0, single_failure=True,  mobility="mobile", label="mobile-high+fail"),
}


def run_baseline(name, cfg_params, seed, n_eval, out, split):
    regime = cfg_params["mobility"]
    env_kw = {k: v for k, v in cfg_params.items() if k not in ("mobility", "label")}
    env = make_env(regime, **env_kw)
    rng = np.random.default_rng(seed)  # noqa: F841  kept for signature consistency

    if name == "pso":
        policy = TargetTrackingPolicy("pso", seed=seed, replan_every=5)
    elif name == "warm_start_pso":
        policy = TargetTrackingPolicy("warm_start_pso", seed=seed, replan_every=5)
    elif name == "mpc":
        policy = TargetTrackingPolicy("mpc", seed=seed, replan_every=1)
    else:
        raise ValueError(name)

    def act(obs, base, _policy=policy):
        if base.t == 0:
            _policy.reset(base)
        return _policy.act(base)

    covs = [run_episode(env, act, s) for s in layout_seeds(split, n_eval)]
    res = summarize(covs)
    res.update({"method": name, "config": cfg_params["label"], "seed": seed,
                "env_config": env.unwrapped.cfg.to_dict()})
    return res


def run_rl(obs_mode, cfg_name, cfg_params, seed, n_eval, out, split, timesteps):
    regime = cfg_params["mobility"]
    env_kw = {k: v for k, v in cfg_params.items() if k not in ("mobility", "label")}
    model_path = Path(out) / cfg_name / "rl" / "models" / f"seed{seed}.zip"

    if not model_path.exists():
        # train if checkpoint not found
        from stable_baselines3.common.env_util import make_vec_env
        from scripts.train_rl import _policy_cls_kwargs

        env_fn = lambda: make_env(regime, obs_mode=obs_mode, **env_kw)  # noqa: E731
        ref_env = make_env(regime, obs_mode=obs_mode, **env_kw)
        pol_cls, pol_kw = _policy_cls_kwargs(obs_mode, ref_env)
        ref_env.close()

        train_env = make_vec_env(env_fn, n_envs=4, seed=seed)
        model = PPO(pol_cls, train_env, seed=seed, verbose=0,
                    **({"policy_kwargs": pol_kw} if pol_kw else {}))
        model.learn(total_timesteps=timesteps, progress_bar=False)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path.with_suffix("")))
    else:
        model = PPO.load(str(model_path))

    eval_env = make_env(regime, obs_mode=obs_mode, **env_kw)

    def act(obs, _base):
        a, _ = model.predict(obs, deterministic=True)
        return a

    covs = [run_episode(eval_env, act, s) for s in layout_seeds(split, n_eval)]
    res = summarize(covs)
    res.update({"method": f"rl_{obs_mode}", "config": cfg_params["label"],
                "seed": seed, "env_config": eval_env.unwrapped.cfg.to_dict()})
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--n-eval", type=int, default=20)
    p.add_argument("--out", default="results_rq2")
    p.add_argument("--split", choices=["test", "val"], default="test")
    p.add_argument("--obs-mode", default="flat",
                   choices=["flat", "sorted", "cluster_summary", "spatial_grid", "graph"])
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()))
    args = p.parse_args()

    for cfg_name in args.configs:
        cfg = CONFIGS[cfg_name]
        print(f"\n=== {cfg_name}: {cfg['label']} ===")
        for seed in args.seeds:
            for method in ["pso", "warm_start_pso", "mpc"]:
                res = run_baseline(method, cfg, seed, args.n_eval, args.out, args.split)
                save_result(args.out, cfg_name, method, seed, res)
                print(f"  {method:16s} seed={seed} final_cov={res['final_cov']:.3f}")
            res = run_rl(args.obs_mode, cfg_name, cfg, seed, args.n_eval,
                         args.out, args.split, args.timesteps)
            save_result(args.out, cfg_name, f"rl_{args.obs_mode}", seed, res)
            print(f"  rl_{args.obs_mode:10s} seed={seed} final_cov={res['final_cov']:.3f}")


if __name__ == "__main__":
    main()
