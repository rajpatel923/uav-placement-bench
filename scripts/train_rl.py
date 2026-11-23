"""Train one RL agent (one algorithm × one regime × one seed) and evaluate it.

Examples:
    python scripts/train_rl.py --algo ppo --regime static --seed 0 --timesteps 200000
    python scripts/train_rl.py --algo ppo --regime static --seed 0 --obs-mode spatial_grid
    python scripts/train_rl.py --algo ppo --regime mobile --seed 0 --obs-mode graph

Fairness rules:
  * Same env factory, evaluation layouts, and metrics as the baselines.
  * Tune with --split val (results_val/); report only --split test (results/).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_baselines3 import DQN, PPO, SAC, A2C          # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402

from uavbench import make_env                               # noqa: E402
from uavbench.evaluation import (DEFAULT_OUT, layout_seeds, # noqa: E402
                                  run_episode, save_result, summarize)

ALGOS    = {"ppo": PPO, "sac": SAC, "a2c": A2C, "dqn": DQN}
DISCRETE = {"dqn"}
ON_POLICY = {"ppo", "a2c"}
OBS_MODES = ["flat", "sorted", "cluster_summary", "spatial_grid", "graph"]


def _policy_cls_kwargs(obs_mode: str, env):
    """Return (policy_class_str, policy_kwargs_dict) for SB3."""
    if obs_mode == "spatial_grid":
        from uavbench.policies.cnn_policy import CnnExtractor
        return "MultiInputPolicy", {
            "features_extractor_class": CnnExtractor,
            "features_extractor_kwargs": {"features_dim": 128},
        }
    if obs_mode == "graph":
        from uavbench.policies.gnn_policy import GNNExtractor
        mf = env.unwrapped.cfg.max_fleet_size
        return "MultiInputPolicy", {
            "features_extractor_class": GNNExtractor,
            "features_extractor_kwargs": {"features_dim": mf * 64},
        }
    return "MlpPolicy", {}


def _measure_latency_ms(model, eval_env, n: int = 100) -> float:
    obs, _ = eval_env.reset(seed=99_999)
    t0 = time.perf_counter()
    for _ in range(n):
        model.predict(obs, deterministic=True)
    return (time.perf_counter() - t0) * 1000.0 / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo",    choices=list(ALGOS), required=True)
    p.add_argument("--regime",  choices=["static", "mobile", "shock"], default="static")
    p.add_argument("--seed",    type=int, default=0)
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--n-envs",  type=int, default=4)
    p.add_argument("--n-eval",  type=int, default=20)
    p.add_argument("--split",   choices=["test", "val"], default="test")
    p.add_argument("--out",     default=None)
    p.add_argument("--device",  default="cpu")
    p.add_argument("--obs-mode", default="flat", choices=OBS_MODES)
    args = p.parse_args()
    out = args.out or DEFAULT_OUT[args.split]

    discrete = args.algo in DISCRETE
    env_fn = lambda: make_env(args.regime, discrete=discrete, obs_mode=args.obs_mode)

    ref_env = make_env(args.regime, discrete=discrete, obs_mode=args.obs_mode)
    pol_cls, pol_kw = _policy_cls_kwargs(args.obs_mode, ref_env)
    ref_env.close()

    n_envs = args.n_envs if args.algo in ON_POLICY else 1
    train_env = make_vec_env(env_fn, n_envs=n_envs, seed=args.seed)
    model = ALGOS[args.algo](
        pol_cls, train_env,
        seed=args.seed, device=args.device, verbose=0,
        **({"policy_kwargs": pol_kw} if pol_kw else {}),
    )

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    train_time = time.time() - t0

    eval_env = make_env(args.regime, discrete=discrete, obs_mode=args.obs_mode)
    latency_ms = _measure_latency_ms(model, eval_env)

    def act(obs, _base):
        action, _ = model.predict(obs, deterministic=True)
        return action

    covs = [run_episode(eval_env, act, s) for s in layout_seeds(args.split, args.n_eval)]
    res = summarize(covs)
    res.update({
        "split": args.split, "timesteps": args.timesteps,
        "train_time_s": train_time, "inference_latency_ms": latency_ms,
        "obs_mode": args.obs_mode,
        "env_config": eval_env.unwrapped.cfg.to_dict(),
    })
    path = save_result(out, args.regime, args.algo, args.seed, res)

    model_dir = Path(out) / args.regime / args.algo / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / f"seed{args.seed}")

    timing_path = Path(out) / args.regime / args.algo / f"timing_seed{args.seed}.json"
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    timing_path.write_text(json.dumps({
        "algo": args.algo, "regime": args.regime, "seed": args.seed,
        "obs_mode": args.obs_mode, "train_time_s": train_time,
        "inference_latency_ms": latency_ms,
    }, indent=2))

    print(f"{args.regime:7s} {args.algo:4s} seed={args.seed:2d} "
          f"final_cov={res['final_cov']:.3f} avg_cov={res['avg_cov']:.3f} "
          f"train={train_time/60:.1f}min latency={latency_ms:.2f}ms -> {path}")


if __name__ == "__main__":
    main()
