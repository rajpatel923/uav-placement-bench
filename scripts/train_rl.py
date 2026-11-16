"""Train one RL agent (one algorithm x one regime x one seed) and evaluate it.

Example:
    python scripts/train_rl.py --algo ppo --regime static --seed 0 --timesteps 200000
    for s in 0 1 2 3 4 5 6 7 8 9; do python scripts/train_rl.py --algo sac --regime mobile --seed $s; done

Fairness rules (do not break these silently):
  * Same env factory, same evaluation layouts, same metrics as the baselines.
  * Give every algorithm the same tuning budget; record what you tuned.
  * Tune with --split val (results_val/); report only --split test (results/).
  * Evaluate with deterministic=True on held-out layouts.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_baselines3 import DQN, PPO, SAC, A2C  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402

from uavbench import make_env  # noqa: E402
from uavbench.evaluation import DEFAULT_OUT, layout_seeds, run_episode, save_result, summarize  # noqa: E402

ALGOS = {"ppo": PPO, "sac": SAC, "a2c": A2C, "dqn": DQN}
DISCRETE = {"dqn"}
ON_POLICY = {"ppo", "a2c"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=list(ALGOS), required=True)
    p.add_argument("--regime", choices=["static", "mobile", "shock"], default="static")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--n-envs", type=int, default=4, help="parallel envs (on-policy algos only)")
    p.add_argument("--n-eval", type=int, default=20)
    p.add_argument("--split", choices=["test", "val"], default="test",
                   help="'val' while tuning, 'test' only for final reported runs")
    p.add_argument("--out", default=None, help="default: results/ (test) or results_val/ (val)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--obs-mode", default="flat", choices=["flat", "sorted", "cluster_summary"],
                   help="Observation representation mode (default: flat)")
    args = p.parse_args()
    out = args.out or DEFAULT_OUT[args.split]

    discrete = args.algo in DISCRETE
    env_fn = lambda: make_env(args.regime, discrete=discrete, obs_mode=args.obs_mode)
    n_envs = args.n_envs if args.algo in ON_POLICY else 1
    train_env = make_vec_env(env_fn, n_envs=n_envs, seed=args.seed)

    # SB3 defaults on purpose. If you tune, tune every algorithm with the same budget.
    model = ALGOS[args.algo]("MlpPolicy", train_env, seed=args.seed, device=args.device, verbose=0)

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    train_time = time.time() - t0

    eval_env = make_env(args.regime, discrete=discrete, obs_mode=args.obs_mode)

    def act(obs, base):
        action, _ = model.predict(obs, deterministic=True)
        return action

    covs = [run_episode(eval_env, act, s) for s in layout_seeds(args.split, args.n_eval)]
    res = summarize(covs)
    res.update({"split": args.split, "timesteps": args.timesteps, "train_time_s": train_time,
                "obs_mode": args.obs_mode, "env_config": eval_env.unwrapped.cfg.to_dict()})
    path = save_result(out, args.regime, args.algo, args.seed, res)

    model_dir = Path(out) / args.regime / args.algo / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / f"seed{args.seed}")
    print(f"{args.regime:7s} {args.algo:4s} seed={args.seed:2d} final_cov={res['final_cov']:.3f} "
          f"avg_cov={res['avg_cov']:.3f} train={train_time/60:.1f} min -> {path}")


if __name__ == "__main__":
    main()
