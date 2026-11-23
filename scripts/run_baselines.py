"""Evaluate classical baselines on the shared evaluation layouts.

Example:
    python scripts/run_baselines.py --regime static --seeds 0 1 2 3 4 5 6 7 8 9
    python scripts/run_baselines.py --regime mobile --replan-every 5 --seeds 0 1 2
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uavbench import make_env  # noqa: E402
from uavbench.baselines import HoverPolicy, TargetTrackingPolicy  # noqa: E402
from uavbench.evaluation import DEFAULT_OUT, layout_seeds, run_episode, save_result, summarize  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regime", choices=["static", "mobile", "shock"], default="static")
    p.add_argument("--algos", nargs="+", default=["kmeans", "pso", "warm_start_pso", "mpc", "random", "lyu_spiral", "hover"])
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--n-eval", type=int, default=20, help="number of evaluation layouts")
    p.add_argument("--replan-every", type=int, default=None,
                   help="re-plan every N steps (default: never for static, 5 for mobile/shock)")
    p.add_argument("--split", choices=["test", "val"], default="test",
                   help="'val' for tuning (e.g. PSO settings), 'test' for reported results")
    p.add_argument("--out", default=None, help="default: results/ (test) or results_val/ (val)")
    p.add_argument("--obs-mode", default="flat", choices=["flat", "sorted", "cluster_summary"],
                   help="Observation representation mode (default: flat)")
    args = p.parse_args()
    out = args.out or DEFAULT_OUT[args.split]

    replan = args.replan_every
    if replan is None and args.regime in ("mobile", "shock"):
        replan = 5

    env = make_env(args.regime, obs_mode=args.obs_mode)
    for algo in args.algos:
        for seed in args.seeds:
            t0 = time.time()
            policy = HoverPolicy() if algo == "hover" else TargetTrackingPolicy(algo, seed=seed, replan_every=replan)

            def act(obs, base, policy=policy):
                if base.t == 0:
                    policy.reset(base)
                return policy.act(base)

            covs = [run_episode(env, act, s) for s in layout_seeds(args.split, args.n_eval)]
            res = summarize(covs)
            res.update({"split": args.split, "replan_every": replan, "wall_time_s": time.time() - t0,
                        "obs_mode": args.obs_mode, "env_config": env.unwrapped.cfg.to_dict()})
            path = save_result(out, args.regime, algo, seed, res)
            print(f"{args.regime:7s} {algo:7s} seed={seed:2d} final_cov={res['final_cov']:.3f} "
                  f"avg_cov={res['avg_cov']:.3f} ({res['wall_time_s']:.1f}s) -> {path}")


if __name__ == "__main__":
    main()
