#!/usr/bin/env bash
# Full benchmark sweep. Run inside tmux/screen; SAC is the slow one.
# Usage: bash scripts/run_all.sh [n_seeds] [timesteps] [obs_mode]
set -euo pipefail
N_SEEDS=${1:-10}
TIMESTEPS=${2:-200000}
OBS_MODE=${3:-flat}
SEEDS=$(seq 0 $((N_SEEDS - 1)))

for REGIME in static mobile; do
  python scripts/run_baselines.py --regime "$REGIME" --seeds $SEEDS \
    --algos kmeans pso random lyu_spiral hover --obs-mode "$OBS_MODE"
  for ALGO in ppo dqn a2c sac; do
    for S in $SEEDS; do
      python scripts/train_rl.py --algo "$ALGO" --regime "$REGIME" --seed "$S" \
        --timesteps "$TIMESTEPS" --obs-mode "$OBS_MODE"
    done
  done
  python scripts/aggregate.py --regime "$REGIME" --metric final_cov
  python scripts/aggregate.py --regime "$REGIME" --metric avg_cov
  python scripts/plot_figures.py --results results --metric final_cov
done
