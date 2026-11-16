import numpy as np
import pytest
from uavbench.env import UAVPlacementEnv, EnvConfig
from uavbench.baselines import lyu_spiral_planner, random_planner


def _env(seed: int = 0) -> UAVPlacementEnv:
    env = UAVPlacementEnv()
    env.reset(seed=seed)
    return env


def test_lyu_spiral_shape():
    env = _env()
    targets = lyu_spiral_planner(env, np.random.default_rng(0))
    assert targets.shape == (env.cfg.n_uavs, 3)


def test_lyu_spiral_altitude_in_range():
    env = _env()
    targets = lyu_spiral_planner(env, np.random.default_rng(0))
    assert np.all(targets[:, 2] >= env.cfg.h_min - 1e-6)
    assert np.all(targets[:, 2] <= env.cfg.h_max + 1e-6)


def test_lyu_spiral_positions_in_area():
    env = _env()
    targets = lyu_spiral_planner(env, np.random.default_rng(0))
    assert np.all(targets[:, :2] >= -1e-6)
    assert np.all(targets[:, :2] <= env.cfg.area + 1e-6)


def test_lyu_spiral_beats_random_on_average():
    """Geometric spiral placement should beat random over 10 static layouts."""
    spiral_covs, random_covs = [], []
    for seed in range(10):
        env = _env(seed)
        t_spiral = lyu_spiral_planner(env, np.random.default_rng(seed))
        t_random = random_planner(env, np.random.default_rng(seed + 1000))
        spiral_covs.append(env.coverage(t_spiral))
        random_covs.append(env.coverage(t_random))
    assert np.mean(spiral_covs) > np.mean(random_covs), (
        f"Spiral {np.mean(spiral_covs):.3f} should beat random {np.mean(random_covs):.3f}"
    )
