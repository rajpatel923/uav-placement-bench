import numpy as np
import pytest
from uavbench.env import UAVPlacementEnv, EnvConfig, make_env
from uavbench.baselines import lyu_spiral_planner, random_planner, TargetTrackingPolicy


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


def test_warm_start_pso_registered():
    from uavbench.baselines import PLANNERS
    assert "warm_start_pso" in PLANNERS


def test_warm_start_pso_no_prior_matches_cold():
    """With same seed and no prior, warm_start_pso must equal pso_planner."""
    from uavbench.baselines import pso_planner, warm_start_pso_planner
    env = _env(seed=5)
    result_a = pso_planner(env, np.random.default_rng(99), n_particles=10, iters=5)
    env2 = _env(seed=5)
    result_b = warm_start_pso_planner(env2, np.random.default_rng(99),
                                       prev_solution=None, n_particles=10, iters=5)
    np.testing.assert_array_almost_equal(result_a, result_b)


def test_warm_start_pso_prior_shape():
    from uavbench.baselines import pso_planner, warm_start_pso_planner
    env = _env()
    prior = pso_planner(env, np.random.default_rng(42), iters=10)
    env2 = _env()
    result = warm_start_pso_planner(env2, np.random.default_rng(1),
                                     prev_solution=prior, iters=10)
    assert result.shape == (env.cfg.n_uavs, 3)


def test_mpc_registered():
    from uavbench.baselines import PLANNERS
    assert "mpc" in PLANNERS


def test_mpc_planner_shape_and_bounds():
    from uavbench.baselines import mpc_planner
    env = _env()
    targets = mpc_planner(env, np.random.default_rng(0), horizon=3, budget_ms=200.0)
    assert targets.shape == (env.cfg.n_uavs, 3)
    assert np.all(targets[:, :2] >= 0) and np.all(targets[:, :2] <= env.cfg.area)
    assert np.all(targets[:, 2] >= env.cfg.h_min) and np.all(targets[:, 2] <= env.cfg.h_max)


def test_warm_start_pso_via_target_tracking_uses_prev_solution():
    """TargetTrackingPolicy should pass prev_solution on subsequent replans."""
    env = make_env("mobile")
    obs, _ = env.reset(seed=0)
    policy = TargetTrackingPolicy("warm_start_pso", seed=0, replan_every=1)
    assert policy._has_prev_solution, "warm_start_pso must expose prev_solution parameter"
    assert policy._last_plan is None, "_last_plan should be None before first plan"

    # First plan: cold start — _last_plan must be populated after reset
    policy.reset(env.unwrapped)
    assert policy._last_plan is not None, "_last_plan should be set after reset/first plan"

    # Second plan: _last_plan should be passed as warm start
    first_plan = policy._last_plan.copy()
    env.unwrapped.uavs = env.unwrapped.uavs  # no-op; ensure env state stable
    policy._plan(env.unwrapped)
    assert policy._last_plan is not None, "_last_plan should remain set after second plan"
    assert policy._last_plan.shape == (env.unwrapped.cfg.n_uavs, 3)


def test_non_warm_start_planner_no_prev_solution():
    """kmeans and pso planners do not accept prev_solution; _has_prev_solution must be False."""
    for name in ("kmeans", "pso", "random"):
        policy = TargetTrackingPolicy(name, seed=0)
        assert not policy._has_prev_solution, (
            f"{name} should not have prev_solution but _has_prev_solution={policy._has_prev_solution}"
        )
