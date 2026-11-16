import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from uavbench import make_env
from uavbench.baselines import HoverPolicy, TargetTrackingPolicy
from uavbench.env import UAVPlacementEnv, EnvConfig
from uavbench.evaluation import run_episode


@pytest.mark.parametrize("regime", ["static", "mobile"])
def test_gymnasium_api(regime):
    check_env(make_env(regime), skip_render_check=True)
    check_env(make_env(regime, discrete=True), skip_render_check=True)


def test_sb3_env_checker():
    from stable_baselines3.common.env_checker import check_env as sb3_check

    sb3_check(make_env("static"), warn=True)


@pytest.mark.parametrize("regime", ["static", "mobile"])
def test_same_seed_same_layout(regime):
    e1, e2 = make_env(regime), make_env(regime)
    o1, _ = e1.reset(seed=123)
    o2, _ = e2.reset(seed=123)
    np.testing.assert_array_equal(o1, o2)
    a = e1.action_space.sample()
    np.testing.assert_array_equal(e1.step(a)[0], e2.step(a)[0])


def test_static_users_do_not_move_mobile_users_do():
    s, m = make_env("static"), make_env("mobile")
    s.reset(seed=1)
    m.reset(seed=1)
    us, um = s.unwrapped.users.copy(), m.unwrapped.users.copy()
    zero = np.zeros(s.action_space.shape, dtype=np.float32)
    s.step(zero)
    m.step(zero)
    np.testing.assert_array_equal(us, s.unwrapped.users)
    assert not np.allclose(um, m.unwrapped.users)


def test_coverage_bounds_and_altitude_limits():
    env = make_env("static")
    env.reset(seed=0)
    for _ in range(20):
        _, r, _, _, info = env.step(env.action_space.sample())
        assert 0.0 <= info["coverage"] <= 1.0
        u = env.unwrapped.uavs
        assert np.all(u[:, 2] >= env.unwrapped.cfg.h_min) and np.all(u[:, 2] <= env.unwrapped.cfg.h_max)


def test_kmeans_beats_hover_on_static():
    """Sanity check: a sensible baseline should beat doing nothing."""
    env = make_env("static")
    km, hover = TargetTrackingPolicy("kmeans", seed=0), HoverPolicy()

    def score(policy, seed):
        def act(obs, base):
            if base.t == 0:
                policy.reset(base)
            return policy.act(base)

        return run_episode(env, act, seed)[-10:].mean()

    seeds = range(5)
    assert np.mean([score(km, s) for s in seeds]) > np.mean([score(hover, s) for s in seeds])


def test_val_and_test_splits_are_disjoint():
    from uavbench.evaluation import layout_seeds

    assert set(layout_seeds("val", 1000)).isdisjoint(layout_seeds("test", 1000))


def test_obs_mode_flat_shape():
    env = UAVPlacementEnv(EnvConfig(obs_mode="flat"))
    obs, _ = env.reset(seed=0)
    cfg = env.cfg
    assert obs.shape == (cfg.n_uavs * 3 + cfg.n_users * 2,)


def test_obs_mode_sorted_shape():
    env = UAVPlacementEnv(EnvConfig(obs_mode="sorted"))
    obs, _ = env.reset(seed=0)
    cfg = env.cfg
    assert obs.shape == (cfg.n_uavs * 3 + cfg.n_users * 2,)


def test_obs_mode_cluster_summary_shape():
    cfg = EnvConfig(obs_mode="cluster_summary")
    env = UAVPlacementEnv(cfg)
    obs, _ = env.reset(seed=0)
    expected = cfg.n_uavs * 3 + cfg.n_clusters * 3
    assert obs.shape == (expected,), f"Expected {expected}, got {obs.shape}"


def test_obs_mode_sorted_is_deterministic():
    env = UAVPlacementEnv(EnvConfig(obs_mode="sorted"))
    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)
    np.testing.assert_array_equal(obs1, obs2)


def test_obs_mode_unknown_raises():
    with pytest.raises(ValueError, match="obs_mode"):
        UAVPlacementEnv(EnvConfig(obs_mode="unknown_mode"))


def test_shock_uav_failure_reduces_active_count():
    cfg = EnvConfig(mobility="shock", shock_step=5, shock_type="uav_failure", shock_n_uavs_fail=1)
    env = UAVPlacementEnv(cfg)
    env.reset(seed=0)
    n_before = env.n_active_uavs
    for _ in range(4):
        env.step(env.action_space.sample())
    assert env.n_active_uavs == n_before, "No UAV should fail before shock_step"
    _, _, _, _, info = env.step(env.action_space.sample())  # step 5 triggers shock
    assert info.get("shock_active") is True
    assert env.n_active_uavs == n_before - 1


def test_shock_user_migration_moves_users():
    cfg = EnvConfig(mobility="shock", shock_step=5, shock_type="user_migration",
                    shock_migration_frac=0.5)
    env = UAVPlacementEnv(cfg)
    env.reset(seed=0)
    users_before = env.users.copy()
    for _ in range(5):
        env.step(env.action_space.sample())
    n_changed = int(np.sum(np.any(env.users != users_before, axis=1)))
    expected_min = int(0.4 * cfg.n_users)
    assert n_changed >= expected_min, f"Expected >={expected_min} relocated users, got {n_changed}"


def test_shock_no_trigger_before_shock_step():
    cfg = EnvConfig(mobility="shock", shock_step=10)
    env = UAVPlacementEnv(cfg)
    env.reset(seed=0)
    for _ in range(9):
        _, _, _, _, info = env.step(env.action_space.sample())
        assert not info.get("shock_active"), f"Shock triggered early at step {env.t}"


def test_shock_all_uavs_fail_zero_coverage():
    """All UAVs failing should give 0 coverage, not crash."""
    cfg = EnvConfig(mobility="shock", shock_step=3, shock_type="uav_failure", shock_n_uavs_fail=4)
    env = UAVPlacementEnv(cfg)
    env.reset(seed=0)
    for _ in range(3):
        obs, rew, term, trunc, info = env.step(env.action_space.sample())
    # After shock fires: n_active_uavs should be 0, coverage should be 0.0, no crash
    assert info["n_active_uavs"] == 0
    assert rew == pytest.approx(0.0, abs=1e-6) or rew <= 0.0  # could be negative if collision penalty
