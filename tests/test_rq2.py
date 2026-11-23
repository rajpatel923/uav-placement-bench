# tests/test_rq2.py
import numpy as np
import pytest
from uavbench.env import EnvConfig, UAVPlacementEnv, make_env


def test_new_fields_exist_with_defaults():
    cfg = EnvConfig()
    assert cfg.single_failure is False
    assert cfg.failure_rate == 0.0


def test_no_failure_by_default():
    env = UAVPlacementEnv(EnvConfig(mobility="static"))
    env.reset(seed=0)
    n = env.cfg.n_uavs
    for _ in range(env.cfg.episode_len):
        _, _, _, _, info = env.step(env.action_space.sample())
        assert info["n_active_uavs"] == n


def test_single_failure_fires_exactly_once():
    env = UAVPlacementEnv(EnvConfig(single_failure=True, mobility="static"))
    for ep in range(30):
        env.reset(seed=ep)
        failures = 0
        prev = env.cfg.n_uavs
        for _ in range(env.cfg.episode_len):
            _, _, _, _, info = env.step(env.action_space.sample())
            if info["n_active_uavs"] < prev:
                failures += 1
                prev = info["n_active_uavs"]
        assert failures == 1, f"Episode {ep}: expected 1 failure, got {failures}"


def test_single_failure_timing():
    env = UAVPlacementEnv(EnvConfig(single_failure=True, mobility="static"))
    ep_len = env.cfg.episode_len
    lo, hi = ep_len // 3, 2 * ep_len // 3
    for ep in range(20):
        env.reset(seed=ep * 100)
        prev = env.cfg.n_uavs
        for t in range(ep_len):
            _, _, _, _, info = env.step(env.action_space.sample())
            if info["n_active_uavs"] < prev:
                assert lo <= t <= hi, f"Ep {ep}: failure at step {t}, want [{lo},{hi}]"
                break


def test_single_failure_with_mobile():
    env = UAVPlacementEnv(EnvConfig(single_failure=True, mobility="mobile"))
    env.reset(seed=0)
    losses, prev = 0, env.cfg.n_uavs
    for _ in range(env.cfg.episode_len):
        _, _, _, _, info = env.step(env.action_space.sample())
        if info["n_active_uavs"] < prev:
            losses += 1; prev = info["n_active_uavs"]
    assert losses == 1


def test_failure_rate_zero_no_change():
    env = UAVPlacementEnv(EnvConfig(failure_rate=0.0))
    env.reset(seed=0)
    n = env.cfg.n_uavs
    for _ in range(env.cfg.episode_len):
        _, _, _, _, info = env.step(env.action_space.sample())
        assert info["n_active_uavs"] == n


def test_failure_rate_nonzero():
    env = make_env("static", failure_rate=1.0)
    env.reset(seed=0)
    for _ in range(5):
        env.step(env.action_space.sample())
    # With failure_rate=1.0, at least one UAV should have failed
    assert not env.unwrapped._failed_mask.all(), "failure_rate=1.0 should fail at least one UAV"


def test_failure_affects_coverage():
    # Both envs see the same user layout (seed=42) and the same actions
    # (from an external rng), so the only difference is the missing UAV.
    env_ok = make_env("static")
    env_ok.reset(seed=42)

    env_fail = make_env("static", single_failure=True)
    env_fail.reset(seed=42)
    # Force early failure so nearly every step has fewer active UAVs
    env_fail.unwrapped._failure_step = 2

    rng = np.random.default_rng(99)
    ep_len = env_ok.unwrapped.cfg.episode_len
    cov_ok_sum = 0.0
    cov_fail_sum = 0.0
    for _ in range(ep_len):
        action = rng.uniform(-1.0, 1.0, env_ok.action_space.shape).astype(np.float32)
        _, _, term_ok, trunc_ok, info_ok = env_ok.step(action)
        _, _, term_fail, trunc_fail, info_fail = env_fail.step(action)
        cov_ok_sum += info_ok["coverage"]
        cov_fail_sum += info_fail["coverage"]

    # With fewer active UAVs (failure at step 2), mean coverage should be lower
    assert cov_fail_sum <= cov_ok_sum + 1e-6, (
        f"Failure should reduce coverage: fail={cov_fail_sum:.3f}, ok={cov_ok_sum:.3f}"
    )
