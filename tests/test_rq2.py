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
    # Run with no failure
    env_ok = make_env("static")
    obs_ok, _ = env_ok.reset(seed=42)
    rew_ok = 0.0
    for _ in range(env_ok.unwrapped.cfg.episode_len):
        _, r, term, trunc, _ = env_ok.step(env_ok.action_space.sample())
        rew_ok += r
        if term or trunc:
            break

    # Run with single_failure — seed so failure fires early
    env_fail = make_env("static", single_failure=True)
    obs_fail, _ = env_fail.reset(seed=42)
    # Force early failure by setting _failure_step to step 2
    env_fail.unwrapped._failure_step = 2
    rew_fail = 0.0
    for _ in range(env_fail.unwrapped.cfg.episode_len):
        _, r, term, trunc, _ = env_fail.step(env_fail.action_space.sample())
        rew_fail += r
        if term or trunc:
            break

    # Coverage with failure should be <= coverage without failure
    assert rew_fail <= rew_ok + 1e-6, (
        f"Failure should not improve coverage: fail={rew_fail:.3f}, ok={rew_ok:.3f}"
    )
