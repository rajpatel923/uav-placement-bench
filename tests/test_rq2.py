# tests/test_rq2.py
import numpy as np
import pytest
from uavbench.env import EnvConfig, UAVPlacementEnv


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
