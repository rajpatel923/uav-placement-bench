"""Tests for CNN feature extractor (Task 2) and GNN feature extractor (Task 3)."""
import numpy as np
import pytest
import torch
from uavbench.env import make_env


def _cnn_env():
    return make_env("static", obs_mode="spatial_grid")


def test_cnn_extractor_output_shape():
    from uavbench.policies.cnn_policy import CnnExtractor
    env = _cnn_env()
    extractor = CnnExtractor(env.observation_space)
    n_uavs = env.unwrapped.cfg.n_uavs
    batch = {"grid": torch.zeros(2, 1, 32, 32), "uavs": torch.zeros(2, n_uavs * 3)}
    out = extractor(batch)
    assert out.shape == (2, 128), f"Expected (2,128), got {out.shape}"


def test_cnn_extractor_no_nan():
    from uavbench.policies.cnn_policy import CnnExtractor
    env = _cnn_env()
    extractor = CnnExtractor(env.observation_space)
    obs, _ = env.reset(seed=0)
    batch = {k: torch.tensor(v).unsqueeze(0) for k, v in obs.items()}
    out = extractor(batch)
    assert not torch.isnan(out).any()


def test_cnn_policy_integration():
    from stable_baselines3 import PPO
    from uavbench.policies.cnn_policy import CnnExtractor
    env = _cnn_env()
    model = PPO(
        "MultiInputPolicy", env, verbose=0, n_steps=64,
        policy_kwargs={"features_extractor_class": CnnExtractor,
                       "features_extractor_kwargs": {"features_dim": 128}},
    )
    obs, _ = env.reset(seed=0)
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == env.action_space.shape


# ── GNN feature extractor tests (Task 3) ─────────────────────────────────────

def _gnn_env(n_uavs=4):
    return make_env("static", obs_mode="graph", n_uavs=n_uavs)


def test_gnn_extractor_4uav_shape():
    from uavbench.policies.gnn_policy import GNNExtractor
    env = _gnn_env(n_uavs=4)
    extractor = GNNExtractor(env.observation_space)
    obs, _ = env.reset(seed=0)
    batch = {k: torch.tensor(v).unsqueeze(0) for k, v in obs.items()}
    out = extractor(batch)
    max_fleet = env.unwrapped.cfg.max_fleet_size
    assert out.shape == (1, max_fleet * 64), f"Got {out.shape}"


def test_gnn_extractor_6uav_same_shape():
    """6-UAV obs through an extractor built on 4-UAV space → same output shape."""
    from uavbench.policies.gnn_policy import GNNExtractor
    env4 = _gnn_env(n_uavs=4)
    env6 = _gnn_env(n_uavs=6)
    extractor = GNNExtractor(env4.observation_space)
    obs6, _ = env6.reset(seed=0)
    batch = {k: torch.tensor(v).unsqueeze(0) for k, v in obs6.items()}
    out = extractor(batch)
    max_fleet = env4.unwrapped.cfg.max_fleet_size
    assert out.shape == (1, max_fleet * 64)


def test_gnn_extractor_no_nan():
    from uavbench.policies.gnn_policy import GNNExtractor
    env = _gnn_env()
    extractor = GNNExtractor(env.observation_space)
    obs, _ = env.reset(seed=0)
    batch = {k: torch.tensor(v).unsqueeze(0) for k, v in obs.items()}
    assert not torch.isnan(extractor(batch)).any()


def test_gnn_inactive_uavs_zeroed():
    """Slots with uav_active=0 must produce zero embedding."""
    from uavbench.policies.gnn_policy import GNNExtractor
    env = _gnn_env(n_uavs=4)
    obs, _ = env.reset(seed=0)
    obs_copy = {k: v.copy() for k, v in obs.items()}
    obs_copy["uav_active"][1:] = 0.0            # keep only UAV 0 active
    extractor = GNNExtractor(env.observation_space)
    extractor.eval()
    with torch.no_grad():
        batch = {k: torch.tensor(v).unsqueeze(0) for k, v in obs_copy.items()}
        out = extractor(batch)
    max_fleet = env.unwrapped.cfg.max_fleet_size
    per_uav = out.view(1, max_fleet, 64)
    assert torch.allclose(per_uav[0, 1:], torch.zeros(max_fleet - 1, 64), atol=1e-6)


def test_gnn_policy_integration():
    from stable_baselines3 import PPO
    from uavbench.policies.gnn_policy import GNNExtractor
    env = _gnn_env(n_uavs=4)
    max_fleet = env.unwrapped.cfg.max_fleet_size
    model = PPO(
        "MultiInputPolicy", env, verbose=0, n_steps=64,
        policy_kwargs={"features_extractor_class": GNNExtractor,
                       "features_extractor_kwargs": {"features_dim": max_fleet * 64}},
    )
    obs, _ = env.reset(seed=0)
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == env.action_space.shape
