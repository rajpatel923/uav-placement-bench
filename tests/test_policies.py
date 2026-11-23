"""Tests for CNN feature extractor (Task 2)."""
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
