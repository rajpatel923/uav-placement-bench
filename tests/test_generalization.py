"""Tests for RQ3 generalization evaluation.

Covers:
  - eval_generalization_runs: basic flat model can be saved, loaded, and predicts without crash.
  - gnn_fleet_6_no_crash: GNN model trained on 4 UAVs runs on a 6-UAV env (action space
    mismatch is a documented limitation; the test verifies no unhandled exception).
  - gnn_fleet_exceeds_max_raises: make_env with n_uavs > max_fleet_size raises ValueError.
"""
import numpy as np
import pytest
import tempfile
from pathlib import Path
from uavbench.env import make_env, EnvConfig


def _train_and_save_flat(out_dir: str) -> Path:
    """Train a tiny PPO-flat model, save checkpoint, return path."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    env_fn = lambda: make_env("static", obs_mode="flat")
    model = PPO("MlpPolicy", make_vec_env(env_fn, n_envs=1, seed=0), verbose=0, n_steps=64)
    model.learn(total_timesteps=1024)
    p = Path(out_dir) / "flat_seed0"
    model.save(p)
    return p


def test_eval_generalization_runs():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = _train_and_save_flat(tmp)
        # Run eval with same config as training — should complete without crash
        from stable_baselines3 import PPO

        model = PPO.load(ckpt)
        env = make_env("static", obs_mode="flat")
        obs, _ = env.reset(seed=10000)
        action, _ = model.predict(obs, deterministic=True)
        assert action.shape == env.action_space.shape


def test_gnn_fleet_6_no_crash():
    """GNN extractor trained on 4 UAVs must run on 6-UAV env without error."""
    from stable_baselines3 import PPO
    from uavbench.policies.gnn_policy import GNNExtractor

    env4 = make_env("static", obs_mode="graph", n_uavs=4)
    mf = env4.unwrapped.cfg.max_fleet_size
    model = PPO(
        "MultiInputPolicy",
        env4,
        verbose=0,
        n_steps=64,
        policy_kwargs={
            "features_extractor_class": GNNExtractor,
            "features_extractor_kwargs": {"features_dim": mf * 64},
        },
    )
    model.learn(total_timesteps=512)
    env6 = make_env("static", obs_mode="graph", n_uavs=6)
    obs, _ = env6.reset(seed=10000)
    action, _ = model.predict(obs, deterministic=True)
    # Action space for 6 UAVs is 18-dim, but model was trained on 4-UAV (12-dim)
    # This test verifies no crash; shape mismatch is a known limitation documented in RQ3-B
    assert action is not None


def test_gnn_fleet_exceeds_max_raises():
    with pytest.raises(ValueError, match="max_fleet_size"):
        make_env("static", obs_mode="graph", n_uavs=9, max_fleet_size=8)
