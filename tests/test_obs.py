"""Tests for uavbench/obs.py observation builders and env.py integration."""
import numpy as np
import pytest
from uavbench.env import EnvConfig, UAVPlacementEnv, make_env


def test_spatial_grid_obs_shape():
    env = make_env("static", obs_mode="spatial_grid")
    obs, _ = env.reset(seed=0)
    assert obs["grid"].shape == (1, 32, 32)
    assert obs["uavs"].shape == (env.unwrapped.cfg.n_uavs * 3,)


def test_graph_obs_shape():
    env = make_env("static", obs_mode="graph")
    cfg = env.unwrapped.cfg
    obs, _ = env.reset(seed=0)
    assert obs["node_feats"].shape == (cfg.max_fleet_size + 150, 4)
    assert obs["edge_index"].shape == (2, 4000)
    assert obs["uav_active"].shape == (cfg.max_fleet_size,)
    assert obs["n_nodes"].shape == (1,)
    assert obs["n_edges"].shape == (1,)


def test_spatial_grid_values_in_range():
    env = make_env("static", obs_mode="spatial_grid")
    obs, _ = env.reset(seed=1)
    assert obs["grid"].min() >= 0.0 and obs["grid"].max() <= 1.0
    assert obs["uavs"].min() >= -1.0 and obs["uavs"].max() <= 1.0


def test_graph_node_type_values():
    env = make_env("static", obs_mode="graph")
    cfg = env.unwrapped.cfg
    obs, _ = env.reset(seed=0)
    assert all(obs["node_type"][:cfg.n_uavs] == 1.0)
    start = cfg.max_fleet_size
    assert all(obs["node_type"][start : start + cfg.n_users] == 2.0)


def test_flat_obs_unchanged():
    """Existing obs_modes must produce identical output after refactor."""
    env = make_env("static", obs_mode="flat")
    cfg = env.unwrapped.cfg
    obs, _ = env.reset(seed=42)
    assert obs.shape == (cfg.n_uavs * 3 + cfg.n_users * 2,)


def test_step_returns_dict_for_spatial_grid():
    env = make_env("static", obs_mode="spatial_grid")
    env.reset(seed=0)
    obs, _, _, _, _ = env.step(env.action_space.sample())
    assert isinstance(obs, dict) and "grid" in obs and "uavs" in obs


def test_max_fleet_size_less_than_n_uavs_raises():
    with pytest.raises(ValueError, match="max_fleet_size"):
        make_env("static", obs_mode="graph", n_uavs=9, max_fleet_size=8)
