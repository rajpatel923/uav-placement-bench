"""Gymnasium environment: multi-UAV aerial base station placement.

Regime I  ("static"):  users are placed in clusters at reset and never move.
Regime II ("mobile"):  users move every step (random waypoint).
Regime III ("shock"):  disaster shock — uav_failure or user_migration triggered at a configurable step.

Simplifications you must state in the paper (and may relax later):
  * Coverage = mean Al-Hourani path loss <= pl_max to the best UAV.
  * No interference, no capacity limit per UAV, no backhaul.
  * UAV kinematics: per-step displacement capped at v_max (horizontal) and
    vz_max (vertical); no energy model.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .channel import AlHouraniChannel


@dataclass
class EnvConfig:
    n_uavs: int = 4
    n_users: int = 100
    area: float = 3000.0           # square side length (m); matches Cyprien/Sookhak 2025
    h_min: float = 30.0            # m; matches Cyprien/Sookhak 2025
    h_max: float = 150.0           # m; matches Cyprien/Sookhak 2025
    v_max: float = 60.0            # max horizontal displacement per step (m) (30 m/s x 2 s step)
    vz_max: float = 10.0           # max vertical displacement per step (m)
    pl_max: float = 100.0          # coverage threshold (dB)
    environment: str = "urban"     # Al-Hourani environment
    freq_hz: float = 2.0e9
    episode_len: int = 60          # steps
    # user layout
    n_clusters: int = 6
    cluster_std: float = 150.0     # m
    uniform_frac: float = 0.2      # fraction of users spread uniformly
    # mobility
    mobility: str = "static"       # "static" | "mobile"
    user_speed: float = 10.0       # m per step (random waypoint)
    # reward shaping
    min_sep: float = 20.0          # UAV-UAV safety distance (m)
    collision_penalty: float = 0.1
    obs_mode: str = "flat"  # "flat" | "sorted" | "cluster_summary" | "spatial_grid" | "graph"
    # Graph / GNN obs support
    graph_edge_radius: float = 1500.0  # UAV↔user edge threshold (m)
    max_fleet_size:    int   = 8       # padding target; must be >= n_uavs
    step_duration_s:   float = 1.0     # seconds per env step (user_speed is in m/s)
    # SINR / interference (disabled by default; enabling changes the objective)
    use_sinr: bool = False       # if True, adds sinr_cov to info dict
    tx_power_dbm: float = 30.0  # UAV transmit power (dBm)
    noise_dbm: float = -100.0   # thermal noise floor (dBm)
    sinr_min_db: float = 0.0    # minimum SINR threshold for sinr_cov metric
    # Regime III: disaster shock (stretch goal — requires supervisor approval for results runs)
    shock_step: int = 30               # episode step at which the shock is triggered
    shock_type: str = "uav_failure"    # "uav_failure" | "user_migration"
    shock_n_uavs_fail: int = 1         # number of UAVs that fail (uav_failure type)
    shock_migration_frac: float = 0.5  # fraction of users that relocate (user_migration type)

    def to_dict(self):
        return asdict(self)


class UAVPlacementEnv(gym.Env):
    """Continuous-action env. Action: (n_uavs * 3,) in [-1, 1], scaled to (v_max, v_max, vz_max)."""

    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None, **overrides):
        super().__init__()
        cfg = config or EnvConfig()
        for k, v in overrides.items():
            if not hasattr(cfg, k):
                raise ValueError(f"Unknown config field: {k}")
            setattr(cfg, k, v)
        if cfg.mobility not in ("static", "mobile", "shock"):
            raise ValueError("mobility must be 'static', 'mobile', or 'shock'")
        self.cfg = cfg
        self.channel = AlHouraniChannel(cfg.environment, cfg.freq_hz)
        U, K = cfg.n_uavs, cfg.n_users
        self.action_space = spaces.Box(-1.0, 1.0, shape=(U * 3,), dtype=np.float32)
        from .obs import make_obs_space
        if cfg.max_fleet_size < cfg.n_uavs:
            raise ValueError(
                f"max_fleet_size ({cfg.max_fleet_size}) must be >= n_uavs ({cfg.n_uavs})"
            )
        self.observation_space = make_obs_space(cfg)
        self._step_scale = np.array([cfg.v_max, cfg.v_max, cfg.vz_max], dtype=float)
        self.uavs = np.zeros((U, 3))
        self.users = np.zeros((K, 2))
        self.waypoints = np.zeros((K, 2))
        self.t = 0
        self._failed_mask = np.ones(cfg.n_uavs, dtype=bool)
        self.n_active_uavs: int = cfg.n_uavs

    # ---------------------------------------------------------------- helpers
    def _sample_users(self) -> np.ndarray:
        c, rng = self.cfg, self.np_random
        n_uniform = int(round(c.uniform_frac * c.n_users))
        n_clustered = c.n_users - n_uniform
        margin = 2 * c.cluster_std
        centers = rng.uniform(margin, c.area - margin, size=(c.n_clusters, 2))
        labels = rng.integers(0, c.n_clusters, size=n_clustered)
        clustered = centers[labels] + rng.normal(0.0, c.cluster_std, size=(n_clustered, 2))
        uniform = rng.uniform(0.0, c.area, size=(n_uniform, 2))
        return np.clip(np.vstack([clustered, uniform]), 0.0, c.area)

    def _obs(self):
        from .obs import (build_flat, build_sorted, build_cluster_summary,
                          build_spatial_grid, build_graph)
        c = self.cfg
        u = self.uavs.copy()
        u[:, :2] /= c.area
        u[:, 2] = (u[:, 2] - c.h_min) / max(c.h_max - c.h_min, 1e-6)
        if c.obs_mode == "flat":
            return build_flat(u, self.users, c)
        if c.obs_mode == "sorted":
            return build_sorted(u, self.users, c)
        if c.obs_mode == "cluster_summary":
            return build_cluster_summary(u, self.users, c)
        if c.obs_mode == "spatial_grid":
            return build_spatial_grid(self.uavs, self.users, c)
        if c.obs_mode == "graph":
            return build_graph(self.uavs, self.users, c, self._failed_mask)
        raise ValueError(f"Unknown obs_mode: {c.obs_mode!r}")

    def coverage(self, uavs: np.ndarray | None = None, users: np.ndarray | None = None) -> float:
        """Fraction of users whose best-UAV path loss is <= pl_max."""
        uavs = self.uavs if uavs is None else uavs
        if uavs.shape[0] == 0:
            return 0.0
        users = self.users if users is None else users
        pl = self.channel.path_loss_matrix(uavs, users)
        return float(np.mean(pl.min(axis=0) <= self.cfg.pl_max))

    def sinr_coverage(self, uavs: np.ndarray | None = None, users: np.ndarray | None = None) -> float:
        """Fraction of users with SINR >= sinr_min_db. Only meaningful when use_sinr=True."""
        uavs = self.uavs if uavs is None else uavs
        if uavs.shape[0] == 0:
            return 0.0
        users = self.users if users is None else users
        sinr = self.channel.sinr_matrix(uavs, users, self.cfg.tx_power_dbm, self.cfg.noise_dbm)
        return float(np.mean(sinr >= self.cfg.sinr_min_db))

    def _too_close_frac(self) -> float:
        U = self.cfg.n_uavs
        if U < 2:
            return 0.0
        d = np.linalg.norm(self.uavs[:, None, :] - self.uavs[None, :, :], axis=-1)
        iu = np.triu_indices(U, k=1)
        return float(np.mean(d[iu] < self.cfg.min_sep))

    def _move_users(self):
        c, rng = self.cfg, self.np_random
        step_m = c.user_speed * c.step_duration_s      # m/s × s = m/step
        delta = self.waypoints - self.users
        dist = np.linalg.norm(delta, axis=1, keepdims=True)
        arrived = dist[:, 0] <= step_m
        step = np.where(dist > 0, delta / np.maximum(dist, 1e-9), 0.0) * np.minimum(dist, step_m)
        self.users = self.users + step
        if arrived.any():
            self.waypoints[arrived] = rng.uniform(0.0, c.area, size=(int(arrived.sum()), 2))

    def _apply_shock(self) -> None:
        c = self.cfg
        if c.shock_type == "uav_failure":
            active = np.where(self._failed_mask)[0]
            n_fail = min(c.shock_n_uavs_fail, len(active))
            if n_fail > 0:
                fail_idxs = self.np_random.choice(active, n_fail, replace=False)
                self._failed_mask[fail_idxs] = False
                self.n_active_uavs -= n_fail
        elif c.shock_type == "user_migration":
            n_move = int(round(c.shock_migration_frac * c.n_users))
            idxs = self.np_random.choice(c.n_users, n_move, replace=False)
            self.users[idxs] = self.np_random.uniform(0.0, c.area, size=(n_move, 2))
            self.waypoints[idxs] = self.np_random.uniform(0.0, c.area, size=(n_move, 2))

    # ------------------------------------------------------------------- API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        c = self.cfg
        self.users = self._sample_users()
        self.waypoints = self.np_random.uniform(0.0, c.area, size=(c.n_users, 2))
        self.uavs = np.column_stack(
            [
                self.np_random.uniform(0.0, c.area, size=c.n_uavs),
                self.np_random.uniform(0.0, c.area, size=c.n_uavs),
                np.full(c.n_uavs, 0.5 * (c.h_min + c.h_max)),
            ]
        )
        self.t = 0
        self._failed_mask = np.ones(c.n_uavs, dtype=bool)
        self.n_active_uavs = c.n_uavs
        cov = self.coverage()
        info = {"coverage": cov}
        if c.use_sinr:
            info["sinr_cov"] = self.sinr_coverage()
        return self._obs(), info

    def step(self, action):
        c = self.cfg
        a = np.clip(np.asarray(action, dtype=float).reshape(c.n_uavs, 3), -1.0, 1.0)
        self.uavs = self.uavs + a * self._step_scale
        self.uavs[:, :2] = np.clip(self.uavs[:, :2], 0.0, c.area)
        self.uavs[:, 2] = np.clip(self.uavs[:, 2], c.h_min, c.h_max)
        self.t += 1
        shock_active = False
        if c.mobility == "shock" and self.t == c.shock_step:
            self._apply_shock()
            shock_active = True
        if c.mobility in ("mobile", "shock"):
            self._move_users()
        active_uavs = self.uavs[self._failed_mask] if c.mobility == "shock" else self.uavs
        cov = self.coverage(active_uavs)
        # Proximity penalty uses all UAVs (including failed ones) — failed UAVs are
        # still physically present. For shock experiments, this biases the reward slightly.
        close = self._too_close_frac()
        reward = cov - c.collision_penalty * close
        truncated = self.t >= c.episode_len
        info = {"coverage": cov, "too_close_frac": close, "shock_active": shock_active,
                "n_active_uavs": self.n_active_uavs}
        if c.use_sinr:
            info["sinr_cov"] = self.sinr_coverage(active_uavs)
        return self._obs(), float(reward), False, truncated, info


class DiscreteActionWrapper(gym.ActionWrapper):
    """Joint discrete actions for DQN (SB3's DQN supports only spaces.Discrete).

    Each UAV picks one of 7 moves: +x, -x, +y, -y, +z, -z, hover.
    Joint space size = 7 ** n_uavs (2401 for 4 UAVs). This blows up fast; for
    more UAVs switch to a multi-agent formulation.
    """

    MOVES = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1], [0, 0, 0]],
        dtype=np.float32,
    )

    def __init__(self, env: UAVPlacementEnv):
        super().__init__(env)
        self.n_uavs = env.unwrapped.cfg.n_uavs
        self.action_space = spaces.Discrete(7**self.n_uavs)

    def action(self, act):
        idx = int(act)
        moves = []
        for _ in range(self.n_uavs):
            moves.append(self.MOVES[idx % 7])
            idx //= 7
        return np.concatenate(moves)


def make_env(regime: str = "static", discrete: bool = False, **overrides) -> gym.Env:
    """Factory used by all scripts so RL and baselines see identical environments.

    regime: "static" | "mobile" | "shock"
    """
    env = UAVPlacementEnv(EnvConfig(mobility=regime), **overrides)
    return DiscreteActionWrapper(env) if discrete else env
