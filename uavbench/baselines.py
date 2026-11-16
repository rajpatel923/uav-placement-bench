"""Classical (non-RL) baselines, run as policies INSIDE the same environment.

Every baseline obeys the same flight limits as the RL agents: it computes target
positions, matches UAVs to targets (Hungarian algorithm), and flies toward them
at most v_max per step. This avoids the unfair "teleporting baseline" comparison.

Altitude rule (from Al-Hourani et al. 2014): for a cluster of radius R_c, the
altitude that minimizes the mean path loss at the cluster edge is
h = R_c * tan(theta_opt), where theta_opt solves Eq. (13). We clip to [h_min, h_max].
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from .env import UAVPlacementEnv


# ----------------------------------------------------------------- planners
def kmeans_planner(env: UAVPlacementEnv, rng: np.random.Generator, edge_quantile: float = 0.9):
    c = env.cfg
    km = KMeans(n_clusters=c.n_uavs, n_init=10, random_state=int(rng.integers(1 << 31)))
    labels = km.fit_predict(env.users)
    theta = np.radians(env.channel.optimal_elevation_deg())
    targets = np.zeros((c.n_uavs, 3))
    for j in range(c.n_uavs):
        members = env.users[labels == j]
        center = km.cluster_centers_[j]
        if len(members) == 0:
            r_c = 0.0
        else:
            r_c = float(np.quantile(np.linalg.norm(members - center, axis=1), edge_quantile))
        targets[j, :2] = center
        targets[j, 2] = np.clip(r_c * np.tan(theta), c.h_min, c.h_max)
    return targets


def _soft_coverage(env: UAVPlacementEnv, uavs: np.ndarray, temp_db: float = 2.0) -> float:
    """Smooth surrogate of coverage (sigmoid of path-loss margin) so PSO gets a gradient-like signal."""
    pl = env.channel.path_loss_matrix(uavs, env.users).min(axis=0)
    return float(np.mean(1.0 / (1.0 + np.exp((pl - env.cfg.pl_max) / temp_db))))


def pso_planner(
    env: UAVPlacementEnv,
    rng: np.random.Generator,
    n_particles: int = 30,
    iters: int = 60,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
):
    """Plain global-best particle swarm optimization over all UAV coordinates."""
    c = env.cfg
    lo = np.tile([0.0, 0.0, c.h_min], c.n_uavs)
    hi = np.tile([c.area, c.area, c.h_max], c.n_uavs)
    dim = lo.size
    x = rng.uniform(lo, hi, size=(n_particles, dim))
    v = rng.normal(0.0, 0.1, size=(n_particles, dim)) * (hi - lo)
    score = lambda p: _soft_coverage(env, p.reshape(c.n_uavs, 3))
    pbest = x.copy()
    pbest_f = np.array([score(p) for p in x])
    g = pbest[np.argmax(pbest_f)].copy()
    for _ in range(iters):
        r1, r2 = rng.random((2, n_particles, dim))
        v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (g - x)
        x = np.clip(x + v, lo, hi)
        f = np.array([score(p) for p in x])
        better = f > pbest_f
        pbest[better], pbest_f[better] = x[better], f[better]
        g = pbest[np.argmax(pbest_f)].copy()
    return g.reshape(c.n_uavs, 3)


def random_planner(env: UAVPlacementEnv, rng: np.random.Generator):
    c = env.cfg
    return np.column_stack(
        [
            rng.uniform(0, c.area, c.n_uavs),
            rng.uniform(0, c.area, c.n_uavs),
            rng.uniform(c.h_min, c.h_max, c.n_uavs),
        ]
    )


def lyu_spiral_planner(env: UAVPlacementEnv, rng: np.random.Generator):
    """Place UAVs on a Fermat spiral centred on the area, altitude from the Al-Hourani rule.

    Lyu et al. (IEEE Commun. Lett., ~2016/2017) distribute a UAV base station along a
    spiral trajectory. For multi-UAV placement we place U points at golden-angle spacing
    on a Fermat spiral scaled to 90 % of the half-area diagonal, with altitude h = r * tan(θ_opt)
    clipped to [h_min, h_max].  Verify the multi-UAV formulation against the paper before
    citing these results as "Lyu et al. baseline."
    """
    c = env.cfg
    theta_opt = np.radians(env.channel.optimal_elevation_deg())
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))  # ~137.508°
    cx, cy = c.area / 2.0, c.area / 2.0
    max_r = c.area / 2.0 * 0.9  # 90 % of half-side to keep positions inside the area

    targets = np.zeros((c.n_uavs, 3))
    for i in range(c.n_uavs):
        angle = i * golden_angle
        r = max_r * np.sqrt((i + 0.5) / c.n_uavs)
        targets[i, 0] = np.clip(cx + r * np.cos(angle), 0.0, c.area)
        targets[i, 1] = np.clip(cy + r * np.sin(angle), 0.0, c.area)
        targets[i, 2] = np.clip(r * np.tan(theta_opt), c.h_min, c.h_max)
    return targets


PLANNERS = {
    "kmeans": kmeans_planner,
    "pso": pso_planner,
    "random": random_planner,
    "lyu_spiral": lyu_spiral_planner,
}


# ------------------------------------------------------------------ policy
class TargetTrackingPolicy:
    """Plan targets, match UAVs to targets, fly toward them under velocity limits.

    replan_every: re-run the planner every N steps (use 1-10 for mobile users;
    in the static regime planning once at reset is enough).
    """

    def __init__(self, planner: str, seed: int = 0, replan_every: int | None = None):
        self.planner = PLANNERS[planner]
        self.rng = np.random.default_rng(seed)
        self.replan_every = replan_every
        self.targets = None
        self._t = 0

    def reset(self, env: UAVPlacementEnv):
        self._t = 0
        self._plan(env)

    def _plan(self, env: UAVPlacementEnv):
        targets = self.planner(env, self.rng)
        cost = np.linalg.norm(env.uavs[:, None, :] - targets[None, :, :], axis=-1)
        rows, cols = linear_sum_assignment(cost)
        self.targets = targets[cols[np.argsort(rows)]]

    def act(self, env: UAVPlacementEnv) -> np.ndarray:
        if self.replan_every and self._t > 0 and self._t % self.replan_every == 0:
            self._plan(env)
        self._t += 1
        delta = (self.targets - env.uavs) / env._step_scale
        return np.clip(delta, -1.0, 1.0).ravel().astype(np.float32)


class HoverPolicy:
    """Do nothing. Lower bound: coverage from the random initial positions."""

    def reset(self, env):
        pass

    def act(self, env):
        return np.zeros(env.action_space.shape, dtype=np.float32)
