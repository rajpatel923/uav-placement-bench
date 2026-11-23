"""Observation builders for all 5 obs_modes.

Called by UAVPlacementEnv._obs(). Returns ndarray for flat/sorted/cluster_summary,
dict[str, ndarray] for spatial_grid and graph.
"""
from __future__ import annotations

import numpy as np
from gymnasium import spaces


MAX_OBS_USERS = 150  # hard ceiling — matches max n_users in RQ3-A experiments


# ── vector obs modes ─────────────────────────────────────────────────────────

def build_flat(uav_norm: np.ndarray, users: np.ndarray, cfg) -> np.ndarray:
    return np.concatenate([uav_norm.ravel(), (users / cfg.area).ravel()]).astype(np.float32)


def build_sorted(uav_norm: np.ndarray, users: np.ndarray, cfg) -> np.ndarray:
    centroid = np.full(2, cfg.area / 2.0)
    order = np.argsort(np.linalg.norm(users - centroid, axis=1))
    return np.concatenate([uav_norm.ravel(), (users[order] / cfg.area).ravel()]).astype(np.float32)


def build_cluster_summary(uav_norm: np.ndarray, users: np.ndarray, cfg) -> np.ndarray:
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=cfg.n_clusters, n_init=1, random_state=0)
    labels = km.fit_predict(users)
    counts = np.bincount(labels, minlength=cfg.n_clusters)
    summary = np.column_stack([
        km.cluster_centers_ / cfg.area,
        counts.astype(float) / cfg.n_users,
    ])
    return np.concatenate([uav_norm.ravel(), summary.ravel()]).astype(np.float32)


# ── spatial_grid ─────────────────────────────────────────────────────────────

def build_spatial_grid(uavs: np.ndarray, users: np.ndarray, cfg) -> dict:
    """32×32 binary occupancy grid of users + normalised UAV vector in [-1, 1]."""
    grid = np.zeros((1, 32, 32), dtype=np.float32)
    idx = np.floor(users / cfg.area * 32).astype(int).clip(0, 31)
    grid[0, idx[:, 1], idx[:, 0]] = 1.0  # row=y, col=x

    u = uavs.astype(np.float64).copy()
    u[:, :2] = u[:, :2] / cfg.area * 2.0 - 1.0
    u[:, 2] = (u[:, 2] - cfg.h_min) / max(cfg.h_max - cfg.h_min, 1e-6) * 2.0 - 1.0
    return {"grid": grid, "uavs": u.ravel().astype(np.float32)}


# ── graph ─────────────────────────────────────────────────────────────────────

def build_graph(
    uavs: np.ndarray,
    users: np.ndarray,
    cfg,
    failed_mask: np.ndarray,
) -> dict:
    """Fixed-capacity graph tensors for PyG/SB3.

    Layout: UAV nodes at indices 0..max_fleet_size-1,
            user nodes at indices max_fleet_size..max_fleet_size+MAX_OBS_USERS-1,
            padding zeros beyond.
    """
    max_fleet = cfg.max_fleet_size
    max_nodes = max_fleet + MAX_OBS_USERS
    max_edges = 4000

    n_uavs = uavs.shape[0]
    n_users = users.shape[0]
    if n_users > MAX_OBS_USERS:
        raise ValueError(
            f"n_users ({n_users}) exceeds MAX_OBS_USERS ({MAX_OBS_USERS}). "
            f"Set MAX_OBS_USERS higher or reduce n_users."
        )

    # node features: [x/area, y/area, z/h_max, is_uav]
    node_feats = np.zeros((max_nodes, 4), dtype=np.float32)
    for i in range(n_uavs):
        node_feats[i] = [uavs[i, 0] / cfg.area, uavs[i, 1] / cfg.area,
                         uavs[i, 2] / cfg.h_max, 1.0]
    for j in range(n_users):
        node_feats[max_fleet + j] = [users[j, 0] / cfg.area, users[j, 1] / cfg.area, 0.0, 0.0]

    # node_type: 0=padding, 1=UAV, 2=user
    node_type = np.zeros(max_nodes, dtype=np.float32)
    node_type[:n_uavs] = 1.0
    node_type[max_fleet : max_fleet + n_users] = 2.0

    # uav_active: 1=active, 0=failed or padding
    uav_active = np.zeros(max_fleet, dtype=np.float32)
    for i in range(n_uavs):
        uav_active[i] = float(failed_mask[i])

    # edges: UAV-UAV fully connected + UAV-User within graph_edge_radius
    src, dst = [], []
    for i in range(n_uavs):
        for j in range(n_uavs):
            if i != j:
                src.append(i); dst.append(j)
    radius = cfg.graph_edge_radius
    for i in range(n_uavs):
        dists = np.linalg.norm(users - uavs[i, :2], axis=1)
        for j in np.where(dists <= radius)[0]:
            u_node = max_fleet + j
            src += [i, u_node]; dst += [u_node, i]

    n_edges = min(len(src), max_edges)
    edge_index = np.zeros((2, max_edges), dtype=np.float32)
    if n_edges > 0:
        edge_index[0, :n_edges] = np.array(src[:n_edges], dtype=np.float32)
        edge_index[1, :n_edges] = np.array(dst[:n_edges], dtype=np.float32)

    return {
        "node_feats": node_feats,
        "edge_index": edge_index,
        "node_type": node_type,
        "uav_active": uav_active,
        "n_nodes": np.array([float(n_uavs + n_users)], dtype=np.float32),
        "n_edges": np.array([float(n_edges)], dtype=np.float32),
    }


# ── observation space factory ─────────────────────────────────────────────────

def make_obs_space(cfg) -> spaces.Space:
    U, K = cfg.n_uavs, cfg.n_users
    if cfg.obs_mode == "flat":
        return spaces.Box(0.0, 1.0, shape=(U * 3 + K * 2,), dtype=np.float32)
    if cfg.obs_mode == "sorted":
        return spaces.Box(0.0, 1.0, shape=(U * 3 + K * 2,), dtype=np.float32)
    if cfg.obs_mode == "cluster_summary":
        return spaces.Box(0.0, 1.0, shape=(U * 3 + cfg.n_clusters * 3,), dtype=np.float32)
    if cfg.obs_mode == "spatial_grid":
        return spaces.Dict({
            "grid": spaces.Box(0.0, 1.0, shape=(1, 32, 32), dtype=np.float32),
            "uavs": spaces.Box(-1.0, 1.0, shape=(U * 3,), dtype=np.float32),
        })
    if cfg.obs_mode == "graph":
        mf = cfg.max_fleet_size
        mn = mf + MAX_OBS_USERS
        return spaces.Dict({
            "node_feats": spaces.Box(-1.0, 1.0,          shape=(mn, 4),    dtype=np.float32),
            "edge_index": spaces.Box(0.0,  float(mn),    shape=(2, 4000),  dtype=np.float32),
            "node_type":  spaces.Box(0.0,  2.0,          shape=(mn,),      dtype=np.float32),
            "uav_active": spaces.Box(0.0,  1.0,          shape=(mf,),      dtype=np.float32),
            "n_nodes":    spaces.Box(0.0,  float(mn),    shape=(1,),       dtype=np.float32),
            "n_edges":    spaces.Box(0.0,  4000.0,       shape=(1,),       dtype=np.float32),
        })
    raise ValueError(f"Unknown obs_mode: {cfg.obs_mode!r}")
