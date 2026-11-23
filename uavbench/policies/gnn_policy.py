"""GATv2-based feature extractor for graph obs_mode.

Architecture:
  GATv2Conv(4 → 64, heads=4, concat=True)   → (n_actual_nodes, 256)
  GATv2Conv(256 → 64, heads=1, concat=False) → (n_actual_nodes, 64)
  Extract UAV nodes; apply uav_active mask → zero out failed/padding slots
  Flatten: (max_fleet * 64,) per sample

Each sample processed separately (PyG variable-size graphs vs SB3 fixed batches).
Acceptable overhead for research use at batch_size ≤ 256.

Requires: torch_geometric>=2.5  (pip install torch_geometric)
"""
from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.nn import GATv2Conv


class GNNExtractor(BaseFeaturesExtractor):
    UAV_EMB_DIM = 64

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int | None = None):
        self.max_fleet = observation_space["uav_active"].shape[0]
        inferred = self.max_fleet * self.UAV_EMB_DIM
        if features_dim is None:
            features_dim = inferred
        if features_dim != inferred:
            raise ValueError(f"features_dim must equal max_fleet*64={inferred}, got {features_dim}")
        super().__init__(observation_space, features_dim)
        self.conv1 = GATv2Conv(in_channels=4,   out_channels=64, heads=4, concat=True)   # → 256
        self.conv2 = GATv2Conv(in_channels=256, out_channels=64, heads=1, concat=False)  # → 64

    def forward(self, observations: dict) -> torch.Tensor:
        batch_size = observations["node_feats"].shape[0]
        device = observations["node_feats"].device
        mf, ed = self.max_fleet, self.UAV_EMB_DIM
        results = []
        for b in range(batch_size):
            # Node layout: UAVs at 0..mf-1, users at mf..mf+n_users-1 (padding zeros beyond).
            # n_nodes = n_uavs + n_users (actual counts), but user indices start at mf,
            # so the node span required by PyG is mf + n_users.
            # Derive n_uavs from node_type (value=1 → UAV node).
            n_total = int(observations["n_nodes"][b, 0].item())  # n_uavs + n_users
            ne = int(observations["n_edges"][b, 0].item())
            node_type_b = observations["node_type"][b]           # shape (mf+MAX_OBS_USERS,)
            n_uavs = int((node_type_b == 1.0).sum().item())
            n_users = n_total - n_uavs
            n_span = mf + n_users  # covers all actual node indices referenced in edge_index

            x = observations["node_feats"][b, :n_span].float()
            if ne > 0:
                ei = observations["edge_index"][b, :, :ne].long()
            else:
                ei = torch.zeros(2, 0, dtype=torch.long, device=device)
            x = F.elu(self.conv1(x, ei))    # (n_span, 256)
            x = self.conv2(x, ei)           # (n_span, 64)
            # UAV embeddings at indices 0..mf-1; zero out failed/padding slots via uav_active
            uav_emb = x[:mf].clone()
            uav_active = observations["uav_active"][b].float()
            uav_emb = uav_emb * uav_active.unsqueeze(-1)
            results.append(uav_emb.flatten())
        return torch.stack(results, dim=0)   # (batch, max_fleet*64)
