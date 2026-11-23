"""CNN feature extractor for spatial_grid obs_mode.

Input Dict: {"grid": (1, 32, 32), "uavs": (3*n_uavs,)}
Output: (batch, features_dim)  default features_dim=128

Compatible with SB3 MultiInputPolicy via policy_kwargs:
    {"features_extractor_class": CnnExtractor,
     "features_extractor_kwargs": {"features_dim": 128}}
"""
from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CnnExtractor(BaseFeaturesExtractor):
    """Fuses a 32×32 user-occupancy grid with a normalised UAV state vector.

    Architecture:
        CNN branch : Conv2d(1→16, 3×3) → ReLU → Conv2d(16→32, 3×3) → ReLU
                     → AdaptiveAvgPool2d(4,4) → Flatten  →  512-d
        UAV branch : passthrough float                             →  3*n_uavs-d
        FC head    : Linear(512 + 3*n_uavs, features_dim) → ReLU  → features_dim-d
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        uav_dim = observation_space["uavs"].shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),           # → 32×4×4 = 512
        )
        self.fc = nn.Linear(512 + uav_dim, features_dim)

    def forward(self, observations: dict) -> torch.Tensor:
        grid_feat = self.cnn(observations["grid"])          # (batch, 512)
        uav_feat = observations["uavs"].float()             # (batch, 3*n_uavs)
        return torch.relu(self.fc(torch.cat([grid_feat, uav_feat], dim=-1)))
