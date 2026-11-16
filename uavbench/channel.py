"""Al-Hourani air-to-ground (A2G) channel model.

References
----------
[1] A. Al-Hourani, S. Kandeepan, S. Lardner, "Optimal LAP Altitude for Maximum
    Coverage," IEEE Wireless Communications Letters, 3(6):569-572, 2014.
    DOI: 10.1109/LWC.2014.2342736
[2] A. Al-Hourani, S. Kandeepan, A. Jamalipour, "Modeling air-to-ground path
    loss for low altitude platforms in urban environments," IEEE GLOBECOM 2014.

Model summary (all angles in DEGREES, as in [1]):
    P_LoS(theta) = 1 / (1 + a * exp(-b * (theta - a)))
    FSPL(d)      = 20 log10(d) + 20 log10(f) + 20 log10(4*pi/c)
    PL_mean      = FSPL + P_LoS * eta_LoS + (1 - P_LoS) * eta_NLoS
A ground user is "covered" by a UAV if PL_mean <= PL_max.

TODO(student): verify every constant below against the original PDFs before
you report results. Do not trust this file (or any AI summary) blindly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

SPEED_OF_LIGHT = 299_792_458.0  # m/s


@dataclass(frozen=True)
class Environment:
    """S-curve parameters (a, b) and mean excess losses (dB) for one environment."""

    name: str
    a: float
    b: float
    eta_los: float
    eta_nlos: float


# Commonly cited values from [1], [2] (2 GHz). Verify against the papers.
ENVIRONMENTS = {
    "suburban": Environment("suburban", a=4.88, b=0.43, eta_los=0.1, eta_nlos=21.0),
    "urban": Environment("urban", a=9.61, b=0.16, eta_los=1.0, eta_nlos=20.0),
    "dense_urban": Environment("dense_urban", a=12.08, b=0.11, eta_los=1.6, eta_nlos=23.0),
    "highrise_urban": Environment("highrise_urban", a=27.23, b=0.08, eta_los=2.3, eta_nlos=34.0),
}


class AlHouraniChannel:
    """Vectorized mean path loss between UAVs and ground users."""

    def __init__(self, env: str | Environment = "urban", freq_hz: float = 2.0e9):
        self.env = ENVIRONMENTS[env] if isinstance(env, str) else env
        self.freq_hz = float(freq_hz)
        # Constant part of free-space path loss: 20log10(f) + 20log10(4*pi/c)
        self._fspl_const = 20.0 * np.log10(self.freq_hz) + 20.0 * np.log10(
            4.0 * np.pi / SPEED_OF_LIGHT
        )

    # ------------------------------------------------------------------ basics
    def p_los(self, theta_deg: np.ndarray) -> np.ndarray:
        """Probability of geometrical line of sight; theta in degrees."""
        a, b = self.env.a, self.env.b
        return 1.0 / (1.0 + a * np.exp(-b * (np.asarray(theta_deg) - a)))

    def fspl(self, d: np.ndarray) -> np.ndarray:
        """Free-space path loss in dB for 3-D distance d (meters)."""
        return 20.0 * np.log10(np.maximum(np.asarray(d, dtype=float), 1e-3)) + self._fspl_const

    def mean_path_loss(self, h: np.ndarray, r: np.ndarray) -> np.ndarray:
        """Mean A2G path loss (dB) for altitude h and horizontal distance r (meters)."""
        h = np.asarray(h, dtype=float)
        r = np.asarray(r, dtype=float)
        d = np.sqrt(h**2 + r**2)
        theta = np.degrees(np.arctan2(h, r))
        p = self.p_los(theta)
        excess = p * self.env.eta_los + (1.0 - p) * self.env.eta_nlos
        return self.fspl(d) + excess

    def path_loss_matrix(self, uav_xyz: np.ndarray, user_xy: np.ndarray) -> np.ndarray:
        """Path loss (dB) matrix of shape (n_uavs, n_users).

        uav_xyz: (U, 3) array of x, y, altitude; user_xy: (K, 2) array.
        """
        diff = uav_xyz[:, None, :2] - user_xy[None, :, :]
        r = np.linalg.norm(diff, axis=-1)
        h = np.broadcast_to(uav_xyz[:, 2:3], r.shape)
        return self.mean_path_loss(h, r)

    def sinr_matrix(
        self,
        uav_xyz: np.ndarray,
        user_xy: np.ndarray,
        tx_power_dbm: float = 30.0,
        noise_dbm: float = -100.0,
    ) -> np.ndarray:
        """Per-user SINR (dB), treating all UAVs as co-channel interferers.

        Signal power  = strongest UAV's received power.
        Interference  = sum of all other UAVs' received power.
        Returns array of shape (n_users,).
        """
        pl = self.path_loss_matrix(uav_xyz, user_xy)       # (U, K) dB
        rx_dbm = tx_power_dbm - pl                          # (U, K) dBm
        rx_lin = 10.0 ** (rx_dbm / 10.0)                  # (U, K) mW linear
        noise_lin = 10.0 ** (noise_dbm / 10.0)            # scalar mW
        signal = rx_lin.max(axis=0)                         # (K,) best-UAV power
        interference = rx_lin.sum(axis=0) - signal          # (K,) all-others power
        sinr_lin = signal / np.maximum(interference + noise_lin, 1e-20)
        return 10.0 * np.log10(sinr_lin)                   # (K,) dB

    # --------------------------------------------------------- optimal altitude
    def optimal_elevation_deg(self) -> float:
        """Solve Eq. (13) of [1] for the optimal elevation angle (degrees).

        pi/(9 ln 10) * tan(theta) + a*b*A*exp(-b(theta-a)) / (a*exp(-b(theta-a)) + 1)^2 = 0
        with A = eta_LoS - eta_NLoS (< 0). The root does not depend on PL_max.
        """
        a, b = self.env.a, self.env.b
        A = self.env.eta_los - self.env.eta_nlos

        def f(theta):
            e = np.exp(-b * (theta - a))
            return (np.pi / (9.0 * np.log(10.0))) * np.tan(np.radians(theta)) + (
                a * b * A * e
            ) / (a * e + 1.0) ** 2

        return float(brentq(f, 0.1, 89.9))

    def coverage_radius(self, h: float, pl_max: float) -> float:
        """Largest horizontal radius r with mean_path_loss(h, r) <= pl_max (0 if none)."""
        if self.mean_path_loss(h, 0.0) > pl_max:
            return 0.0
        hi = 1.0
        while self.mean_path_loss(h, hi) <= pl_max:
            hi *= 2.0
            if hi > 1e7:
                return float("inf")
        # Path loss is not guaranteed monotone in r for every environment, but it is
        # for the standard parameter sets; bisection finds the outer boundary.
        return float(brentq(lambda r: self.mean_path_loss(h, r) - pl_max, 0.0, hi))

    def optimal_altitude(self, pl_max: float) -> tuple[float, float]:
        """Return (h_opt, R_max) for a single LAP under budget pl_max, per [1]."""
        theta = np.radians(self.optimal_elevation_deg())
        # On the optimal ray, PL = Lambda(R, theta); solve for R.
        def g(R):
            return self.mean_path_loss(R * np.tan(theta), R) - pl_max

        R = brentq(g, 1e-3, 1e7)
        return float(R * np.tan(theta)), float(R)
