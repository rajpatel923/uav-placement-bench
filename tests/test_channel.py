import numpy as np
import pytest

from uavbench.channel import AlHouraniChannel, ENVIRONMENTS

# Optimal elevation angles (degrees) for the four standard environments.
# These match the values commonly reported from Al-Hourani et al. (2014).
# TODO(student): confirm against the table/figure in the original paper.
EXPECTED_THETA_OPT = {
    "suburban": 20.34,
    "urban": 42.44,
    "dense_urban": 54.62,
    "highrise_urban": 75.52,
}


@pytest.mark.parametrize("env", list(ENVIRONMENTS))
def test_optimal_elevation_matches_published(env):
    theta = AlHouraniChannel(env).optimal_elevation_deg()
    assert theta == pytest.approx(EXPECTED_THETA_OPT[env], abs=0.05)


@pytest.mark.parametrize("env", list(ENVIRONMENTS))
def test_theta_opt_independent_of_pl_max(env):
    """Al-Hourani: the optimal angle does not depend on the path-loss budget."""
    ch = AlHouraniChannel(env)
    theta = ch.optimal_elevation_deg()
    for pl_max in (95.0, 105.0, 115.0):
        h, R = ch.optimal_altitude(pl_max)
        assert np.degrees(np.arctan2(h, R)) == pytest.approx(theta, abs=1e-6)


def test_p_los_monotone_and_bounded():
    ch = AlHouraniChannel("urban")
    theta = np.linspace(0, 90, 181)
    p = ch.p_los(theta)
    assert np.all((p >= 0) & (p <= 1))
    assert np.all(np.diff(p) >= 0)


def test_fspl_known_value():
    # FSPL at 1 km, 2 GHz is about 98.45 dB.
    ch = AlHouraniChannel("urban", freq_hz=2e9)
    assert ch.fspl(1000.0) == pytest.approx(98.45, abs=0.05)


def test_optimal_altitude_maximizes_radius():
    """R(h) at h_opt should beat nearby altitudes (numerical sanity check)."""
    ch = AlHouraniChannel("urban")
    h_opt, R_opt = ch.optimal_altitude(100.0)
    for h in (0.8 * h_opt, 1.2 * h_opt):
        assert ch.coverage_radius(h, 100.0) <= R_opt + 1e-6


def test_path_loss_matrix_shape():
    ch = AlHouraniChannel("urban")
    uavs = np.array([[0, 0, 100], [500, 500, 150]], dtype=float)
    users = np.random.default_rng(0).uniform(0, 1000, size=(7, 2))
    assert ch.path_loss_matrix(uavs, users).shape == (2, 7)


def test_sinr_matrix_shape():
    ch = AlHouraniChannel("urban")
    uav_xyz = np.array([[500.0, 500.0, 100.0], [2500.0, 2500.0, 100.0]])
    user_xy = np.random.default_rng(0).uniform(0, 3000, (10, 2))
    sinr = ch.sinr_matrix(uav_xyz, user_xy)
    assert sinr.shape == (10,), f"Expected (10,), got {sinr.shape}"


def test_sinr_single_uav_equals_snr():
    """With one UAV there is no interference; SINR should equal SNR."""
    ch = AlHouraniChannel("urban")
    uav_xyz = np.array([[1500.0, 1500.0, 100.0]])
    user_xy = np.array([[1500.0, 1600.0]])
    tx_power_dbm, noise_dbm = 30.0, -100.0
    sinr = ch.sinr_matrix(uav_xyz, user_xy, tx_power_dbm, noise_dbm)
    pl = ch.path_loss_matrix(uav_xyz, user_xy)[0, 0]
    snr = tx_power_dbm - pl - noise_dbm
    assert abs(sinr[0] - snr) < 0.01, f"Expected SNR={snr:.2f} dB, got SINR={sinr[0]:.2f} dB"


def test_sinr_finite_values():
    ch = AlHouraniChannel("urban")
    rng = np.random.default_rng(1)
    uav_xyz = rng.uniform(0, 3000, (4, 3))
    uav_xyz[:, 2] = 100.0
    user_xy = rng.uniform(0, 3000, (20, 2))
    sinr = ch.sinr_matrix(uav_xyz, user_xy)
    assert np.all(np.isfinite(sinr)), "SINR contains non-finite values"


def test_sinr_less_than_snr_with_multiple_uavs():
    """Adding a second UAV as interferer should reduce SINR below SNR."""
    ch = AlHouraniChannel("urban")
    user_xy = np.array([[1500.0, 1500.0]])
    # UAV1 close to user; UAV2 also close — creates interference
    uav1 = np.array([[1500.0, 1600.0, 100.0]])
    uav2 = np.array([[1500.0, 1700.0, 100.0], [1500.0, 1600.0, 100.0]])
    sinr_no_interference = ch.sinr_matrix(uav1, user_xy)[0]
    sinr_with_interference = ch.sinr_matrix(uav2, user_xy)[0]
    assert sinr_with_interference < sinr_no_interference, (
        "SINR should decrease when a second close interfering UAV is added"
    )
