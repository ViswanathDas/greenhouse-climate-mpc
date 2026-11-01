"""EKF behaviour under sensor dropout (NaN measurements)."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from src.model.greenhouse import GreenhouseParams, greenhouse_ode
from src.estimation.ekf import default_ekf

DT = 150.0
U = np.array([80.0, 0.2])
D = np.array([8.0, 0.0, 6.0])   # night: 8 °C outside, no sun, 6 g/kg


def run(n_steps, dropout, seed=0):
    p = GreenhouseParams()
    rng = np.random.default_rng(seed)
    x = np.array([18.0, 14.0, 9.0])
    ekf = default_ekf(x + np.array([1.0, 1.0, 0.5]), DT, p)
    errs, traces = [], []
    for k in range(n_steps):
        x = solve_ivp(greenhouse_ode, [0, DT], x, args=(U, D, p)).y[:, -1]
        ekf.predict(U, D)
        y = x[[0, 2]] + rng.normal(0, [0.3, 0.3])
        if dropout(k):
            y = np.full(2, np.nan)
        ekf.update(y)
        errs.append(np.abs(ekf.state - x))
        traces.append(np.trace(ekf.covariance))
    return np.array(errs), np.array(traces)


def test_all_nan_skips_update():
    p = GreenhouseParams()
    ekf = default_ekf(np.array([18.0, 14.0, 9.0]), DT, p)
    x_before, P_before = ekf.state, ekf.covariance
    innov = ekf.update(np.array([np.nan, np.nan]))
    assert np.all(np.isnan(innov))
    np.testing.assert_array_equal(ekf.state, x_before)
    np.testing.assert_array_equal(ekf.covariance, P_before)


def test_partial_measurement_uses_available_channel():
    p = GreenhouseParams()
    ekf = default_ekf(np.array([18.0, 14.0, 9.0]), DT, p)
    innov = ekf.update(np.array([19.0, np.nan]))
    assert not np.isnan(innov[0]) and np.isnan(innov[1])
    assert ekf.state[0] > 18.0


def test_covariance_grows_during_dropout_and_recovers():
    # Let the filter settle first, then blind it for 10 steps.
    _, tr = run(70, dropout=lambda k: 40 <= k < 50)
    assert tr[49] > tr[39]         # uncertainty grows while blind
    assert tr[69] < tr[49]         # and shrinks once data return


def test_estimate_stays_bounded_through_dropout():
    errs, _ = run(60, dropout=lambda k: 20 <= k < 30)
    assert errs[-10:, 0].max() < 1.0   # T_air error back under 1 °C
    assert np.all(np.isfinite(errs))


def test_covariance_stays_symmetric_positive_definite():
    p = GreenhouseParams()
    ekf = default_ekf(np.array([18.0, 14.0, 9.0]), DT, p)
    for _ in range(20):
        ekf.predict(U, D)
        ekf.update(np.array([18.2, 9.1]))
    P = ekf.covariance
    np.testing.assert_allclose(P, P.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(P) > 0)
