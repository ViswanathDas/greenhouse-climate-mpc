"""
Unit tests for greenhouse model and EKF.
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model.greenhouse import GreenhouseParams, greenhouse_ode, linearise
from src.estimation.ekf import default_ekf


class TestGreenhouseModel:
    def setup_method(self):
        self.params = GreenhouseParams()
        self.x0 = np.array([22.0, 18.0, 9.0])
        self.u0 = np.array([50.0, 0.1])
        self.d0 = np.array([10.0, 200.0, 7.0])

    def test_ode_returns_shape(self):
        dx = greenhouse_ode(0, self.x0, self.u0, self.d0, self.params)
        assert dx.shape == (3,)

    def test_zero_heating_cools_house(self):
        """With outdoor temp < indoor and no heating, house should cool."""
        d_cold = np.array([5.0, 0.0, 5.0])   # cold dark night
        u_off  = np.array([0.0, 0.05])
        dx = greenhouse_ode(0, self.x0, u_off, d_cold, self.params)
        assert dx[0] < 0, "House should cool on a cold dark night with no heating"

    def test_heating_warms_house(self):
        """High heating input should produce positive dT_air."""
        x_cold = np.array([5.0, 5.0, 6.0])
        u_heat = np.array([200.0, 0.0])
        d_cold = np.array([0.0, 0.0, 4.0])
        dx = greenhouse_ode(0, x_cold, u_heat, d_cold, self.params)
        assert dx[0] > 0

    def test_linearise_shapes(self):
        A, B, Bd = linearise(self.x0, self.u0, self.d0, self.params)
        assert A.shape  == (3, 3)
        assert B.shape  == (3, 2)
        assert Bd.shape == (3, 3)

    def test_linearise_A_stability(self):
        """Eigenvalues of A should have negative real parts (stable open-loop for typical conditions)."""
        A, _, _ = linearise(self.x0, self.u0, self.d0, self.params)
        eigvals = np.linalg.eigvals(A)
        assert np.all(eigvals.real < 0), f"Unstable A matrix at operating point: {eigvals}"


class TestEKF:
    def setup_method(self):
        self.params = GreenhouseParams()
        self.x0 = np.array([22.0, 18.0, 9.0])
        self.ekf = default_ekf(x0=self.x0, dt=150.0, params=self.params)

    def test_initial_state(self):
        np.testing.assert_array_equal(self.ekf.state, self.x0)

    def test_predict_changes_state(self):
        u = np.array([50.0, 0.1])
        d = np.array([10.0, 200.0, 7.0])
        self.ekf.predict(u=u, d=d)
        assert not np.allclose(self.ekf.state, self.x0), "State should change after predict"

    def test_update_returns_innovation(self):
        u = np.array([50.0, 0.1])
        d = np.array([10.0, 200.0, 7.0])
        self.ekf.predict(u=u, d=d)
        y = np.array([22.1, 9.2])   # noisy measurement
        innov = self.ekf.update(y=y)
        assert innov.shape == (2,)

    def test_covariance_positive_definite(self):
        u = np.array([50.0, 0.1])
        d = np.array([10.0, 200.0, 7.0])
        for _ in range(5):
            self.ekf.predict(u=u, d=d)
            self.ekf.update(y=np.array([22.0, 9.0]))
        P = self.ekf.covariance
        eigvals = np.linalg.eigvals(P)
        assert np.all(eigvals > 0), "Covariance matrix should remain positive definite"
