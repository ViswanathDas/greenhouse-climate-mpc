"""
Extended Kalman Filter — Greenhouse State Estimation
=====================================================
Estimates the full state vector [T_air, T_cov, H] from noisy measurements
of air temperature and humidity only (T_cov is typically not directly measured).

Measurement model:
    y = [T_air, H] + noise  (2 sensors, 3 states — T_cov unobserved)
"""

import numpy as np
from ..model.greenhouse import greenhouse_ode, linearise, GreenhouseParams


class GreenhouseEKF:
    """
    Discrete-time Extended Kalman Filter for greenhouse state estimation.

    State: x = [T_air, T_cov, H]
    Measurement: y = [T_air, H]  (C = [[1,0,0],[0,0,1]])
    """

    C = np.array([[1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0]])  # measurement matrix

    def __init__(
        self,
        x0: np.ndarray,
        P0: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        dt: float,
        params: GreenhouseParams,
    ):
        """
        Parameters
        ----------
        x0 : array [3,]  Initial state estimate
        P0 : array [3,3]  Initial state covariance
        Q  : array [3,3]  Process noise covariance
        R  : array [2,2]  Measurement noise covariance
        dt : float        Sampling interval [s]
        params : GreenhouseParams
        """
        self.x = x0.copy()
        self.P = P0.copy()
        self.Q = Q.copy()
        self.R = R.copy()
        self.dt = dt
        self.params = params

        self.n = 3
        self.m = 2

    def predict(self, u: np.ndarray, d: np.ndarray):
        """
        Prediction step: propagate state and covariance forward by dt.

        Parameters
        ----------
        u : array [2,]  Control inputs applied during this step
        d : array [3,]  Disturbances (measured feedforward)
        """
        p = self.params
        dt = self.dt

        # Linearise around current estimate
        A, B, Bd = linearise(self.x, u, d, p)

        # Discretise A (first-order Euler; sufficient for short dt)
        Phi = np.eye(self.n) + A * dt

        # Propagate state (nonlinear prediction for accuracy)
        from scipy.integrate import solve_ivp
        sol = solve_ivp(
            greenhouse_ode,
            [0, dt],
            self.x,
            args=(u, d, p),
            method='RK45',
            dense_output=False,
        )
        self.x = sol.y[:, -1]

        # Propagate covariance
        self.P = Phi @ self.P @ Phi.T + self.Q

    def update(self, y: np.ndarray):
        """
        Measurement update step.

        Parameters
        ----------
        y : array [2,]  Measured [T_air, H]
        """
        C = self.C
        S = C @ self.P @ C.T + self.R          # Innovation covariance
        K = self.P @ C.T @ np.linalg.inv(S)    # Kalman gain

        y_hat = C @ self.x                     # Predicted measurement
        innovation = y - y_hat

        self.x = self.x + K @ innovation
        I_KC = np.eye(self.n) - K @ C
        self.P = I_KC @ self.P                 # Joseph form: I_KC @ P @ I_KC.T + K @ R @ K.T (numerically stable)

        return innovation

    @property
    def state(self) -> np.ndarray:
        return self.x.copy()

    @property
    def covariance(self) -> np.ndarray:
        return self.P.copy()


def default_ekf(x0: np.ndarray, dt: float, params: GreenhouseParams) -> GreenhouseEKF:
    """
    Construct EKF with sensible default noise parameters.

    Sensor noise:
        Temperature sensor: ±0.3°C (1-sigma)
        Humidity sensor:    ±1.5 g/kg (1-sigma)
    Process noise: small, reflecting model uncertainty
    """
    P0 = np.diag([1.0, 2.0, 1.0])       # initial uncertainty
    Q  = np.diag([0.001, 0.005, 0.01])  # process noise (model error)
    R  = np.diag([0.09, 2.25])          # 0.3² for T, 1.5² for H

    return GreenhouseEKF(x0=x0, P0=P0, Q=Q, R=R, dt=dt, params=params)
