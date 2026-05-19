"""
Greenhouse Climate MPC Controller
==================================
Receding-horizon Model Predictive Control for temperature and humidity setpoint tracking.

Formulation (linearised MPC around operating point):
    min   Σ_{k=0}^{N-1} [ (x_k - x_ref)' Q (x_k - x_ref) + u_k' R u_k + Δu_k' S Δu_k ]
          + (x_N - x_ref)' P_f (x_N - x_ref)
    s.t.  x_{k+1} = Φ x_k + Γ u_k + Γ_d d_k
          u_min ≤ u_k ≤ u_max
          Δu_min ≤ u_k - u_{k-1} ≤ Δu_max

States tracked:   T_air (°C), H (g/kg)  [T_cov not controlled, only estimated]
Inputs:           u_heat (W/m²), u_vent (fraction)
"""

import numpy as np
import cvxpy as cp
from dataclasses import dataclass
from typing import Optional

from ..model.greenhouse import linearise, GreenhouseParams


@dataclass
class MPCParams:
    N: int = 12                        # prediction horizon (steps)
    dt: float = 150.0                  # step size [s] (2.5 min)

    # Tracking weights [T_air, T_cov, H]
    Q_diag: tuple = (10.0, 0.1, 2.0)
    # Terminal cost (larger = more stable)
    Pf_diag: tuple = (20.0, 0.2, 4.0)
    # Input cost [u_heat, u_vent]
    R_diag: tuple = (0.01, 5.0)
    # Rate-of-change penalty
    S_diag: tuple = (0.05, 10.0)

    # Input constraints
    u_min: tuple = (0.0, 0.0)         # [W/m², fraction]
    u_max: tuple = (200.0, 1.0)
    du_min: tuple = (-20.0, -0.1)
    du_max: tuple = (20.0, 0.1)


class GreenhouseMPC:
    """
    Linearised MPC controller for greenhouse climate.

    The controller linearises the plant model at each call (successive linearisation)
    and solves a QP over the prediction horizon using CVXPY.
    """

    def __init__(self, params: GreenhouseParams, mpc_params: MPCParams):
        self.plant_params = params
        self.p = mpc_params
        self._u_prev = np.array([50.0, 0.1])   # warm-start previous input

    def compute(
        self,
        x0: np.ndarray,
        x_ref: np.ndarray,
        d_seq: np.ndarray,
        u_prev: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Solve the MPC optimisation problem and return the first optimal input.

        Parameters
        ----------
        x0 : array [3,]
            Current (estimated) state [T_air, T_cov, H]
        x_ref : array [3,]
            Setpoint [T_air_ref, T_cov_ref (unused), H_ref]
        d_seq : array [N, 3]
            Disturbance sequence over prediction horizon [T_out, I_sol, H_out]
        u_prev : array [2,] or None
            Previous input (for rate-of-change constraint)

        Returns
        -------
        dict with keys:
            'u_opt'    : array [2,]  — first optimal input
            'x_pred'   : array [N+1, 3]  — predicted state trajectory
            'u_seq'    : array [N, 2]    — full input sequence
            'cost'     : float           — optimal cost value
            'status'   : str             — solver status
        """
        if u_prev is None:
            u_prev = self._u_prev

        p = self.p
        N = p.N
        n = 3   # state dimension
        m = 2   # input dimension

        # Linearise at current operating point using mean disturbance
        d_op = d_seq[0]
        A, B, Bd = linearise(x0, u_prev, d_op, self.plant_params)

        # Discretise (Euler; step is small enough for stability)
        dt = p.dt
        Phi = np.eye(n) + A * dt
        Gamma = B * dt
        Gamma_d = Bd * dt

        # Weight matrices
        Q  = np.diag(p.Q_diag)
        Pf = np.diag(p.Pf_diag)
        R  = np.diag(p.R_diag)
        S  = np.diag(p.S_diag)

        u_min = np.array(p.u_min)
        u_max = np.array(p.u_max)
        du_min = np.array(p.du_min)
        du_max = np.array(p.du_max)

        # CVXPY variables
        x = cp.Variable((N + 1, n))
        u = cp.Variable((N, m))

        cost = 0.0
        constraints = [x[0] == x0]

        for k in range(N):
            dx = x[k] - x_ref
            du = u[k] - (u[k - 1] if k > 0 else u_prev)
            cost += cp.quad_form(dx, Q) + cp.quad_form(u[k], R) + cp.quad_form(du, S)

            d_k = d_seq[min(k, len(d_seq) - 1)]
            constraints += [
                x[k + 1] == Phi @ x[k] + Gamma @ u[k] + Gamma_d @ d_k,
                u[k] >= u_min,
                u[k] <= u_max,
            ]
            if k > 0:
                constraints += [
                    u[k] - u[k - 1] >= du_min,
                    u[k] - u[k - 1] <= du_max,
                ]
            else:
                constraints += [
                    u[0] - u_prev >= du_min,
                    u[0] - u_prev <= du_max,
                ]

        # Terminal cost
        cost += cp.quad_form(x[N] - x_ref, Pf)

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP, warm_start=True, eps_abs=1e-4, eps_rel=1e-4)

        if prob.status in ["optimal", "optimal_inaccurate"]:
            u_opt = u.value[0]
            self._u_prev = u_opt
            return {
                "u_opt": u_opt,
                "x_pred": x.value,
                "u_seq": u.value,
                "cost": float(prob.value),
                "status": prob.status,
            }
        else:
            # Fallback: return previous input (hold)
            return {
                "u_opt": u_prev.copy(),
                "x_pred": None,
                "u_seq": None,
                "cost": None,
                "status": prob.status,
            }
