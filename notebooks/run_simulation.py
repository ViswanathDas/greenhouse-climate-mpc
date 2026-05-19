"""
Greenhouse Climate MPC — Full Closed-Loop Simulation
======================================================
Runs a 24-hour simulation of the greenhouse under MPC control with EKF state estimation.

Usage:
    python notebooks/run_simulation.py

Output:
    - Console: per-step RMS tracking errors
    - Figures: simulation dashboard, EKF diagnostics
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.model.greenhouse import GreenhouseParams, greenhouse_ode
from src.estimation.ekf import default_ekf
from src.controller.mpc import GreenhouseMPC, MPCParams
from src.utils.weather import get_disturbance_sequence
from src.utils.plotting import plot_simulation, plot_ekf_diagnostics
from scipy.integrate import solve_ivp


# ─── Simulation settings ───────────────────────────────────────────────────────
DT        = 150.0     # control step [s] (2.5 min)
N_HOURS   = 24        # simulation duration
N_STEPS   = int(N_HOURS * 3600 / DT)
SEASON    = "spring"

# Setpoint: 22°C air temperature, 9 g/kg humidity
X_REF = np.array([22.0, 18.0, 9.0])   # [T_air, T_cov (unused), H]

# Initial condition: cold house at 10°C, 6 g/kg
x0_true = np.array([10.0, 9.0, 6.0])

# Sensor noise standard deviations
SIGMA_T = 0.3    # °C
SIGMA_H = 1.5    # g/kg


def main():
    rng = np.random.default_rng(seed=0)

    params = GreenhouseParams()
    mpc_params = MPCParams(N=12, dt=DT)

    # Disturbance sequence (weather over full simulation + prediction horizon)
    d_all = get_disturbance_sequence(
        n_steps=N_STEPS + mpc_params.N + 10,
        dt_seconds=DT,
        season=SEASON,
    )

    # EKF initial estimate: slightly off from truth
    x0_est = x0_true + np.array([1.0, 2.0, 0.5])
    ekf = default_ekf(x0=x0_est, dt=DT, params=params)

    # MPC controller
    mpc = GreenhouseMPC(params=params, mpc_params=mpc_params)

    # Storage
    t_arr       = np.zeros(N_STEPS)
    x_true_arr  = np.zeros((N_STEPS, 3))
    x_est_arr   = np.zeros((N_STEPS, 3))
    u_arr       = np.zeros((N_STEPS, 2))
    innov_arr   = np.zeros((N_STEPS, 2))
    P_trace_arr = np.zeros(N_STEPS)

    x_true = x0_true.copy()
    u_prev = np.array([50.0, 0.1])

    print(f"Starting {N_HOURS}h simulation ({N_STEPS} steps, dt={DT}s)")
    print(f"Setpoint: T_air={X_REF[0]}°C, H={X_REF[2]} g/kg")
    print(f"Initial:  T_air={x0_true[0]}°C, H={x0_true[2]} g/kg\n")

    for k in range(N_STEPS):
        t_k = k * DT
        d_k = d_all[k]

        # 1. Noisy measurement
        y_noisy = np.array([
            x_true[0] + rng.normal(0, SIGMA_T),
            x_true[2] + rng.normal(0, SIGMA_H),
        ])

        # 2. EKF predict → update
        ekf.predict(u=u_prev, d=d_k)
        innovation = ekf.update(y=y_noisy)

        x_hat = ekf.state

        # 3. MPC solve (using estimated state + look-ahead disturbances)
        d_horizon = d_all[k: k + mpc_params.N + 1]
        if len(d_horizon) < mpc_params.N:
            d_horizon = np.vstack([
                d_horizon,
                np.tile(d_horizon[-1], (mpc_params.N - len(d_horizon) + 1, 1))
            ])

        result = mpc.compute(x0=x_hat, x_ref=X_REF, d_seq=d_horizon, u_prev=u_prev)
        u_k = result["u_opt"]

        # 4. Apply input to true plant (continuous integration over DT)
        sol = solve_ivp(
            greenhouse_ode,
            [0, DT],
            x_true,
            args=(u_k, d_k, params),
            method='RK45',
        )
        x_true = sol.y[:, -1]

        # 5. Store
        t_arr[k]        = t_k
        x_true_arr[k]   = x_true
        x_est_arr[k]    = x_hat
        u_arr[k]        = u_k
        innov_arr[k]    = innovation
        P_trace_arr[k]  = np.trace(ekf.covariance)

        u_prev = u_k

        if k % 48 == 0:  # print every 2 hours
            print(f"  t={t_k/3600:.1f}h | T_air={x_true[0]:.2f}°C (ref {X_REF[0]}) "
                  f"| H={x_true[2]:.2f} g/kg (ref {X_REF[2]}) "
                  f"| u_heat={u_k[0]:.1f}W/m² u_vent={u_k[1]:.3f}")

    # ─── Performance metrics ───────────────────────────────────────────────────
    T_err = x_true_arr[:, 0] - X_REF[0]
    H_err = x_true_arr[:, 2] - X_REF[2]
    print(f"\nPerformance Summary:")
    print(f"  RMS temperature error: {np.sqrt(np.mean(T_err**2)):.3f} °C")
    print(f"  RMS humidity error:    {np.sqrt(np.mean(H_err**2)):.3f} g/kg")
    print(f"  Max temperature error: {np.abs(T_err).max():.3f} °C")
    print(f"  Max humidity error:    {np.abs(H_err).max():.3f} g/kg")

    # ─── Plots ─────────────────────────────────────────────────────────────────
    plot_simulation(
        t=t_arr,
        x_true=x_true_arr,
        x_est=x_est_arr,
        u_seq=u_arr,
        d_seq=d_all[:N_STEPS],
        x_ref=X_REF,
        title=f"Greenhouse MPC — {N_HOURS}h Simulation ({SEASON.capitalize()})",
    )

    plot_ekf_diagnostics(t=t_arr, innovations=innov_arr, P_trace=P_trace_arr)


if __name__ == "__main__":
    main()
