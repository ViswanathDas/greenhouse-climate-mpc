"""
Greenhouse Thermal + Humidity Dynamics Model
=============================================
First-principles ODE model based on Vanthoor et al. (2011) simplified formulation.

States:
    T_air  : Greenhouse air temperature [°C]
    T_cov  : Greenhouse cover temperature [°C]
    H      : Absolute humidity ratio of indoor air [g/kg dry air]

Inputs (controllable):
    u_heat : Heating pipe power [W/m²]
    u_vent : Ventilation opening fraction [0..1]

Disturbances (measured feedforward):
    T_out  : Outdoor air temperature [°C]
    I_sol  : Solar irradiance [W/m²]
    H_out  : Outdoor humidity [g/kg dry air]
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class GreenhouseParams:
    """Physical parameters for a single-span glasshouse."""
    # Geometry
    floor_area: float = 1000.0        # [m²]
    cover_area: float = 1100.0        # [m²]  (roof + side walls)
    volume: float = 5000.0            # [m³]  (air volume)

    # Thermal capacities
    C_air: float = 1.0e4             # [J/(m² K)]  air (5 m x 1.2 kg/m³ x 1006 J/kgK ≈ 6.0e3) + canopy
    C_cov: float = 8.4e3             # [J/(m² K)]  4 mm glass (2500 kg/m³ x 840 J/kgK x 0.004 m)

    # Heat transfer coefficients
    U_pipe: float = 1.0              # [—]  pipe-to-air efficiency (fraction of u_heat absorbed by air)
    k_cov_air: float = 6.0           # [W/(m² K)]  convective air-cover HTC
    k_cov_out: float = 8.0           # [W/(m² K)]  cover-outdoor HTC (wind dependent)
    tau_sol: float = 0.65            # [—]  solar transmission of glass
    alpha_air: float = 0.5           # [—]  fraction of solar absorbed by air/crop

    # Ventilation
    vent_max: float = 0.02           # [m³/(s·m²)]  max specific ventilation rate
    vent_enthalpy_eff: float = 1.0   # [—]

    # Humidity
    cp_air: float = 1006.0           # [J/(kg K)]
    rho_air: float = 1.2             # [kg/m³]
    lambda_evap: float = 2.45e6      # [J/kg]  latent heat of vaporisation
    ET_base: float = 0.03            # [g/(m² s)]  base crop evapotranspiration rate


def greenhouse_ode(t, x, u, d, params: GreenhouseParams):
    """
    State derivatives for the greenhouse model.

    Parameters
    ----------
    t : float
        Time [s] (not used directly; included for ODE solver compatibility)
    x : array [3,]
        State vector [T_air (°C), T_cov (°C), H (g/kg)]
    u : array [2,]
        Control inputs [u_heat (W/m²), u_vent (0..1)]
    d : array [3,]
        Disturbances [T_out (°C), I_sol (W/m²), H_out (g/kg)]
    params : GreenhouseParams

    Returns
    -------
    dxdt : array [3,]
    """
    T_air, T_cov, H = x
    u_heat, u_vent = np.clip(u, [0.0, 0.0], [300.0, 1.0])
    T_out, I_sol, H_out = d

    p = params
    A = p.floor_area
    A_c = p.cover_area
    V = p.volume

    # --- Temperature dynamics ---

    # Heating pipe contribution to air
    Q_heat = p.U_pipe * u_heat * A                    # [W]

    # Solar irradiance absorbed by air/crop
    Q_sol_air = p.tau_sol * p.alpha_air * I_sol * A  # [W]

    # Convective exchange air <-> cover
    Q_conv_air_cov = p.k_cov_air * A_c * (T_air - T_cov)  # [W] (positive = air loses heat)

    # Ventilation heat exchange
    vent_rate = p.vent_max * u_vent * A               # [m³/s]
    Q_vent_air = p.rho_air * p.cp_air * vent_rate * (T_out - T_air)  # [W]

    dT_air = (Q_heat + Q_sol_air - Q_conv_air_cov + Q_vent_air) / (p.C_air * A)

    # Solar absorbed by cover
    Q_sol_cov = p.tau_sol * (1.0 - p.alpha_air) * I_sol * A_c * 0.05  # small fraction
    # Cover-outdoor exchange
    Q_cov_out = p.k_cov_out * A_c * (T_out - T_cov)

    dT_cov = (Q_conv_air_cov + Q_sol_cov + Q_cov_out) / (p.C_cov * A_c)

    # --- Humidity dynamics ---

    # Crop evapotranspiration (simplified: constant + temperature-driven component)
    ET = p.ET_base * (1.0 + 0.02 * max(T_air - 18.0, 0.0))  # [g/(m² s)]
    W_et = ET * A                                              # [g/s]

    # Ventilation humidity exchange
    W_vent = vent_rate * p.rho_air * (H_out - H)             # [g/s] = [kg air/s] x [g/kg]

    dH = (W_et + W_vent) / (V * p.rho_air)                   # [g/kg/s] = [g/s] / [kg air]

    return np.array([dT_air, dT_cov, dH])


def linearise(x_op, u_op, d_op, params: GreenhouseParams, eps=1e-4):
    """
    Numerical Jacobian linearisation around an operating point.

    Returns (A, B, Bd) matrices for:
        dx_dot = A @ dx + B @ du + Bd @ dd
    """
    n = len(x_op)
    m = len(u_op)
    nd = len(d_op)

    f0 = greenhouse_ode(0, x_op, u_op, d_op, params)

    A = np.zeros((n, n))
    for i in range(n):
        xp = x_op.copy(); xp[i] += eps
        A[:, i] = (greenhouse_ode(0, xp, u_op, d_op, params) - f0) / eps

    B = np.zeros((n, m))
    for i in range(m):
        up = u_op.copy(); up[i] += eps
        B[:, i] = (greenhouse_ode(0, x_op, up, d_op, params) - f0) / eps

    Bd = np.zeros((n, nd))
    for i in range(nd):
        dp = d_op.copy(); dp[i] += eps
        Bd[:, i] = (greenhouse_ode(0, x_op, u_op, dp, params) - f0) / eps

    return A, B, Bd
