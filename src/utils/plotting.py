"""
Visualisation utilities for greenhouse simulation results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional


def plot_simulation(
    t: np.ndarray,
    x_true: np.ndarray,
    x_est: np.ndarray,
    u_seq: np.ndarray,
    d_seq: np.ndarray,
    x_ref: np.ndarray,
    title: str = "Greenhouse MPC Simulation",
    save_path: Optional[str] = None,
):
    """
    Full simulation dashboard: states (true + estimated), inputs, disturbances.

    Parameters
    ----------
    t      : array [T,]     — time vector [s]
    x_true : array [T, 3]   — true states [T_air, T_cov, H]
    x_est  : array [T, 3]   — EKF estimates
    u_seq  : array [T, 2]   — applied inputs [u_heat, u_vent]
    d_seq  : array [T, 3]   — disturbances [T_out, I_sol, H_out]
    x_ref  : array [3,]     — setpoint
    """
    t_h = t / 3600  # hours

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(title, fontsize=13, fontweight='bold')
    gs = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.35)

    # --- Top left: Air temperature ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t_h, x_true[:, 0], 'k-', linewidth=1.2, label='True T_air')
    ax1.plot(t_h, x_est[:, 0], 'b--', linewidth=1.0, label='EKF estimate')
    ax1.axhline(x_ref[0], color='r', linestyle=':', linewidth=1.2, label=f'Setpoint ({x_ref[0]:.1f}°C)')
    ax1.set_ylabel('Temperature [°C]')
    ax1.set_title('Air Temperature')
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    # --- Top right: Humidity ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t_h, x_true[:, 2], 'k-', linewidth=1.2, label='True H')
    ax2.plot(t_h, x_est[:, 2], 'b--', linewidth=1.0, label='EKF estimate')
    ax2.axhline(x_ref[2], color='r', linestyle=':', linewidth=1.2, label=f'Setpoint ({x_ref[2]:.1f} g/kg)')
    ax2.set_ylabel('Humidity [g/kg]')
    ax2.set_title('Humidity Ratio')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    # --- Middle left: Heating input ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.step(t_h, u_seq[:, 0], 'darkorange', linewidth=1.2, where='post')
    ax3.set_ylabel('Heating power [W/m²]')
    ax3.set_title('Control Input: Heating')
    ax3.set_ylim([-5, 220])
    ax3.grid(True, alpha=0.3)

    # --- Middle right: Ventilation input ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.step(t_h, u_seq[:, 1], 'steelblue', linewidth=1.2, where='post')
    ax4.set_ylabel('Vent opening [fraction]')
    ax4.set_title('Control Input: Ventilation')
    ax4.set_ylim([-0.02, 1.05])
    ax4.grid(True, alpha=0.3)

    # --- Bottom left: Outdoor temperature ---
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(t_h, d_seq[:, 0], 'gray', linewidth=1.0)
    ax5.set_ylabel('T_out [°C]')
    ax5.set_xlabel('Time [h]')
    ax5.set_title('Disturbance: Outdoor Temperature')
    ax5.grid(True, alpha=0.3)

    # --- Bottom right: Solar irradiance ---
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.fill_between(t_h, d_seq[:, 1], alpha=0.5, color='gold')
    ax6.plot(t_h, d_seq[:, 1], 'goldenrod', linewidth=0.8)
    ax6.set_ylabel('I_sol [W/m²]')
    ax6.set_xlabel('Time [h]')
    ax6.set_title('Disturbance: Solar Irradiance')
    ax6.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()

    return fig


def plot_ekf_diagnostics(t, innovations, P_trace, save_path=None):
    """Plot EKF innovation sequence and covariance trace to assess filter health."""
    t_h = t / 3600

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle('EKF Diagnostics', fontsize=12, fontweight='bold')

    axes[0].plot(t_h, innovations[:, 0], 'b-', linewidth=0.8, label='T_air innovation')
    axes[0].plot(t_h, innovations[:, 1], 'r-', linewidth=0.8, label='H innovation')
    axes[0].axhline(0, color='k', linestyle='--', linewidth=0.5)
    axes[0].set_ylabel('Innovation')
    axes[0].legend(fontsize=8)
    axes[0].set_title('Measurement Innovations (should be zero-mean white noise)')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_h, P_trace, 'purple', linewidth=1.0)
    axes[1].set_ylabel('trace(P)')
    axes[1].set_xlabel('Time [h]')
    axes[1].set_title('Covariance Trace (should decrease from P0 and stabilise)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    return fig
