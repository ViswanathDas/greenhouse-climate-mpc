"""
KNMI Weather Data Utilities
============================
Fetches and formats weather data from the KNMI open data API for use as
disturbance inputs in the greenhouse simulation.

KNMI Open Data API: https://developer.dataplatform.knmi.nl/open-data-api
Station used by default: De Bilt (260) — central Netherlands reference station.

Returned disturbance vector per timestep: [T_out (°C), I_sol (W/m²), H_out (g/kg)]
"""

import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Optional


KNMI_API_BASE = "https://api.dataplatform.knmi.nl/open-data/v1"

# Default station: De Bilt (260) — covers Dutch greenhouse belt well
DEFAULT_STATION = "260"


def synthesise_weather(
    n_hours: int = 24,
    dt_seconds: float = 150.0,
    season: str = "spring",
    seed: int = 42,
) -> np.ndarray:
    """
    Generate synthetic but physically realistic Dutch weather for simulation.

    Used when KNMI API is unavailable or for reproducible unit tests.

    Parameters
    ----------
    n_hours   : int     — simulation duration in hours
    dt_seconds: float   — timestep in seconds
    season    : str     — 'spring', 'summer', 'autumn', 'winter'
    seed      : int     — random seed for noise

    Returns
    -------
    d_seq : array [n_steps, 3]  — [T_out (°C), I_sol (W/m²), H_out (g/kg)]
    """
    rng = np.random.default_rng(seed)

    season_params = {
        "spring": dict(T_mean=12.0, T_amp=5.0, I_peak=400.0, H_mean=7.0),
        "summer": dict(T_mean=20.0, T_amp=6.0, I_peak=650.0, H_mean=9.0),
        "autumn": dict(T_mean=10.0, T_amp=4.0, I_peak=200.0, H_mean=7.5),
        "winter": dict(T_mean=4.0,  T_amp=3.0, I_peak=80.0,  H_mean=5.5),
    }
    sp = season_params.get(season, season_params["spring"])

    n_steps = int(n_hours * 3600 / dt_seconds)
    t = np.arange(n_steps) * dt_seconds  # seconds from midnight

    # Temperature: diurnal sinusoid + noise
    T_out = (
        sp["T_mean"]
        + sp["T_amp"] * np.sin(2 * np.pi * t / 86400 - np.pi / 2)
        + rng.normal(0, 0.5, n_steps)
    )

    # Solar irradiance: half-sine during daylight hours (6:00–20:00) + noise
    day_frac = (t % 86400) / 86400
    daylight = np.clip(np.sin(np.pi * (day_frac - 0.25) / 0.58), 0, None)
    I_sol = sp["I_peak"] * daylight + rng.normal(0, 15, n_steps).clip(0)

    # Humidity: slow variation around mean + noise
    H_out = (
        sp["H_mean"]
        + 1.5 * np.sin(2 * np.pi * t / 86400 + np.pi)
        + rng.normal(0, 0.3, n_steps)
    ).clip(2.0, 20.0)

    return np.column_stack([T_out, I_sol, H_out])


def fetch_knmi_hourly(
    date_start: datetime,
    n_hours: int = 48,
    station: str = DEFAULT_STATION,
    api_key: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Fetch hourly KNMI weather observations from the open data API.

    Returns DataFrame with columns: datetime, T_out, I_sol, H_out
    or None if the API call fails (falls back to synthesise_weather).

    Note: Requires a free KNMI API key from https://developer.dataplatform.knmi.nl
    Set api_key or KNMI_API_KEY environment variable.
    """
    import os
    key = api_key or os.environ.get("KNMI_API_KEY")
    if not key:
        print("KNMI_API_KEY not set — using synthetic weather data.")
        return None

    # KNMI Open Data API — hourly observations dataset
    dataset = "Actuele10mindataKNMIstations"
    try:
        url = f"{KNMI_API_BASE}/datasets/{dataset}/versions/2/files"
        # Simplified: fetch available file list and download the relevant ones
        # Full implementation requires iterating file list by timestamp
        # For now, return None and use synthetic data
        print("KNMI API integration: full fetch not yet implemented — using synthetic data.")
        return None
    except requests.RequestException as e:
        print(f"KNMI API error: {e} — using synthetic data.")
        return None


def get_disturbance_sequence(
    n_steps: int,
    dt_seconds: float = 150.0,
    season: str = "spring",
    use_api: bool = False,
    api_key: Optional[str] = None,
) -> np.ndarray:
    """
    Main entry point: return disturbance sequence for simulation.

    Tries KNMI API if use_api=True, falls back to synthetic otherwise.

    Returns array [n_steps, 3]: [T_out, I_sol, H_out]
    """
    if use_api:
        df = fetch_knmi_hourly(datetime.now(), n_hours=int(n_steps * dt_seconds / 3600) + 1, api_key=api_key)
        if df is not None:
            # Interpolate to simulation timestep
            # (implementation: resample df to dt_seconds resolution)
            pass  # TODO: interpolation step

    n_hours = int(n_steps * dt_seconds / 3600) + 1
    return synthesise_weather(n_hours=n_hours, dt_seconds=dt_seconds, season=season)[:n_steps]
