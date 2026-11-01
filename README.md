# Greenhouse Climate MPC

[![tests](https://github.com/ViswanathDas/greenhouse-climate-mpc/actions/workflows/tests.yml/badge.svg)](https://github.com/ViswanathDas/greenhouse-climate-mpc/actions/workflows/tests.yml)

Model Predictive Control for greenhouse climate management — tracking temperature and humidity setpoints under real-world disturbances (solar irradiance, outdoor temperature, ventilation), with Extended Kalman Filter state estimation on noisy sensor readings.

## Background

Modern greenhouse production systems run 24/7 across highly variable outdoor conditions. The challenge is not just setpoint tracking — it is doing so efficiently, anticipating disturbances (weather, day/night cycles), and handling sensor noise without overactuating heating, cooling, or ventilation systems. This project builds a simulation environment and an MPC controller for that problem.

Inspired by academic work on Vanthoor-style thermal-mass greenhouse models and their application to precision horticulture. The MPC formulation here draws on techniques from my MSc thesis (APF-MPC + MIMPC for autonomous vehicle planning) adapted to a thermal process control context.

## Project Structure

```
greenhouse-climate-mpc/
├── src/
│   ├── model/          # Greenhouse thermal + humidity dynamics (nonlinear ODE)
│   ├── controller/     # MPC formulation (linearised + nonlinear variants)
│   ├── estimation/     # Extended Kalman Filter for state estimation
│   └── utils/          # Weather data loading, plotting helpers
├── notebooks/          # Exploratory analysis and demo runs
├── data/               # KNMI weather data, setpoint profiles
├── tests/              # Unit tests for model, EKF, MPC
└── docs/               # Architecture notes, model derivation
```

## What is implemented

| Part | Status |
|---|---|
| Thermal + humidity model (air temperature, cover temperature, absolute humidity), first-principles ODE | Done |
| Extended Kalman Filter on noisy temperature and humidity sensors, estimating the unmeasured cover temperature | Done |
| EKF handling of sensor dropout (missing channels passed as NaN, update uses the available channels or is skipped) | Done, tested |
| Linearised MPC (successive linearisation, cvxpy QP, 30 min horizon, input and rate constraints) | Working, tuning in progress |
| Weather disturbances | Synthetic Dutch seasonal profiles by default; KNMI open data API hook in `src/utils/weather.py` (needs an API key) |
| CO2 balance, condensation on the cover, energy cost in the objective | Not yet |

## Results (24 h spring day, synthetic weather, start at 10 °C)

| Metric | Value |
|---|---|
| RMS air temperature error, after the 2 h warm-up | 1.58 °C |
| RMS humidity error, after the 2 h warm-up | 3.40 g/kg |

Temperature tracks well at night. Around midday solar gain pushes the air 2 to 3 °C above the setpoint because ventilation is the only cooling and its opening rate is limited. Humidity sits above the 9 g/kg target at night because ventilating harder costs heat. Both are the next things to work on (energy-aware cost, a dehumidification input, better tuning).

**Fix history.** An earlier version had a unit error in the humidity balance (ventilation moisture flow divided by 1000 and the air mass multiplied by 1/1000), which drove humidity to physically impossible values. The thermal capacities were also below that of the air alone. Both are corrected, and the MPC humidity weight was retuned afterwards.

## Tech Stack

- Python 3.11+
- NumPy, SciPy (ODE integration, linearisation)
- cvxpy (MPC quadratic programme)
- matplotlib (visualisation)
- requests (KNMI weather API)
- pytest (testing)

## Running

```bash
pip install -r requirements.txt
python notebooks/run_simulation.py       # 24 h closed loop with synthetic weather
pytest tests/ -q                         # model, EKF and dropout tests
```

## Status

Active development. Model and EKF (including dropout handling) are done and tested. The MPC runs closed loop and is being tuned.

## Background Reading

- Vanthoor et al. (2011) — "A methodology for model-based greenhouse design"
- Rawlings, Mayne & Diehl — *Model Predictive Control: Theory, Computation, and Design*
- My MSc thesis: [Integrated Planning and Control for Autonomous Vehicles](http://resolver.tudelft.nl/uuid:84761bf6-db46-454f-8d42-4f97896f55fd) (MPC + MIMPC methodology)
