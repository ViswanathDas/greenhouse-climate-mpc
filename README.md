# Greenhouse Climate MPC

Model Predictive Control for greenhouse climate management — tracking temperature and humidity setpoints under real-world disturbances (solar irradiance, outdoor temperature, ventilation), with Extended Kalman Filter state estimation on noisy sensor readings.

## Background

Modern greenhouse production systems run 24/7 across highly variable outdoor conditions. The challenge is not just setpoint tracking — it is doing so efficiently, anticipating disturbances (weather, day/night cycles), and handling sensor noise without overactuating heating, cooling, or ventilation systems. This project builds a simulation environment and MPC controller that addresses exactly that.

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

## Key Features

- **Thermal + humidity model**: First-principles ODE model (air temperature, cover temperature, humidity ratio) driven by heating power, ventilation rate, solar irradiance
- **EKF state estimation**: Handles noisy temperature and humidity sensor readings; estimates unmeasured states (cover temperature, vapour pressure deficit)
- **MPC controller**: Receding horizon optimisation over 30-minute prediction window; constraints on heating/ventilation inputs; penalises both setpoint deviation and actuator effort
- **Weather disturbance feed**: KNMI open data API integration for real Dutch weather (outdoor temperature, solar irradiance, wind speed) as feedforward disturbance
- **Visualisation**: Step response plots, prediction vs actual trajectories, EKF innovation sequences

## Tech Stack

- Python 3.11+
- NumPy, SciPy (ODE integration, linearisation)
- cvxpy (MPC quadratic programme)
- matplotlib (visualisation)
- requests (KNMI weather API)
- pytest (testing)

## Results (Target)

| Metric | Open-loop | MPC |
|--------|-----------|-----|
| Temperature RMS error (°C) | — | < 0.5 |
| Humidity RMS error (%RH) | — | < 3.0 |
| Actuator switching events | — | < open-loop |

## Running the Demo

```bash
pip install -r requirements.txt
python src/utils/fetch_weather.py        # pull KNMI data
python notebooks/run_simulation.py       # full closed-loop run
```

## Status

Active development. Core model and EKF are implemented. MPC formulation in progress.

## Background Reading

- Vanthoor et al. (2011) — "A methodology for model-based greenhouse design"
- Rawlings, Mayne & Diehl — *Model Predictive Control: Theory, Computation, and Design*
- My MSc thesis: [Integrated Planning and Control for Autonomous Vehicles](http://resolver.tudelft.nl/uuid:84761bf6-db46-454f-8d42-4f97896f55fd) (MPC + MIMPC methodology)
