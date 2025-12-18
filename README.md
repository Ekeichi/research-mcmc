# Trail Time Predictor

Bayesian model to predict trail running times using your Strava data.

## What it does

Uses PyMC to fit a model with base pace, elevation cost, fatigue factor, and heart rate adjustment. Then predicts your finish time for a given distance/elevation.

## Setup

```bash
python3.11 -m venv venv311
source venv311/bin/activate
pip install pymc arviz pandas matplotlib
```

Needs Python 3.11 (PyMC doesn't work with 3.14 yet).

## Run

```bash
python3.11 mcmc.py
```

Edit `d_target` and `d_plus_target` in the script to change your race parameters.

## Data

Put your Strava export in `data/activities.csv`. Needs columns: `type`, `moving_time`, `distance`, `total_elevation_gain`, `average_heartrate`.

## Results

Example prediction for 43km with 2200m D+:

```
Predicted pace: 9.8 min/km
Finish time: 7h00 (90% CI: 6h42 - 7h22)
```

### Model parameters

| Parameter | Mean | Description |
|-----------|------|-------------|
| pace_base | 4.9 min/km | Base flat pace |
| effort_dplus | 8.8 | Cost per 100m elevation |
| beta_fatigue | 0.009 | Slowdown per km |
| beta_hr | -0.09 | HR adjustment |
