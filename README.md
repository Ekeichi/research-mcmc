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

=== PRÉDICTION POUR 43km avec 2200m D+ ===
Allure moyenne prédite: 9.8 min/km
Temps moyen: 422 min (7.0h)
Intervalle 90%: 401 - 442 min
               (6.7h - 7.4h)
               mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
pace_base     4.908  0.177   4.579    5.243      0.002    0.002    6142.0    5640.0    1.0
effort_dplus  8.808  0.670   7.566   10.073      0.008    0.007    6885.0    5860.0    1.0
beta_fatigue  0.009  0.003   0.003    0.015      0.000    0.000    6751.0    5527.0    1.0
beta_hr      -0.092  0.013  -0.117   -0.067      0.000    0.000    8116.0    5987.0    1.0
