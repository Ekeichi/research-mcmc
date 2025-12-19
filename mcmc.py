import pandas as pd
import numpy as np
import pymc as pm

def time_to_seconds(time_str):
    if pd.isna(time_str):
        return np.nan
    parts = str(time_str).split(':')
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    return float(time_str)

df = pd.read_csv('data/activities.csv')
df = df[df['type'].isin(['Run', 'Trail Run'])].copy()

df['moving_time_sec'] = df['moving_time'].apply(time_to_seconds)
df['minutes'] = df['moving_time_sec'] / 60
df['dist_km'] = df['distance']
df['d_plus_km'] = df['total_elevation_gain'] / 1000
df['avg_hr'] = df['average_heartrate'].fillna(df['average_heartrate'].mean())

df = df[df['dist_km'] > 2]
df = df.dropna(subset=['minutes', 'dist_km', 'd_plus_km'])

print(f"Nombre d'activités après nettoyage: {len(df)}")
print(f"Moyenne minutes: {df['minutes'].mean():.1f}")

dist_mean = df['dist_km'].mean()
dist_std = df['dist_km'].std()
elev_mean = df['d_plus_km'].mean()
elev_std = df['d_plus_km'].std()

X_dist = (df['dist_km'] - dist_mean) / dist_std
X_elev = (df['d_plus_km'] - elev_mean) / elev_std

import pymc as pm
import numpy as np

with pm.Model() as mixture_trail_model:
    w = pm.Dirichlet('w', a=np.array([1, 1]))

    pace_base = pm.Normal('pace_base', 
                          mu=np.array([6.0, 4.8]), 
                          sigma=np.array([1.0, 0.5]), 
                          shape=2)
    
    effort_dplus = pm.Normal('effort_dplus', 
                             mu=np.array([9.0, 5.0]), 
                             sigma=np.array([2.0, 1.0]), 
                             shape=2)
    
    beta_fatigue = pm.Normal('beta_fatigue', mu=0.012, sigma=0.003)
    beta_hr = pm.Normal('beta_hr', mu=0, sigma=0.2)

    hr_diff = df['average_heartrate'].values - 145
    dplus_ratio = df['total_elevation_gain'].values / (df['dist_km'].values * 100 + 1e-5)
    
    mu_0 = (pace_base[0] + (effort_dplus[0] * dplus_ratio) + 
            (beta_fatigue * df['dist_km'].values) + (beta_hr * hr_diff)) * df['dist_km'].values
    
    mu_1 = (pace_base[1] + (effort_dplus[1] * dplus_ratio) + 
            (beta_fatigue * df['dist_km'].values) + (beta_hr * hr_diff)) * df['dist_km'].values

    mu_combined = pm.math.stack([mu_0, mu_1], axis=1)

    sigma = pm.HalfNormal('sigma', sigma=10)
    
    y_obs = pm.NormalMixture('y_obs', 
                             w=w, 
                             mu=mu_combined, 
                             sigma=sigma, 
                             observed=df['minutes'].values)

    trace = pm.sample(2000, tune=1500, target_accept=0.95)

import arviz as az

az.plot_trace(trace)
az.plot_posterior(trace, var_names=['pace_base', 'effort_dplus', 'beta_fatigue', 'sigma'])
az.plot_pair(trace, var_names=['pace_base', 'effort_dplus', 'beta_fatigue'], divergences=True)

d_target = 79
d_plus_target = 2200

pace_samples = trace.posterior['pace_base'].values[:, :, 1].flatten()
effort_samples = trace.posterior['effort_dplus'].values[:, :, 1].flatten()
fatigue_samples = trace.posterior['beta_fatigue'].values.flatten()

dplus_ratio_target = d_plus_target / (d_target * 100)
predicted_pace = pace_samples + (effort_samples * dplus_ratio_target) + (fatigue_samples * d_target)
predictions = predicted_pace * d_target

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.hist(predictions, bins=50, edgecolor='black', alpha=0.7)
plt.axvline(np.mean(predictions), color='red', linestyle='--', label=f'Moyenne: {np.mean(predictions):.0f} min')
plt.axvline(np.percentile(predictions, 5), color='orange', linestyle=':', label=f'5%: {np.percentile(predictions, 5):.0f} min')
plt.axvline(np.percentile(predictions, 95), color='orange', linestyle=':', label=f'95%: {np.percentile(predictions, 95):.0f} min')
plt.title(f"Distribution du temps final - {d_target}km, {d_plus_target}m D+")
plt.xlabel("Minutes")
plt.ylabel("Fréquence")
plt.legend()
plt.show()

mean_time = np.mean(predictions)
mean_pace = np.mean(predicted_pace)
print(f"\n=== PRÉDICTION POUR {d_target}km avec {d_plus_target}m D+ ===")
print(f"Allure moyenne prédite: {mean_pace:.1f} min/km")
print(f"Temps moyen: {mean_time:.0f} min ({mean_time/60:.1f}h)")
print(f"Intervalle 90%: {np.percentile(predictions, 5):.0f} - {np.percentile(predictions, 95):.0f} min")
print(f"               ({np.percentile(predictions, 5)/60:.1f}h - {np.percentile(predictions, 95)/60:.1f}h)")

print(az.summary(trace, var_names=['pace_base', 'effort_dplus', 'beta_fatigue', 'beta_hr']))
