import pandas as pd
import numpy as np
import pymc as pm
import json

def time_to_seconds(time_str):
    if pd.isna(time_str):
        return np.nan
    parts = str(time_str).split(':')
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    return float(time_str)

# ============================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================

def extract_pace_variability(pace_json):
    """Extract coefficient of variation from pace data - indicates terrain irregularity"""
    if pd.isna(pace_json):
        return np.nan
    try:
        data = json.loads(pace_json)
        velocities = data.get('velocity', [])
        if len(velocities) < 10:
            return np.nan
        velocities = [v for v in velocities if v > 0]  # Filter zeros
        if len(velocities) < 10:
            return np.nan
        return np.std(velocities) / (np.mean(velocities) + 1e-5)
    except:
        return np.nan

def extract_elevation_metrics(elev_json):
    """Extract slope variability and terrain inversions from elevation profile"""
    if pd.isna(elev_json):
        return np.nan, np.nan
    try:
        data = json.loads(elev_json)
        altitudes = data.get('altitude', [])
        if len(altitudes) < 10:
            return np.nan, np.nan
        diffs = np.diff(altitudes)
        # Slope variability = broken/technical terrain
        slope_std = np.std(diffs)
        # Inversions (up→down transitions) = rolling terrain
        signs = np.sign(diffs)
        inversions = np.sum(np.diff(signs) != 0) / len(diffs)  # Normalized
        return slope_std, inversions
    except:
        return np.nan, np.nan

def extract_best_pace(efforts_json, target_distance='5K'):
    """Extract pace (min/km) from best efforts for a given distance"""
    if pd.isna(efforts_json):
        return np.nan
    try:
        efforts = json.loads(efforts_json)
        for e in efforts:
            if e['name'] == target_distance:
                pace_min_km = (e['moving_time'] / 60) / (e['distance'] / 1000)
                return pace_min_km
        return np.nan
    except:
        return np.nan

# ============================================================
# DATA LOADING & PREPROCESSING
# ============================================================

df = pd.read_csv('data/activities.csv')
df = df[df['type'].isin(['Run', 'Trail Run'])].copy()

# Basic features
df['moving_time_sec'] = df['moving_time'].apply(time_to_seconds)
df['minutes'] = df['moving_time_sec'] / 60
df['dist_km'] = df['distance']
df['d_plus_km'] = df['total_elevation_gain'] / 1000
df['avg_hr'] = df['average_heartrate'].fillna(df['average_heartrate'].mean())

# ============================================================
# ENRICHED FEATURES FROM JSON DATA
# ============================================================

print("Extracting enriched features...")

# Pace variability (terrain irregularity indicator)
df['pace_cv'] = df['pace_data'].apply(extract_pace_variability)

# Elevation metrics
elev_metrics = df['elevation_data'].apply(lambda x: pd.Series(extract_elevation_metrics(x)))
df['slope_variability'] = elev_metrics[0]
df['terrain_inversions'] = elev_metrics[1]

# Best efforts - reference paces
df['pace_5k'] = df['best_efforts'].apply(lambda x: extract_best_pace(x, '5K'))
df['pace_10k'] = df['best_efforts'].apply(lambda x: extract_best_pace(x, '10K'))
df['pace_1k'] = df['best_efforts'].apply(lambda x: extract_best_pace(x, '1K'))

# Rest ratio (technical terrain indicator)
df['elapsed_sec'] = df['elapsed_time'].apply(time_to_seconds)
df['rest_ratio'] = df['elapsed_sec'] / (df['moving_time_sec'] + 1e-5)

# Filter and clean
df = df[df['dist_km'] > 2]
df = df.dropna(subset=['minutes', 'dist_km', 'd_plus_km'])

# ============================================================
# FEATURE SUMMARY
# ============================================================

print(f"\n{'='*60}")
print(f"DONNÉES ENRICHIES - RÉSUMÉ")
print(f"{'='*60}")
print(f"Nombre d'activités après nettoyage: {len(df)}")
print(f"Moyenne minutes: {df['minutes'].mean():.1f}")
print(f"\n--- Features enrichies ---")
print(f"Pace CV (variabilité allure): {df['pace_cv'].mean():.3f} (disponible: {df['pace_cv'].notna().sum()})")
print(f"Slope variability: {df['slope_variability'].mean():.3f} (disponible: {df['slope_variability'].notna().sum()})")
print(f"Terrain inversions: {df['terrain_inversions'].mean():.3f} (disponible: {df['terrain_inversions'].notna().sum()})")
print(f"Rest ratio moyen: {df['rest_ratio'].mean():.3f}")
print(f"\n--- Best Efforts (allures référence) ---")
print(f"Pace 1K moyen: {df['pace_1k'].mean():.2f} min/km (disponible: {df['pace_1k'].notna().sum()})")
print(f"Pace 5K moyen: {df['pace_5k'].mean():.2f} min/km (disponible: {df['pace_5k'].notna().sum()})")
print(f"Pace 10K moyen: {df['pace_10k'].mean():.2f} min/km (disponible: {df['pace_10k'].notna().sum()})")
print(f"{'='*60}\n")

# Prepare enriched features (fill NaN with median or 0)
pace_cv_clean = df['pace_cv'].fillna(df['pace_cv'].median()).fillna(0).values
rest_ratio_clean = df['rest_ratio'].fillna(1.0).clip(1.0, 2.0).values  # Clip extreme values
terrain_inv_clean = df['terrain_inversions'].fillna(df['terrain_inversions'].median()).fillna(0).values

# Reference pace from best efforts (use 5K or fallback to computed)
pace_ref = df['pace_5k'].fillna(df['pace_10k']).fillna(df['minutes'] / df['dist_km']).values

dist_mean = df['dist_km'].mean()
dist_std = df['dist_km'].std()
elev_mean = df['d_plus_km'].mean()
elev_std = df['d_plus_km'].std()

X_dist = (df['dist_km'] - dist_mean) / dist_std
X_elev = (df['d_plus_km'] - elev_mean) / elev_std

# ============================================================
# ENRICHED PYMC MODEL
# ============================================================

with pm.Model() as mixture_trail_model:
    w = pm.Dirichlet('w', a=np.array([1, 1]))

    # Base pace - now informed by best efforts data
    pace_base = pm.Normal('pace_base', 
                          mu=np.array([6.0, 4.8]), 
                          sigma=np.array([1.0, 0.5]), 
                          shape=2)
    
    # Elevation impact
    effort_dplus = pm.Normal('effort_dplus', 
                             mu=np.array([9.0, 5.0]), 
                             sigma=np.array([2.0, 1.0]), 
                             shape=2)
    
    # Fatigue over distance
    beta_fatigue = pm.Normal('beta_fatigue', mu=0.012, sigma=0.003)
    
    # Heart rate effect
    beta_hr = pm.Normal('beta_hr', mu=0, sigma=0.2)
    
    # NEW: Terrain technicality effect (rest_ratio > 1 = more stops)
    beta_terrain = pm.Normal('beta_terrain', mu=0.5, sigma=0.3)
    
    # NEW: Pace variability effect (irregular pacing = slower)
    beta_pace_var = pm.Normal('beta_pace_var', mu=1.0, sigma=0.5)
    
    # NEW: Terrain inversions effect (rolling terrain = slower)
    beta_inversions = pm.Normal('beta_inversions', mu=2.0, sigma=1.0)

    # Prepare input features
    hr_diff = df['average_heartrate'].fillna(145).values - 145
    dplus_ratio = df['total_elevation_gain'].values / (df['dist_km'].values * 100 + 1e-5)
    terrain_penalty = (rest_ratio_clean - 1.0)  # 0 if rest_ratio=1, positive if stops
    
    # ENRICHED mu formulas with new features
    mu_0 = (pace_base[0] + 
            (effort_dplus[0] * dplus_ratio) + 
            (beta_fatigue * df['dist_km'].values) + 
            (beta_hr * hr_diff) +
            (beta_terrain * terrain_penalty) +
            (beta_pace_var * pace_cv_clean) +
            (beta_inversions * terrain_inv_clean)
           ) * df['dist_km'].values
    
    mu_1 = (pace_base[1] + 
            (effort_dplus[1] * dplus_ratio) + 
            (beta_fatigue * df['dist_km'].values) + 
            (beta_hr * hr_diff) +
            (beta_terrain * terrain_penalty) +
            (beta_pace_var * pace_cv_clean) +
            (beta_inversions * terrain_inv_clean)
           ) * df['dist_km'].values

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

d_target = 42
d_plus_target = 100

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

# Summary of all parameters including new enriched ones
print("\n" + "="*60)
print("RÉSUMÉ DU MODÈLE ENRICHI")
print("="*60)
print(az.summary(trace, var_names=[
    'pace_base', 'effort_dplus', 'beta_fatigue', 'beta_hr',
    'beta_terrain', 'beta_pace_var', 'beta_inversions', 'sigma'
]))

