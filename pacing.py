import pymc as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
import xml.etree.ElementTree as ET
from math import radians, sin, cos, sqrt, atan2
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
        velocities = [v for v in velocities if v > 0]
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
        slope_std = np.std(diffs)
        signs = np.sign(diffs)
        inversions = np.sum(np.diff(signs) != 0) / len(diffs)
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
df = df.dropna(subset=['minutes', 'dist_km', 'total_elevation_gain'])

# ============================================================
# FEATURE SUMMARY
# ============================================================

print(f"\n{'='*60}")
print(f"DONNÉES ENRICHIES - RÉSUMÉ")
print(f"{'='*60}")
print(f"Nombre d'activités: {len(df)}")
print(f"Pace CV moyen: {df['pace_cv'].mean():.3f} (n={df['pace_cv'].notna().sum()})")
print(f"Pace 5K moyen: {df['pace_5k'].mean():.2f} min/km (n={df['pace_5k'].notna().sum()})")
print(f"Rest ratio moyen: {df['rest_ratio'].mean():.3f}")
print(f"{'='*60}\n")

# Prepare enriched features
pace_cv_clean = df['pace_cv'].fillna(df['pace_cv'].median()).fillna(0).values
rest_ratio_clean = df['rest_ratio'].fillna(1.0).clip(1.0, 2.0).values
terrain_inv_clean = df['terrain_inversions'].fillna(df['terrain_inversions'].median()).fillna(0).values

GPX_FILE = 'data/6D.gpx'
FC_CIBLE = 155

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

def parse_gpx_elevation_per_km(gpx_path, elevation_threshold=3.0):
    tree = ET.parse(gpx_path)
    root = tree.getroot()
    ns = {'gpx': 'http://www.topografix.com/GPX/1/1'}
    
    points = []
    for trkpt in root.findall('.//gpx:trkpt', ns):
        lat = float(trkpt.get('lat'))
        lon = float(trkpt.get('lon'))
        ele_tag = trkpt.find('gpx:ele', ns)
        ele = float(ele_tag.text) if ele_tag is not None else 0
        points.append((lat, lon, ele))
    
    cumul_dist = 0
    km_elevations = []
    current_km = 1
    dplus_in_km = 0
    reference_ele = points[0][2]
    
    for i in range(1, len(points)):
        lat1, lon1, ele1 = points[i-1]
        lat2, lon2, ele2 = points[i]
        cumul_dist += haversine(lat1, lon1, lat2, lon2)
        
        ele_diff = ele2 - reference_ele
        if ele_diff > elevation_threshold:
            dplus_in_km += ele_diff
            reference_ele = ele2
        elif ele_diff < -elevation_threshold:
            reference_ele = ele2
        
        if cumul_dist >= current_km:
            km_elevations.append(dplus_in_km)
            current_km += 1
            dplus_in_km = 0
    
    if dplus_in_km > 0:
        km_elevations.append(dplus_in_km)
    
    return np.array(km_elevations)

dplus_per_km = parse_gpx_elevation_per_km(GPX_FILE, elevation_threshold=5.0)
DISTANCE_CIBLE = len(dplus_per_km)
D_PLUS_CIBLE = dplus_per_km.sum()
print(f"GPX chargé: {DISTANCE_CIBLE}km, {D_PLUS_CIBLE:.0f}m D+")

# ============================================================
# ENRICHED MIXTURE MODEL
# ============================================================

with pm.Model() as mixture_model:
    w = pm.Dirichlet('w', a=np.array([1, 1]))
    
    # [0] Entraînement, [1] Performance
    pace_base = pm.Normal('pace_base', mu=[7.5, 5.0], sigma=[1.0, 0.5], shape=2)
    effort_dplus = pm.Normal('effort_dplus', mu=[15.0, 6.0], sigma=[2.0, 1.0], shape=2)
    
    # Fatigue over distance
    beta_fatigue = pm.Normal('beta_fatigue', mu=0.009, sigma=0.003)
    
    # Heart rate effect
    beta_hr = pm.Normal('beta_hr', mu=0, sigma=0.2)
    
    # NEW: Terrain technicality effect (rest_ratio > 1 = more stops)
    beta_terrain = pm.Normal('beta_terrain', mu=0.5, sigma=0.3)
    
    # NEW: Pace variability effect (irregular pacing = slower)
    beta_pace_var = pm.Normal('beta_pace_var', mu=1.0, sigma=0.5)
    
    # NEW: Terrain inversions effect (rolling terrain = slower)
    beta_inversions = pm.Normal('beta_inversions', mu=2.0, sigma=1.0)

    # Prepare input features
    dplus_ratio = df['total_elevation_gain'].values / (df['dist_km'].values * 100 + 1e-5)
    hr_diff = df['average_heartrate'].fillna(145).values - 145
    terrain_penalty = (rest_ratio_clean - 1.0)
    
    # ENRICHED mu formulas
    mu_0 = (pace_base[0] + 
            effort_dplus[0] * dplus_ratio + 
            beta_fatigue * df['dist_km'].values + 
            beta_hr * hr_diff +
            beta_terrain * terrain_penalty +
            beta_pace_var * pace_cv_clean +
            beta_inversions * terrain_inv_clean
           ) * df['dist_km'].values
    
    mu_1 = (pace_base[1] + 
            effort_dplus[1] * dplus_ratio + 
            beta_fatigue * df['dist_km'].values + 
            beta_hr * hr_diff +
            beta_terrain * terrain_penalty +
            beta_pace_var * pace_cv_clean +
            beta_inversions * terrain_inv_clean
           ) * df['dist_km'].values
    
    mu_combined = pm.math.stack([mu_0, mu_1], axis=1)
    sigma = pm.HalfNormal('sigma', sigma=10)
    
    y_obs = pm.NormalMixture('y_obs', w=w, mu=mu_combined, sigma=sigma, observed=df['minutes'].values)
    
    trace = pm.sample(2000, tune=1000, target_accept=0.95, return_inferencedata=True)

def simulate_pacing_tunnel(trace, dplus_per_km_array, fc_target, terrain_tech=0.05, pace_var=0.15, terrain_inv=0.1):
    """
    Simulate pacing scenarios using enriched model parameters.
    
    Args:
        trace: PyMC trace
        dplus_per_km_array: Elevation per km from GPX
        fc_target: Target heart rate
        terrain_tech: Estimated terrain technicality (0=flat, 0.1=technical trail)
        pace_var: Estimated pace variability for the course
        terrain_inv: Estimated terrain inversion ratio
    """
    n_samples = 1000
    post = trace.posterior
    dist_total = len(dplus_per_km_array)
    
    chain_idx = np.random.randint(0, post.chain.size, n_samples)
    draw_idx = np.random.randint(0, post.draw.size, n_samples)
    
    # Original parameters
    b_base = post['pace_base'].values[chain_idx, draw_idx, 1]
    b_elev = post['effort_dplus'].values[chain_idx, draw_idx, 1]
    b_fatigue = post['beta_fatigue'].values[chain_idx, draw_idx]
    b_hr = post['beta_hr'].values[chain_idx, draw_idx]
    
    # NEW enriched parameters
    b_terrain = post['beta_terrain'].values[chain_idx, draw_idx]
    b_pace_var = post['beta_pace_var'].values[chain_idx, draw_idx]
    b_inversions = post['beta_inversions'].values[chain_idx, draw_idx]
    
    hr_diff = fc_target - 145
    ratio_dplus_per_km = dplus_per_km_array / 100
    
    km_range = np.arange(1, dist_total + 1)
    scenarios = np.zeros((n_samples, len(km_range)))
    
    for i in range(n_samples):
        allures = (b_base[i] + 
                   (b_elev[i] * ratio_dplus_per_km) + 
                   (b_fatigue[i] * km_range) + 
                   (b_hr[i] * hr_diff) +
                   (b_terrain[i] * terrain_tech) +
                   (b_pace_var[i] * pace_var) +
                   (b_inversions[i] * terrain_inv))
        scenarios[i, :] = np.cumsum(allures) / 60

    return km_range, scenarios

# Estimate terrain characteristics from GPX
gpx_terrain_tech = 0.08  # Trail technique estimé
gpx_pace_var = 0.20      # Variabilité attendue (trail)
gpx_terrain_inv = np.std(np.diff(dplus_per_km)) / 100  # Inversions from GPX

print(f"Terrain estimé: technicité={gpx_terrain_tech}, variabilité={gpx_pace_var}, inversions={gpx_terrain_inv:.3f}")

km, scenarios = simulate_pacing_tunnel(trace, dplus_per_km, FC_CIBLE, 
                                        terrain_tech=gpx_terrain_tech,
                                        pace_var=gpx_pace_var,
                                        terrain_inv=gpx_terrain_inv)

# ============================================================
# VISUALIZATION
# ============================================================

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

# Time formatting functions
def format_time(hours):
    """Convert decimal hours to HH:MM format"""
    h = int(hours)
    m = int((hours - h) * 60)
    return f"{h}:{m:02d}"

def format_time_full(hours):
    """Convert decimal hours to HH:MM:SS format"""
    h = int(hours)
    m = int((hours - h) * 60)
    s = int(((hours - h) * 60 - m) * 60)
    return f"{h}:{m:02d}:{s:02d}"

# Calculate percentiles
median_time = np.median(scenarios, axis=0)
lower_bound = np.percentile(scenarios, 5, axis=0)
upper_bound = np.percentile(scenarios, 95, axis=0)
very_low = np.percentile(scenarios, 1, axis=0)
very_high = np.percentile(scenarios, 99, axis=0)

# Premium style configuration
plt.style.use('dark_background')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['SF Pro Display', 'Helvetica Neue', 'Segoe UI', 'Arial'],
    'font.weight': 'normal',
    'axes.labelweight': 'normal',
    'axes.titleweight': 'bold',
})

# Create figure with gradient background effect
fig = plt.figure(figsize=(12, 12), facecolor='#0a0a0f')
ax = fig.add_axes([0.08, 0.12, 0.84, 0.72])  # Custom positioning for Instagram square
ax.set_facecolor('#0a0a0f')

# Create subtle background gradient effect
for i in range(100):
    alpha = 0.003 * (1 - i/100)
    ax.axhspan(ax.get_ylim()[0] if i > 0 else 0, 100, 
               facecolor='#1a1a2e', alpha=alpha)

# ═══════════════════════════════════════════════════════════════
# TUNNEL LAYERS - Premium gradient effect
# ═══════════════════════════════════════════════════════════════

# Outer glow layer (99% uncertainty)
for offset in [0.15, 0.10, 0.05]:
    ax.fill_between(km, very_low, very_high, 
                    color='#4338ca', alpha=offset * 0.5, linewidth=0)

# 99% tunnel
ax.fill_between(km, very_low, very_high, 
                color='#312e81', alpha=0.25, linewidth=0)

# 90% tunnel with gradient effect
ax.fill_between(km, lower_bound, upper_bound, 
                color='#4f46e5', alpha=0.35, linewidth=0)

# Inner glow
ax.fill_between(km, 
                lower_bound + (upper_bound - lower_bound) * 0.25,
                upper_bound - (upper_bound - lower_bound) * 0.25,
                color='#6366f1', alpha=0.25, linewidth=0)

# ═══════════════════════════════════════════════════════════════
# SCENARIO LINES - Neon style
# ═══════════════════════════════════════════════════════════════

# Fast scenario (green neon)
for lw, alpha in [(8, 0.1), (5, 0.2), (3, 0.4), (1.5, 1.0)]:
    ax.plot(km, lower_bound, color='#10b981', lw=lw, alpha=alpha, solid_capstyle='round')

# Slow scenario (amber neon)
for lw, alpha in [(8, 0.1), (5, 0.2), (3, 0.4), (1.5, 1.0)]:
    ax.plot(km, upper_bound, color='#f59e0b', lw=lw, alpha=alpha, solid_capstyle='round')

# ═══════════════════════════════════════════════════════════════
# MEDIAN LINE - Main focus with intense glow
# ═══════════════════════════════════════════════════════════════

# Outer glow layers
for lw, alpha in [(20, 0.03), (15, 0.05), (10, 0.1), (7, 0.15), (5, 0.3), (3, 0.5)]:
    ax.plot(km, median_time, color='#818cf8', lw=lw, alpha=alpha, solid_capstyle='round')

# Core line
ax.plot(km, median_time, color='#c7d2fe', lw=2.5, solid_capstyle='round')

# Bright center
ax.plot(km, median_time, color='#ffffff', lw=1, alpha=0.8, solid_capstyle='round')

# ═══════════════════════════════════════════════════════════════
# MARKERS - Key points
# ═══════════════════════════════════════════════════════════════

# Start point
ax.scatter([km[0]], [median_time[0]], s=150, c='#22c55e', zorder=10, edgecolors='white', linewidth=2)
ax.scatter([km[0]], [median_time[0]], s=400, c='#22c55e', alpha=0.2, zorder=9)

# End point
ax.scatter([km[-1]], [median_time[-1]], s=200, c='#ef4444', zorder=10, edgecolors='white', linewidth=2, marker='*')
ax.scatter([km[-1]], [median_time[-1]], s=600, c='#ef4444', alpha=0.2, zorder=9)

# Halfway marker
mid_idx = len(km) // 2
ax.scatter([km[mid_idx]], [median_time[mid_idx]], s=80, c='#a855f7', zorder=10, edgecolors='white', linewidth=1.5)

# ═══════════════════════════════════════════════════════════════
# GRID - Subtle and elegant
# ═══════════════════════════════════════════════════════════════

ax.grid(True, which='major', linestyle='-', alpha=0.08, color='#6366f1', linewidth=0.5)
ax.grid(True, which='minor', linestyle=':', alpha=0.04, color='#4f46e5', linewidth=0.3)
ax.minorticks_on()

# ═══════════════════════════════════════════════════════════════
# TITLES & LABELS - Modern typography
# ═══════════════════════════════════════════════════════════════

# Main title with emoji
fig.text(0.5, 0.92, 'PREDICTIVE PACING TUNNEL', 
         ha='center', fontsize=28, fontweight='bold', color='#f8fafc',
         fontfamily='sans-serif')

# Subtitle with race info
fig.text(0.5, 0.87, f'{DISTANCE_CIBLE} KM  •  {D_PLUS_CIBLE:.0f}M ELEVATION  •  HR {FC_CIBLE} BPM',
         ha='center', fontsize=13, color='#94a3b8', fontfamily='sans-serif',
         fontweight='light')

# Axis labels
ax.set_xlabel('DISTANCE (KM)', fontsize=11, color='#cbd5e1', labelpad=15, fontweight='medium')
ax.set_ylabel('RACE TIME', fontsize=11, color='#cbd5e1', labelpad=15, fontweight='medium')

# Format Y-axis as time
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: format_time(y)))

# Tick styling
ax.tick_params(axis='both', colors='#64748b', labelsize=10)

# ═══════════════════════════════════════════════════════════════
# LEGEND - Custom styled
# ═══════════════════════════════════════════════════════════════

legend_elements = [
    mpatches.Patch(facecolor='#c7d2fe', edgecolor='none', label='Target pace'),
    mpatches.Patch(facecolor='#10b981', edgecolor='none', label='Fast scenario (5%)'),
    mpatches.Patch(facecolor='#f59e0b', edgecolor='none', label='Slow scenario (95%)'),
    mpatches.Patch(facecolor='#4f46e5', alpha=0.5, edgecolor='none', label='90% confidence'),
]

legend = ax.legend(handles=legend_elements, loc='upper left', 
                   frameon=True, facecolor='#1e1b4b', edgecolor='#4338ca',
                   fontsize=9, labelcolor='#e2e8f0')
legend.get_frame().set_alpha(0.9)

# ═══════════════════════════════════════════════════════════════
# FINISH TIME BOX - Premium card style
# ═══════════════════════════════════════════════════════════════

# Card background
card_x = 0.68
card_y = 0.18
card = mpatches.FancyBboxPatch((card_x, card_y), 0.28, 0.18, 
                                boxstyle="round,pad=0.02,rounding_size=0.02",
                                facecolor='#1e1b4b', edgecolor='#6366f1',
                                linewidth=2, transform=fig.transFigure,
                                alpha=0.95, zorder=100)
fig.patches.append(card)

# Card content
fig.text(card_x + 0.14, card_y + 0.14, 'FINISH TIME', ha='center', fontsize=10, 
         color='#a5b4fc', fontweight='bold', zorder=101)
fig.text(card_x + 0.14, card_y + 0.085, format_time_full(median_time[-1]), ha='center', 
         fontsize=24, color='#ffffff', fontweight='bold', zorder=101)
fig.text(card_x + 0.14, card_y + 0.03, f'Range: {format_time(lower_bound[-1])} — {format_time(upper_bound[-1])}', 
         ha='center', fontsize=9, color='#94a3b8', zorder=101)

# ═══════════════════════════════════════════════════════════════
# BRANDING - Subtle watermark
# ═══════════════════════════════════════════════════════════════

fig.text(0.5, 0.03, 'Powered by Kairos One  •  @peakflow.tech', 
         ha='center', fontsize=9, color='#475569', fontweight='light',
         fontstyle='italic')

# ═══════════════════════════════════════════════════════════════
# SPINES - Clean edges
# ═══════════════════════════════════════════════════════════════

for spine in ax.spines.values():
    spine.set_color('#312e81')
    spine.set_linewidth(1)

# Set axis limits with padding
ax.set_xlim(0, DISTANCE_CIBLE + 1)
y_min = very_low.min() - 0.5
y_max = very_high.max() + 0.5
ax.set_ylim(y_min, y_max)

# ═══════════════════════════════════════════════════════════════
# SAVE & SHOW
# ═══════════════════════════════════════════════════════════════

plt.savefig('pacing_tunnel_instagram.png', dpi=300, facecolor='#0a0a0f', 
            edgecolor='none', bbox_inches='tight', pad_inches=0.3)
plt.show()

# Console summary
print(f"\n{'═'*60}")
print(f"📊 PACING SUMMARY")
print(f"{'═'*60}")
print(f"Distance: {DISTANCE_CIBLE} km | Elevation: {D_PLUS_CIBLE:.0f}m")
print(f"Predicted median time: {format_time_full(median_time[-1])}")
print(f"90% interval: {format_time_full(lower_bound[-1])} - {format_time_full(upper_bound[-1])}")
print(f"{'═'*60}")
print(f"\n✅ Saved: pacing_tunnel_instagram.png (300 DPI)")