"""
Phase 1: Data Acquisition & Setup (ArcGIS CSV version)
Hurricane RI Prediction — Quantum Feature Engineering Pipeline

Input:  hurdat2_arcgis.csv  (downloaded from ArcGIS, place in data/ folder)
Output: data/hurdat2_regional_6hr.csv
"""

import numpy as np
import pandas as pd
import os
from math import radians, sin, cos, sqrt, atan2

# ── Constants ──────────────────────────────────────────────────────────────────
INPUT_CSV      = "data/atlantic_hurricane_tracks.csv"
OUTPUT_CSV     = "data/hurdat2_regional_6hr.csv"

NORFOLK_LAT    = 36.85
NORFOLK_LON    = -76.29
YEAR_START     = 1980
YEAR_END       = 2025
MIN_WIND_KT    = 35
DIST_THRESH_DEG = 5.0

os.makedirs("data", exist_ok=True)

# ── Helper: Haversine distance in degrees ──────────────────────────────────────
def haversine_deg(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = (sin(dlat / 2) ** 2
         + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2)
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return (R * c) / 111.0

def dist_from_norfolk(lat, lon):
    return haversine_deg(lat, lon, NORFOLK_LAT, NORFOLK_LON)

# ── Step 1: Load ArcGIS CSV ────────────────────────────────────────────────────
print("── Step 1: Loading ArcGIS HURDAT2 CSV ───────────────────────────────")
df = pd.read_csv(INPUT_CSV, low_memory=False)

print(f"  Raw rows loaded:    {len(df):,}")
print(f"  Columns:            {list(df.columns)}")

# ── Step 2: Standardize column names ──────────────────────────────────────────
print("\n── Step 2: Standardizing columns ────────────────────────────────────")
df = df.rename(columns={
    'name':       'storm_name',
    'datetime':   'time',
    'latitude':   'lat',
    'longitude':  'lon',
    'status':     'storm_type',
})

# Parse datetime
df['time'] = pd.to_datetime(df['time'], utc=True)

# Coerce pressure to numeric — empty strings become NaN automatically
df['pressure_mb'] = pd.to_numeric(df['pressure_mb'], errors='coerce')
df['wind_kt']     = pd.to_numeric(df['wind_kt'],     errors='coerce')
df['lat']         = pd.to_numeric(df['lat'],          errors='coerce')
df['lon']         = pd.to_numeric(df['lon'],          errors='coerce')

print(f"  Parsed {len(df):,} observations across {df['storm_id'].nunique():,} storms")

# ── Step 3: Year filter ────────────────────────────────────────────────────────
print(f"\n── Step 3: Applying filters ─────────────────────────────────────────")
df = df[df['year'].between(YEAR_START, YEAR_END)].copy()
print(f"  After year filter ({YEAR_START}–{YEAR_END}): "
      f"{df['storm_id'].nunique()} storms, {len(df):,} obs")

# ── Step 4: Minimum wind filter (storm must reach TS strength at some point) ──
qualifying_ids = (
    df.groupby('storm_id')['wind_kt']
      .max()
      .ge(MIN_WIND_KT)
      .pipe(lambda s: s[s].index)
)
df = df[df['storm_id'].isin(qualifying_ids)].copy()
print(f"  After wind ≥ {MIN_WIND_KT} kt filter:    "
      f"{df['storm_id'].nunique()} storms, {len(df):,} obs")

# ── Step 5: Regional filter (within DIST_THRESH_DEG of Norfolk) ───────────────
df['dist_norfolk_deg'] = df.apply(
    lambda r: dist_from_norfolk(r['lat'], r['lon']), axis=1
)
min_dist_per_storm = df.groupby('storm_id')['dist_norfolk_deg'].min()
regional_ids = min_dist_per_storm[min_dist_per_storm <= DIST_THRESH_DEG].index
df = df[df['storm_id'].isin(regional_ids)].copy()
print(f"  After regional filter (≤ {DIST_THRESH_DEG}° of Norfolk): "
      f"{df['storm_id'].nunique()} storms, {len(df):,} obs")

# ── Step 6: Diagnostics ───────────────────────────────────────────────────────
print(f"\n── Dataset Summary ───────────────────────────────────────────────────")
print(f"  Year range:              {df['year'].min()}–{df['year'].max()}")
print(f"  Total storms:            {df['storm_id'].nunique()}")
print(f"  Total 6-hr observations: {len(df):,}")
print(f"  Avg obs per storm:       {len(df) / df['storm_id'].nunique():.1f}")
print(f"  Missing pressure:        {df['pressure_mb'].isna().mean()*100:.1f}%")
print(f"  Missing wind:            {df['wind_kt'].isna().mean()*100:.1f}%")

print(f"\n  Storm type distribution:")
print(df['storm_type'].value_counts().to_string())

print(f"\n  Category distribution:")
if 'category' in df.columns:
    print(df['category'].value_counts().to_string())

print(f"\n  Storms per decade:")
df['decade'] = (df['year'] // 10 * 10).astype(str) + 's'
print(df.groupby('decade')['storm_id'].nunique().to_string())

# Check key Hampton Roads landmark storms are present
landmark_storms = ['ISABEL', 'IRENE', 'MATTHEW', 'DORIAN', 'FLOYD', 'HUGO']
found = df[df['storm_name'].isin(landmark_storms)]['storm_name'].unique()
print(f"\n  Key Hampton Roads storms present: {sorted(found)}")

# Landfall events near region (bonus from ArcGIS data)
if 'is_landfall' in df.columns:
    landfall_count = df[df['is_landfall'] == 1]['storm_id'].nunique()
    print(f"  Storms with landfall events in dataset: {landfall_count}")

# ── Step 7: Select final columns and save ─────────────────────────────────────
keep_cols = [
    'storm_id', 'storm_name', 'year', 'time',
    'storm_type', 'lat', 'lon',
    'wind_kt', 'pressure_mb', 'dist_norfolk_deg',
    # Bonus columns from ArcGIS — useful for Phase 6 case studies
    'category', 'is_landfall', 'landfall_state'
]
keep_cols = [c for c in keep_cols if c in df.columns]
df = df[keep_cols].sort_values(['storm_id', 'time']).reset_index(drop=True)

df.to_csv(OUTPUT_CSV, index=False)
print(f"\n  Output saved → {OUTPUT_CSV}")
print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\n  Sample (first 5 rows):")
print(df.head().to_string())