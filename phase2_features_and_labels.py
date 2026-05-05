"""
Phase 2: Feature Engineering & Labeling
Hurricane RI Prediction — Quantum Feature Engineering Pipeline

Inputs:
  data/hurdat2_regional_6hr.csv       (Phase 1 output — ArcGIS CSV format)
  data/al_ships_1982_2023.txt         (SHIPS developmental data, optional)
      → Download from:
        https://rammb2.cira.colostate.edu/research/tropical-cyclones/ships/development_data/

Outputs:
  data/labeled_dataset.csv            (RI labels + raw features, pre-normalization)
  data/normalized_dataset.csv         (features normalized to [0, π] for angle encoding)
  data/feature_scaler.joblib          (fitted scaler for reuse at inference time)

RI Label Rules:
  ri_label = 1   : wind_kt >= 35 at t0 AND wind increases >= 30 kt in next 24 hours
  ri_label = 0   : wind_kt >= 35 at t0 AND wind increases < 30 kt in next 24 hours
  ri_label = NaN : wind_kt < 35 at t0 (pre-TS, genesis stage — not RI-applicable)
               OR : final 4 obs of storm (no complete 24-hour future window)

SHIPS variables extracted (t=0 only):
  RSST → sst_c        Sea surface temperature        (raw/10 → °C)
  SHRD → shear_ms     850-200 hPa wind shear         (raw/10 kt → m/s via *0.514)
  OHC  → ohc_kjcm2    Ocean heat content             (kJ/cm²)
  RHMD → rh_pct       500-300 hPa relative humidity  (%)
  VVAV → vort_s1      850 hPa vorticity              (raw * 1e-5 s⁻¹)
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import joblib
import os
import re
import warnings
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

T0_INDEX = 2  # t=0 is the 3rd column on each predictor line (-12, -6, 0, ...)

# ── Config ─────────────────────────────────────────────────────────────────────
HURDAT_CSV     = "data/hurdat2_regional_6hr.csv"
SHIPS_FILE = "data/lsdiaga_1982_2023_sat_ts_7day.txt"
OUT_LABELED    = "data/labeled_dataset.csv"
OUT_NORMALIZED = "data/normalized_dataset.csv"
OUT_SCALER     = "data/feature_scaler.joblib"

RI_THRESHOLD_KT  = 30     # knot increase in 24 h required for RI label
TS_THRESHOLD_KT  = 35     # minimum wind to be RI-applicable (tropical storm strength)
STEPS_24H        = 4      # 4 × 6-hr steps = 24 hours

# The 6 core features fed into the quantum circuit (angle encoding)
QUANTUM_FEATURES = ['wind_kt', 'pressure_mb', 'sst_c', 'shear_ms', 'ohc_kjcm2', 'rh_pct']

os.makedirs("data", exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Compute RI Labels
# ══════════════════════════════════════════════════════════════════════════════

def compute_ri_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes RI labels strictly within each storm's track.

    Key guard: observations where wind_kt < TS_THRESHOLD_KT (35 kt) receive
    ri_label = NaN — these are pre-tropical-storm observations included in the
    full track but not applicable for RI labeling (that would be genesis, not RI).

    Steps:
      1. Sort by storm then time
      2. Shift wind_kt forward 4 steps (24 h) within each storm
      3. Compute delta
      4. Apply RI threshold only to TS-strength-or-above observations
    """
    df = df.sort_values(['storm_id', 'time']).copy()

    # Wind 24 hours ahead — shift within storm only (no cross-storm contamination)
    df['wind_future_24h'] = (
        df.groupby('storm_id')['wind_kt']
          .shift(-STEPS_24H)
    )
    df['wind_delta_24h'] = df['wind_future_24h'] - df['wind_kt']

    # Three-way assignment:
    #   NaN  → below TS threshold OR no future window
    #   1    → RI event
    #   0    → no RI
    conditions = [
        df['wind_kt'] < TS_THRESHOLD_KT,           # pre-TS: not applicable
        df['wind_delta_24h'].isna(),                # no future 24h window
        df['wind_delta_24h'] >= RI_THRESHOLD_KT,   # RI event
    ]
    choices = [np.nan, np.nan, 1.0]
    df['ri_label'] = np.select(conditions, choices, default=0.0)

    # Drop helper column — not needed downstream
    df = df.drop(columns=['wind_future_24h'])
    return df


def print_label_summary(df: pd.DataFrame):
    labeled = df[df['ri_label'].notna()]
    ri_pos = (labeled['ri_label'] == 1).sum()
    ri_neg = (labeled['ri_label'] == 0).sum()
    unlabeled_genesis = (df['wind_kt'] < TS_THRESHOLD_KT).sum()
    unlabeled_eostorm = df['ri_label'].isna().sum() - unlabeled_genesis

    print(f"  Total observations:              {len(df):,}")
    print(f"  Labelable (wind ≥ {TS_THRESHOLD_KT} kt):        {len(labeled):,}")
    print(f"    RI=1 (rapid intensification):  {ri_pos:,}  ({ri_pos/len(labeled)*100:.1f}%)")
    print(f"    RI=0 (no RI):                  {ri_neg:,}  ({ri_neg/len(labeled)*100:.1f}%)")
    print(f"    Class imbalance ratio:         1 : {ri_neg/ri_pos:.1f}")
    print(f"  Unlabeled — pre-TS genesis:      {unlabeled_genesis:,}")
    print(f"  Unlabeled — end-of-storm:        {unlabeled_eostorm:,}")

    # RI rate per decade
    df_l = df[df['ri_label'].notna()].copy()
    df_l['decade'] = (df_l['year'] // 10 * 10).astype(str) + 's'
    print(f"\n  RI rate by decade:")
    decade_stats = (
        df_l.groupby('decade')['ri_label']
            .agg(['sum', 'count'])
            .rename(columns={'sum': 'ri_events', 'count': 'total'})
    )
    decade_stats['ri_rate_pct'] = (decade_stats['ri_events'] / decade_stats['total'] * 100).round(1)
    print(decade_stats.to_string())


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4A — Parse SHIPS Developmental Data
# ══════════════════════════════════════════════════════════════════════════════

# Replace with:
SHIPS_VARS = {
    'CSST': ('sst_c',      0.1),          # CSST = current SST (tenths °C → °C)
    'SHRD': ('shear_ms',   0.1 * 0.514),  # unchanged — already working
    'COHC': ('ohc_kjcm2',  1.0),          # COHC = column ocean heat content (kJ/cm²)
    'RHMD': ('rh_pct',     1.0),          # unchanged — already working
    'VVAV': ('vort_s1',    1e-5),         # unchanged
}

HEADER_RE = re.compile(
    r'^([A-Z]{2}\d{6})\s+(\d{10})\s+([-\d]+)\s+([-\d]+)\s+(\d+)'
)

def parse_ships(filepath: str) -> pd.DataFrame:
    """
    Parse the SHIPS Atlantic 7-day developmental data file.

    HEAD line format:
      SNAME YYMMDD HH VMAX LAT LON MSLP ATCF_ID ... HEAD
      e.g.: ALBE 820602 12 20 21.7 87.1 1005 AL011982 ... HEAD

    Predictor line format:
      val(-12) val(-6) val(0) val(+6) ... val(+168) TAG
      t=0 value is always at column index T0_INDEX=2.
      Missing values are coded as 9999.
    """
    records = []
    current_header = None
    current_predictors = {}

    def flush_record():
        if current_header is None:
            return
        row = {**current_header}
        for tag, (col, scale) in SHIPS_VARS.items():
            raw = current_predictors.get(tag)
            row[col] = raw * scale if raw is not None else np.nan
        records.append(row)

    with open(filepath, 'r', encoding='utf-8-sig', errors='replace') as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            tag = parts[-1]

            if tag == 'HEAD':
                flush_record()
                current_predictors = {}
                try:
                    # ATCF ID: 2 uppercase letters + 6 digits (e.g. AL011982)
                    atcf_id = next(
                        (p for p in parts if re.match(r'^[A-Z]{2}\d{6}$', p)),
                        None
                    )
                    if atcf_id is None:
                        current_header = None
                        continue

                    # Date: parts[1]=YYMMDD, parts[2]=HH
                    yymmdd = parts[1]
                    hh     = int(parts[2])
                    yy     = int(yymmdd[:2])
                    mm     = int(yymmdd[2:4])
                    dd     = int(yymmdd[4:6])
                    yyyy   = 1900 + yy if yy >= 82 else 2000 + yy

                    current_header = {
                        'storm_id': atcf_id,
                        'time':     pd.Timestamp(yyyy, mm, dd, hh, tz='UTC'),
                    }
                except Exception:
                    current_header = None

            elif tag in SHIPS_VARS and current_header is not None:
                values = parts[:-1]   # strip the tag
                if len(values) > T0_INDEX:
                    try:
                        val = int(values[T0_INDEX])
                        current_predictors[tag] = val if val != 9999 else None
                    except ValueError:
                        pass

    flush_record()   # flush the final record

    if not records:
        return pd.DataFrame()

    ships_df = pd.DataFrame(records)
    ships_df['time'] = pd.to_datetime(ships_df['time'], utc=True)
    print(f"  SHIPS cases loaded:  {len(ships_df):,}")
    print(f"  SHIPS year range:    "
          f"{ships_df['time'].dt.year.min()}–{ships_df['time'].dt.year.max()}")
    env_cols = [col for _, (col, _) in SHIPS_VARS.items()]
    for col in env_cols:
        pct = ships_df[col].isna().mean() * 100
        print(f"  Missing {col:12s}: {pct:.1f}%")
    return ships_df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4B — Merge SHIPS into HURDAT2
# ══════════════════════════════════════════════════════════════════════════════

def merge_ships(hurdat_df: pd.DataFrame, ships_df: pd.DataFrame) -> pd.DataFrame:
    hurdat_df['time'] = pd.to_datetime(hurdat_df['time'], utc=True)
    ships_df['time']  = pd.to_datetime(ships_df['time'],  utc=True)
    env_cols = [col for _, (col, _) in SHIPS_VARS.items()]
    merged = hurdat_df.merge(
        ships_df[['storm_id', 'time'] + env_cols],
        on=['storm_id', 'time'],
        how='left'
    )
    # Replace with:
    first_env_col = next((col for _, (col, _) in SHIPS_VARS.items()
                          if merged[col].notna().any()), None)
    matched_pct = merged[first_env_col].notna().mean() * 100 if first_env_col else 0.0
    print(f"  HURDAT2 obs with SHIPS match: {matched_pct:.1f}%")
    return merged


def add_empty_ships_columns(df: pd.DataFrame) -> pd.DataFrame:
    for _, (col, _) in SHIPS_VARS.items():
        df[col] = np.nan
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Impute Missing Values
# ══════════════════════════════════════════════════════════════════════════════

def impute_features(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Median imputation per feature.
    The >2 missing guard only counts features that are ACTUALLY AVAILABLE
    (i.e., not entirely NaN across the dataset — e.g., SHIPS columns when
    SHIPS data hasn't been loaded yet). This prevents dropping all rows
    when an entire data source is absent.
    """
    available = [f for f in features if f in df.columns]

    # Determine which features have ANY real data (not all-NaN)
    active_features = [f for f in available if df[f].notna().any()]
    inactive_features = [f for f in available if f not in active_features]

    if inactive_features:
        print(f"  Features with no data (entire source missing): {inactive_features}")
        print(f"  Drop guard applied only to active features: {active_features}")

    # Count missing only across active features on labelable rows
    labelable = df['ri_label'].notna()
    if active_features:
        missing_count = df.loc[labelable, active_features].isna().sum(axis=1)
        drop_idx = missing_count[missing_count > 2].index
        before = len(df)
        df = df.drop(index=drop_idx).copy()
        print(f"  Dropped {before - len(df):,} labelable rows (>2 active features missing)")
    else:
        print("  No active features to evaluate — skipping drop guard")

    # Median impute remaining NaNs (fit only on labelable rows)
    for col in active_features:
        median_val = df.loc[df['ri_label'].notna(), col].median()
        n_imputed  = df[col].isna().sum()
        if n_imputed > 0:
            df[col] = df[col].fillna(median_val)
            print(f"  Imputed {n_imputed:,} values in '{col}' (median = {median_val:.2f})")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Normalize Features to [0, π] for Angle Encoding
# ══════════════════════════════════════════════════════════════════════════════

def normalize_for_quantum(df: pd.DataFrame, features: list):
    """
    Scale each feature to [0, π] via MinMaxScaler.
    Fitted ONLY on RI-labelable rows so unlabeled genesis observations
    don't distort the scaler's min/max.

    Returns df_norm with '_norm' suffix columns and the fitted scaler.
    Save the scaler — must be applied identically at inference time.
    """
    available  = [f for f in features if f in df.columns]
    fit_rows   = df[df['ri_label'].notna()].copy()

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    scaler.fit(fit_rows[available])

    # Apply to the entire dataframe (including unlabeled rows for completeness)
    norm_values = scaler.transform(df[available])
    norm_cols   = [f + '_norm' for f in available]
    df_out      = df.copy()
    df_out[norm_cols] = norm_values

    print(f"  Normalized {len(available)} features → [0, π]")
    for orig, norm_col in zip(available, norm_cols):
        mn = df_out.loc[df_out['ri_label'].notna(), norm_col].min()
        mx = df_out.loc[df_out['ri_label'].notna(), norm_col].max()
        print(f"    {orig:15s} → {norm_col}  [{mn:.4f}, {mx:.4f}]")

    return df_out, scaler


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Load Phase 1 output ───────────────────────────────────────────────────
    print("── Step 3: Loading HURDAT2 regional time series ─────────────────────")
    df = pd.read_csv(HURDAT_CSV)
    df['time'] = pd.to_datetime(df['time'], utc=True)
    print(f"  Loaded {len(df):,} observations across {df['storm_id'].nunique()} storms")

    # ── Compute RI labels ─────────────────────────────────────────────────────
    df = compute_ri_labels(df)
    print_label_summary(df)
    print()

    # ── Load & merge SHIPS ────────────────────────────────────────────────────
    print("── Step 4: Loading SHIPS environmental variables ────────────────────")
    if os.path.exists(SHIPS_FILE):
        print(f"  SHIPS file found: {SHIPS_FILE}")
        ships_df = parse_ships(SHIPS_FILE)
        df = merge_ships(df, ships_df) if not ships_df.empty else add_empty_ships_columns(df)
    else:
        print(f"  WARNING: SHIPS file not found at '{SHIPS_FILE}'")
        print(f"  Download from:")
        print(f"    https://rammb2.cira.colostate.edu/research/tropical-cyclones/ships/development_data/")
        print(f"  Continuing with NaN for sst_c, shear_ms, ohc_kjcm2, rh_pct, vort_s1.")
        print(f"  wind_kt and pressure_mb (from HURDAT2) are fully available.")
        df = add_empty_ships_columns(df)
    print()

    # ── Impute missing values ─────────────────────────────────────────────────
    print("── Step 5: Imputing missing feature values ──────────────────────────")
    df = impute_features(df, QUANTUM_FEATURES)
    print()

    # ── Save labeled (pre-normalization) dataset ──────────────────────────────
    df.to_csv(OUT_LABELED, index=False)
    print(f"  Labeled dataset saved → {OUT_LABELED}  ({df.shape[0]:,} rows × {df.shape[1]} cols)")
    print()

    # ── Normalize features ────────────────────────────────────────────────────
    print("── Step 6: Normalizing features to [0, π] for angle encoding ────────")
    # Only normalize features that have actual data
    active_quantum_features = [f for f in QUANTUM_FEATURES if df[f].notna().any()]
    if len(active_quantum_features) < len(QUANTUM_FEATURES):
        missing_src = [f for f in QUANTUM_FEATURES if f not in active_quantum_features]
        print(f"  Note: {missing_src} are all NaN (SHIPS not loaded) — "
              f"normalizing {len(active_quantum_features)} available features only.")
    df_norm, scaler = normalize_for_quantum(df, active_quantum_features)
    print()

    # ── Save outputs ──────────────────────────────────────────────────────────
    df_norm.to_csv(OUT_NORMALIZED, index=False)
    joblib.dump(scaler, OUT_SCALER)

    print(f"── Outputs ───────────────────────────────────────────────────────────")
    print(f"  Labeled dataset    → {OUT_LABELED}")
    print(f"  Normalized dataset → {OUT_NORMALIZED}  ({df_norm.shape[0]:,} rows × {df_norm.shape[1]} cols)")
    print(f"  Feature scaler     → {OUT_SCALER}")

    # ── Final summary (labelable rows only) ──────────────────────────────────
    print(f"\n── Final Labelable Dataset ───────────────────────────────────────────")
    labelable = df_norm[df_norm['ri_label'].notna()]
    norm_cols = [c for c in df_norm.columns if c.endswith('_norm')]
    print(f"  Rows available for model training:  {len(labelable):,}")
    print(f"  RI positive (ri_label=1):           {(labelable['ri_label']==1).sum():,}")
    print(f"  RI negative (ri_label=0):           {(labelable['ri_label']==0).sum():,}")
    print(f"\n  Normalized features ready for quantum encoding:")
    for c in norm_cols:
        print(f"    {c}")
    print(f"\n  Sample (5 labeled rows):")
    display_cols = ['storm_id', 'storm_name', 'time', 'wind_kt',
                    'wind_delta_24h', 'ri_label'] + norm_cols
    display_cols = [c for c in display_cols if c in df_norm.columns]
    print(labelable[display_cols].head().to_string())