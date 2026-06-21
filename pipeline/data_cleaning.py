# ============================================================================
# data_cleaning.py — Phase 1: Data Cleaning & Feature Engineering
# ============================================================================
# This module handles:
#   1. Loading raw dataset
#   2. Filtering by validation_status (keep approved/processing/created1/NaN)
#   3. UTC → IST timezone conversion (FIX #1 — critical for peak-hour features)
#   4. Temporal feature extraction (hour, day, peak flags)
#   5. Exploding offence_code and violation_type lists
#   6. Deriving per-row severity weights
#   7. Saving cleaned output
#
# Run in Colab: Copy each # %% section as a separate cell
# Run locally:  python pipeline/data_cleaning.py
# ============================================================================

# %% [markdown]
# # Phase 1: Data Cleaning & Feature Engineering

# %%
import pandas as pd
import numpy as np
import ast
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Try importing config (works when run as module); fallback for Colab
try:
    from pipeline.config import (
        RAW_DATA_PATH, CLEANED_CSV, OUTPUT_DIR,
        SOURCE_TZ, TARGET_TZ,
        DROP_VALIDATION_STATUSES,
        VEHICLE_SEVERITY, VIOLATION_SEVERITY,
        MORNING_PEAK, EVENING_PEAK
    )
except ImportError:
    # --- Colab fallback: define constants inline ---
    RAW_DATA_PATH = "dataset.csv"
    OUTPUT_DIR = "data"
    CLEANED_CSV = "data/cleaned_violations.csv"
    SOURCE_TZ = "UTC"
    TARGET_TZ = "Asia/Kolkata"
    DROP_VALIDATION_STATUSES = ["rejected", "duplicate"]
    MORNING_PEAK = (8, 11)
    EVENING_PEAK = (17, 21)

    VEHICLE_SEVERITY = {
        "SCOOTER": 1.0, "MOPED": 1.0, "MOTOR CYCLE": 1.0,
        "PASSENGER AUTO": 1.5, "CAR": 2.0, "JEEP": 2.0,
        "VAN": 2.5, "MAXI-CAB": 3.0, "SCHOOL VEHICLE": 3.0,
        "LGV": 3.5, "TEMPO": 3.5, "GOODS AUTO": 3.0,
        "MINI LORRY": 4.0, "LORRY/GOODS VEHICLE": 4.5,
        "PRIVATE BUS": 5.0, "BUS (BMTC/KSRTC)": 5.0,
        "TOURIST BUS": 5.0, "FACTORY BUS": 5.0,
        "HGV": 5.0, "TANKER": 6.0, "TRACTOR": 4.0, "OTHERS": 2.0,
    }

    VIOLATION_SEVERITY = {
        "DOUBLE PARKING": 1.0, "PARKING IN A MAIN ROAD": 0.9,
        "PARKING NEAR ROAD CROSSING": 0.85,
        "PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS": 0.85,
        "PARKING OPPOSITE TO ANOTHER PARKED VEHICLE": 0.8,
        "PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC": 0.75,
        "WRONG PARKING": 0.6, "NO PARKING": 0.5,
        "PARKING ON FOOTPATH": 0.3, "PARKING OTHER THAN BUS STOP": 0.5,
        "H T V PROHIBITED": 0.7,
    }
    Path(OUTPUT_DIR).mkdir(exist_ok=True)


# ============================================================================
# STEP 1: Load raw data
# ============================================================================
# %%
print("=" * 70)
print("STEP 1: Loading raw dataset")
print("=" * 70)

df = pd.read_csv(RAW_DATA_PATH)
print(f"  Raw shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")


# ============================================================================
# STEP 2: Filter by validation_status
# ============================================================================
# Keep: approved, processing, created1, NaN (unvalidated ≠ confirmed bad)
# Drop: rejected (confirmed false positive), duplicate (redundant)
# ============================================================================
# %%
print("\n" + "=" * 70)
print("STEP 2: Filtering by validation_status")
print("=" * 70)

print(f"  Before filtering:")
print(f"    {df['validation_status'].value_counts(dropna=False).to_dict()}")

# Drop rows with explicitly bad statuses
mask_drop = df["validation_status"].isin(DROP_VALIDATION_STATUSES)
df = df[~mask_drop].copy()

print(f"\n  After filtering (dropped {mask_drop.sum()} rejected/duplicate rows):")
print(f"    Shape: {df.shape}")
print(f"    {df['validation_status'].value_counts(dropna=False).to_dict()}")


# ============================================================================
# STEP 3: UTC → IST timezone conversion (FIX #1)
# ============================================================================
# This is the FIRST transformation on timestamps. Every downstream feature
# (hour_of_day, is_peak_hour, day_of_week, shift assignment) depends on this.
#
# Without this fix: a violation at 8:30 PM IST is timestamped 15:00 UTC
# and would be misclassified as an off-peak afternoon event.
# ============================================================================
# %%
print("\n" + "=" * 70)
print("STEP 3: Converting UTC → IST (FIX #1)")
print("=" * 70)

# Parse with mixed format (some rows have microseconds, some don't)
df["created_datetime"] = pd.to_datetime(
    df["created_datetime"], format="mixed", utc=True
)

# Convert to IST
df["created_datetime_ist"] = df["created_datetime"].dt.tz_convert(TARGET_TZ)

# Show before/after
sample = df[["created_datetime", "created_datetime_ist"]].head(3)
print(f"  Sample conversion:")
for _, row in sample.iterrows():
    print(f"    UTC: {row['created_datetime']}  →  IST: {row['created_datetime_ist']}")


# ============================================================================
# STEP 4: Extract temporal features (from IST timestamps)
# ============================================================================
# %%
print("\n" + "=" * 70)
print("STEP 4: Extracting temporal features (from IST)")
print("=" * 70)

# Core time features — all derived from IST, not UTC
df["hour"] = df["created_datetime_ist"].dt.hour
df["day_of_week"] = df["created_datetime_ist"].dt.day_name()
df["day_num"] = df["created_datetime_ist"].dt.dayofweek  # 0=Monday
df["month"] = df["created_datetime_ist"].dt.month
df["date"] = df["created_datetime_ist"].dt.date
df["week"] = df["created_datetime_ist"].dt.isocalendar().week.astype(int)
df["year_week"] = (
    df["created_datetime_ist"].dt.year.astype(str) + "-W" +
    df["week"].astype(str).str.zfill(2)
)

# Peak hour flags
df["is_morning_peak"] = df["hour"].between(MORNING_PEAK[0], MORNING_PEAK[1] - 1)
df["is_evening_peak"] = df["hour"].between(EVENING_PEAK[0], EVENING_PEAK[1] - 1)
df["is_peak_hour"] = df["is_morning_peak"] | df["is_evening_peak"]
df["is_weekend"] = df["day_num"].isin([5, 6])  # Saturday=5, Sunday=6

# Time-of-day category for patrol shift recommendations
def get_shift(hour):
    if 6 <= hour < 14:
        return "morning_shift"
    elif 14 <= hour < 22:
        return "evening_shift"
    else:
        return "night_shift"

df["shift"] = df["hour"].apply(get_shift)

print(f"  Hour distribution (top 5):")
print(f"    {df['hour'].value_counts().nlargest(5).to_dict()}")
print(f"  Peak hour violations: {df['is_peak_hour'].sum()} "
      f"({df['is_peak_hour'].mean():.1%})")
print(f"  Weekend violations: {df['is_weekend'].sum()} "
      f"({df['is_weekend'].mean():.1%})")


# ============================================================================
# STEP 5: Parse & explode offence_code and violation_type
# ============================================================================
# offence_code is stored as a string like "[112,104]"
# violation_type is stored as '["WRONG PARKING","NO PARKING"]'
# We need to:
#   a) Parse them into actual lists
#   b) Count violations correctly (a row with 3 codes = 3 violations)
#   c) Derive severity from the most severe violation in each row
# ============================================================================
# %%
print("\n" + "=" * 70)
print("STEP 5: Parsing offence codes & violation types")
print("=" * 70)

def safe_parse_list(val):
    """Safely parse a string representation of a list."""
    if pd.isna(val):
        return []
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list):
            return parsed
        return [parsed]
    except (ValueError, SyntaxError):
        return []

# Parse into actual lists
df["offence_code_list"] = df["offence_code"].apply(safe_parse_list)
df["violation_type_list"] = df["violation_type"].apply(safe_parse_list)

# Count of distinct offence codes per row (severity signal)
df["num_offence_codes"] = df["offence_code_list"].apply(len)

# Get the most severe violation type per row
def get_max_violation_severity(violation_list):
    """Return the highest severity score from a list of violation types."""
    if not violation_list:
        return 0.3  # default low severity
    scores = [VIOLATION_SEVERITY.get(v.strip(), 0.3) for v in violation_list]
    return max(scores)

df["violation_severity_score"] = df["violation_type_list"].apply(
    get_max_violation_severity
)

# Create a flat string of violation types for easy filtering later
df["violation_types_flat"] = df["violation_type_list"].apply(
    lambda x: " | ".join(x) if x else "UNKNOWN"
)

print(f"  Multi-code violations: {(df['num_offence_codes'] > 1).sum()} "
      f"({(df['num_offence_codes'] > 1).mean():.1%})")
print(f"  Avg codes per violation: {df['num_offence_codes'].mean():.2f}")
print(f"  Violation severity distribution:")
print(f"    {df['violation_severity_score'].describe().to_dict()}")


# ============================================================================
# STEP 6: Vehicle severity scores
# ============================================================================
# A tanker blocking a lane has a very different impact than a scooter on the
# side of the road. We encode this domain knowledge as a severity weight.
# ============================================================================
# %%
print("\n" + "=" * 70)
print("STEP 6: Mapping vehicle severity scores")
print("=" * 70)

df["vehicle_severity"] = df["vehicle_type"].map(VEHICLE_SEVERITY).fillna(2.0)

# Combined row-level severity: vehicle impact × violation type impact × code count
# This represents the total "obstruction event severity" for a single detection
df["row_severity"] = (
    df["vehicle_severity"] *
    df["violation_severity_score"] *
    np.log1p(df["num_offence_codes"])  # log scaling for code count
)

print(f"  Vehicle type distribution (top 10):")
for vtype, count in df["vehicle_type"].value_counts().head(10).items():
    severity = VEHICLE_SEVERITY.get(vtype, 2.0)
    print(f"    {vtype}: {count} violations (severity={severity})")


# ============================================================================
# STEP 7: Clean location fields
# ============================================================================
# %%
print("\n" + "=" * 70)
print("STEP 7: Cleaning location fields")
print("=" * 70)

# Handle missing police station (only 5 rows)
df["police_station"] = df["police_station"].fillna("UNKNOWN")

# Handle missing junction_name
df["junction_name"] = df["junction_name"].fillna("No Junction")

# Flag: is this at a named junction?
df["has_junction"] = df["junction_name"] != "No Junction"

# Extract junction code (e.g., "BTP051" from "BTP051 - Safina Plaza Junction")
df["junction_code"] = df["junction_name"].apply(
    lambda x: x.split(" - ")[0].strip() if " - " in str(x) else None
)

# Handle missing center_code
df["center_code"] = df["center_code"].fillna(-1).astype(int)

print(f"  Rows with named junctions: {df['has_junction'].sum()} "
      f"({df['has_junction'].mean():.1%})")
print(f"  Unique police stations: {df['police_station'].nunique()}")
print(f"  Unique junction names: {df['junction_name'].nunique()}")


# ============================================================================
# STEP 8: Drop fully null columns & select final columns
# ============================================================================
# %%
print("\n" + "=" * 70)
print("STEP 8: Finalizing cleaned dataset")
print("=" * 70)

# Drop columns that are 100% null or not needed
drop_cols = [
    "description",           # 100% null
    "closed_datetime",       # 100% null
    "action_taken_timestamp" # 100% null
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Final column list (ordered logically)
final_columns = [
    # Identifiers
    "id", "vehicle_number", "vehicle_type", "device_id", "created_by_id",
    # Location
    "latitude", "longitude", "location", "police_station", "center_code",
    "junction_name", "has_junction", "junction_code",
    # Time (IST)
    "created_datetime", "created_datetime_ist",
    "hour", "day_of_week", "day_num", "month", "date", "week", "year_week",
    "is_morning_peak", "is_evening_peak", "is_peak_hour", "is_weekend", "shift",
    # Violation details
    "violation_type", "offence_code",
    "offence_code_list", "violation_type_list", "violation_types_flat",
    "num_offence_codes", "validation_status",
    # Severity scores
    "vehicle_severity", "violation_severity_score", "row_severity",
    # Metadata
    "data_sent_to_scita", "modified_datetime",
]

# Keep only columns that exist (some optional ones may not)
final_columns = [c for c in final_columns if c in df.columns]
df = df[final_columns]

print(f"  Final shape: {df.shape}")
print(f"  Date range (IST): {df['created_datetime_ist'].min()} → "
      f"{df['created_datetime_ist'].max()}")
print(f"  Unique vehicles: {df['vehicle_number'].nunique()}")
print(f"  Unique devices: {df['device_id'].nunique()}")


# ============================================================================
# STEP 9: Save cleaned data
# ============================================================================
# %%
print("\n" + "=" * 70)
print("STEP 9: Saving cleaned dataset")
print("=" * 70)

Path(OUTPUT_DIR).mkdir(exist_ok=True)
df.to_csv(CLEANED_CSV, index=False)
print(f"  Saved to: {CLEANED_CSV}")
print(f"  Size: {Path(CLEANED_CSV).stat().st_size / 1e6:.1f} MB")


# ============================================================================
# STEP 10: Summary statistics
# ============================================================================
# %%
print("\n" + "=" * 70)
print("SUMMARY — Phase 1 Complete")
print("=" * 70)
print(f"""
  Rows:              {len(df):,}
  Date range (IST):  {df['created_datetime_ist'].min().strftime('%Y-%m-%d')} → {df['created_datetime_ist'].max().strftime('%Y-%m-%d')}
  Unique vehicles:   {df['vehicle_number'].nunique():,}
  Unique devices:    {df['device_id'].nunique():,}
  Police stations:   {df['police_station'].nunique()}
  Named junctions:   {df['has_junction'].sum():,} ({df['has_junction'].mean():.1%})
  Peak-hour violations: {df['is_peak_hour'].sum():,} ({df['is_peak_hour'].mean():.1%})
  Multi-code violations: {(df['num_offence_codes'] > 1).sum():,} ({(df['num_offence_codes'] > 1).mean():.1%})
  Avg vehicle severity: {df['vehicle_severity'].mean():.2f}
  Avg violation severity: {df['violation_severity_score'].mean():.2f}
""")


# ============================================================================
# Allow running as standalone script
# ============================================================================
if __name__ == "__main__":
    print("Phase 1 complete. Output saved to:", CLEANED_CSV)
