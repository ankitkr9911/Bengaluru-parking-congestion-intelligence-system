# ============================================================================
# queueing_model.py — Phase 4: Dwell-Time Proxy + M/M/∞ + BPR Delay
# ============================================================================
# This module computes REAL congestion delay estimates, not arbitrary scores:
#
#   1. Dwell-time proxy: same vehicle detected multiple times at same location
#      on same day → estimate how long the vehicle stayed illegally parked
#
#   2. M/M/∞ queueing model: Poisson arrivals, exponential service time,
#      infinite servers. Output: E[N] = expected number of illegally parked
#      vehicles at any given time at each cell.
#
#   3. BPR volume-delay function (FIX #3): Standard transportation engineering
#      formula to convert E[N] into minutes-of-delay per vehicle per hour.
#      travel_time = free_flow_time × (1 + α × (V/C)^β)
#      where C is reduced by lanes blocked by illegally parked vehicles.
#
#   4. λ normalization by device count (FIX #2): prevents camera-density bias
# ============================================================================

# %%
import pandas as pd
import numpy as np
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    from pipeline.config import (
        OSM_ENRICHED_CSV, HOTSPOTS_CSV, QUEUEING_CSV, OUTPUT_DIR,
        BPR_ALPHA, BPR_BETA, ROAD_CAPACITY_PER_LANE, DEFAULT_LANES
    )
except ImportError:
    OSM_ENRICHED_CSV = "data/osm_enriched.csv"
    HOTSPOTS_CSV = "data/hotspots.csv"
    QUEUEING_CSV = "data/queueing_results.csv"
    OUTPUT_DIR = "data"
    BPR_ALPHA = 0.15
    BPR_BETA = 4.0
    ROAD_CAPACITY_PER_LANE = {"arterial": 1800, "collector": 1200, "local": 800}
    DEFAULT_LANES = {"arterial": 4, "collector": 2, "local": 2}


# ============================================================================
# STEP 1: Load data
# ============================================================================
# %%
print("=" * 70)
print("STEP 1: Loading data")
print("=" * 70)

df = pd.read_csv(OSM_ENRICHED_CSV)
df["created_datetime_ist"] = pd.to_datetime(df["created_datetime_ist"], format="mixed")
df["date"] = pd.to_datetime(df["date"])

cell_stats = pd.read_csv(HOTSPOTS_CSV)

print(f"  Violations: {len(df):,} rows")
print(f"  H3 cells: {len(cell_stats):,}")


# ============================================================================
# STEP 2: Dwell-time estimation from repeat detections
# ============================================================================
# Same vehicle, same H3 cell, same day → multiple detections = vehicle was
# parked there for the duration between first and last detection.
#
# Two signals:
#   a) Same-day dwell time (minutes) → genuine occupancy duration
#   b) Cross-day repeat count → chronic/habitual offender location
# ============================================================================
# %%
print("\n" + "=" * 70)
print("STEP 2: Estimating dwell times from repeat detections")
print("=" * 70)

# Sort by vehicle, cell, time
df_sorted = df.sort_values(["vehicle_number", "h3_cell", "created_datetime_ist"])

# --- (a) Same-day dwell time ---
# Group by vehicle × cell × date, get time span
same_day_groups = df_sorted.groupby(
    ["vehicle_number", "h3_cell", "date"]
)["created_datetime_ist"]

dwell_data = same_day_groups.agg(
    first_seen="min",
    last_seen="max",
    detection_count="count"
)
dwell_data = dwell_data[dwell_data["detection_count"] > 1].copy()  # Only repeats

# Dwell time in minutes
dwell_data["dwell_minutes"] = (
    (dwell_data["last_seen"] - dwell_data["first_seen"]).dt.total_seconds() / 60
)

# Filter out unrealistically long durations (>8 hours = likely different visits)
dwell_data = dwell_data[dwell_data["dwell_minutes"].between(5, 480)]

print(f"  Same-day repeat events: {len(dwell_data):,}")
print(f"  Dwell time stats (minutes):")
print(f"    Mean: {dwell_data['dwell_minutes'].mean():.1f}")
print(f"    Median: {dwell_data['dwell_minutes'].median():.1f}")
print(f"    Std: {dwell_data['dwell_minutes'].std():.1f}")
print(f"    25th percentile: {dwell_data['dwell_minutes'].quantile(0.25):.1f}")
print(f"    75th percentile: {dwell_data['dwell_minutes'].quantile(0.75):.1f}")

# Global mean dwell time (fallback for cells with too few samples)
GLOBAL_MEAN_DWELL = dwell_data["dwell_minutes"].mean()
print(f"  Global mean dwell time: {GLOBAL_MEAN_DWELL:.1f} minutes")

# Per-cell average dwell time
cell_dwell = (
    dwell_data.reset_index()
    .groupby("h3_cell")["dwell_minutes"]
    .agg(mean_dwell="mean", median_dwell="median", dwell_samples="count")
)

# --- (b) Cross-day chronic offender score ---
cross_day = (
    df_sorted.groupby(["vehicle_number", "h3_cell"])["date"]
    .nunique()
    .reset_index(name="days_seen")
)
chronic_offenders_per_cell = (
    cross_day[cross_day["days_seen"] >= 3]  # 3+ different days = chronic
    .groupby("h3_cell")
    .agg(chronic_offender_count=("vehicle_number", "count"),
         max_days_seen=("days_seen", "max"))
)

print(f"\n  Cross-day chronic offenders:")
print(f"    Cells with chronic offenders: {len(chronic_offenders_per_cell):,}")
print(f"    Total chronic offender-cell pairs: "
      f"{chronic_offenders_per_cell['chronic_offender_count'].sum():,}")


# ============================================================================
# STEP 3: M/M/∞ queueing model
# ============================================================================
# λ = arrival rate (violations per hour per cell, normalized by device count)
# μ = service rate (1 / mean_dwell_time in hours)
# E[N] = λ / μ = expected number of illegally parked vehicles at any time
# ============================================================================
# %%
print("\n" + "=" * 70)
print("STEP 3: Running M/M/∞ queueing model")
print("=" * 70)

# Calculate total observation hours from the dataset time span
time_span = df["created_datetime_ist"].max() - df["created_datetime_ist"].min()
total_hours = time_span.total_seconds() / 3600
print(f"  Dataset time span: {time_span.days} days ({total_hours:.0f} hours)")

# Merge dwell time and chronic offender data into cell_stats
cell_stats = cell_stats.merge(cell_dwell, on="h3_cell", how="left")
cell_stats = cell_stats.merge(chronic_offenders_per_cell, on="h3_cell", how="left")

# Fill missing dwell times with global mean
cell_stats["mean_dwell"] = cell_stats["mean_dwell"].fillna(GLOBAL_MEAN_DWELL)
cell_stats["dwell_samples"] = cell_stats["dwell_samples"].fillna(0)
cell_stats["chronic_offender_count"] = cell_stats["chronic_offender_count"].fillna(0)

# --- λ: arrival rate (device-normalized, FIX #2) ---
# violations_per_device is already computed in hotspot_detection.py
# Convert to per-hour rate
cell_stats["lambda_raw"] = cell_stats["violation_count"] / total_hours
cell_stats["lambda_normalized"] = cell_stats["violations_per_device"] / total_hours

# --- μ: service rate (1/mean_dwell_time in hours) ---
cell_stats["mu"] = 60 / cell_stats["mean_dwell"]  # Convert from minutes to per-hour

# --- E[N]: expected number of illegally parked vehicles at steady state ---
cell_stats["expected_n"] = cell_stats["lambda_normalized"] / cell_stats["mu"]

print(f"  Queueing model results:")
print(f"    Mean λ (normalized): {cell_stats['lambda_normalized'].mean():.4f} violations/hr")
print(f"    Mean μ: {cell_stats['mu'].mean():.4f} departures/hr")
print(f"    Mean E[N]: {cell_stats['expected_n'].mean():.3f} vehicles")
print(f"    Max E[N]: {cell_stats['expected_n'].max():.3f} vehicles")


# ============================================================================
# STEP 4: BPR volume-delay function (FIX #3)
# ============================================================================
# Standard Bureau of Public Roads formula:
#   travel_time = free_flow_time × (1 + α × (V/C)^β)
#
# Where:
#   V = current volume (assumed proportional to capacity for urban roads)
#   C = effective capacity = base_capacity - capacity_lost_to_parking
#   capacity_lost = E[N] × lane_blockage_fraction × capacity_per_lane
#
# Output: estimated additional delay (minutes per vehicle per km)
# ============================================================================
# %%
print("\n" + "=" * 70)
print("STEP 4: Computing BPR volume-delay estimates")
print("=" * 70)

# Assume urban roads operate at 70% of capacity during peak hours
VOLUME_TO_CAPACITY_RATIO = 0.70

# Free-flow travel time (minutes per km) by road type
FREE_FLOW_SPEED = {  # km/h
    "arterial":  40,
    "collector": 30,
    "local":     20,
}

def compute_bpr_delay(row):
    """
    Compute additional travel-time delay caused by illegal parking.
    
    Returns: additional minutes of delay per vehicle per km during peak hours.
    """
    road_type = row.get("road_type", "local")
    lanes = row.get("lanes", DEFAULT_LANES.get(road_type, 2))
    expected_n = row["expected_n"]
    
    # Capacity per direction (assume 2-directional road)
    lanes_per_direction = max(lanes / 2, 1)
    capacity_per_lane = ROAD_CAPACITY_PER_LANE.get(road_type, 800)
    base_capacity = lanes_per_direction * capacity_per_lane
    
    # Each illegally parked vehicle effectively blocks ~0.5 lanes on average
    # (ranges from 0.3 for scooter to 1.0 for bus/tanker)
    avg_lane_blockage = row.get("mean_vehicle_severity", 2.0) / 6.0  # Normalized 0-1
    lanes_blocked = expected_n * avg_lane_blockage
    
    # Effective capacity after illegal parking
    effective_lanes = max(lanes_per_direction - lanes_blocked, 0.5)
    effective_capacity = effective_lanes * capacity_per_lane
    
    # Current volume (at peak: assume 70% of original capacity)
    current_volume = base_capacity * VOLUME_TO_CAPACITY_RATIO
    
    # BPR function
    vc_ratio = current_volume / max(effective_capacity, 1)
    
    # Free-flow time (minutes per km)
    speed = FREE_FLOW_SPEED.get(road_type, 20)
    free_flow_time = 60 / speed  # minutes per km
    
    # Actual travel time with congestion
    actual_time = free_flow_time * (1 + BPR_ALPHA * (vc_ratio ** BPR_BETA))
    
    # Additional delay = actual - free_flow
    additional_delay = actual_time - free_flow_time
    
    return additional_delay

cell_stats["bpr_delay_min_per_km"] = cell_stats.apply(compute_bpr_delay, axis=1)

# Clip extreme values (some cells might have unrealistic numbers)
cell_stats["bpr_delay_min_per_km"] = cell_stats["bpr_delay_min_per_km"].clip(0, 30)

print(f"  BPR delay results (additional minutes per km during peak):")
print(f"    Mean: {cell_stats['bpr_delay_min_per_km'].mean():.2f}")
print(f"    Median: {cell_stats['bpr_delay_min_per_km'].median():.2f}")
print(f"    Max: {cell_stats['bpr_delay_min_per_km'].max():.2f}")

# Top 10 by delay
top_delay = cell_stats.nlargest(10, "bpr_delay_min_per_km")
print(f"\n  Top 10 cells by estimated delay:")
for _, row in top_delay.iterrows():
    print(f"    {row['police_station']:25s} | "
          f"E[N]={row['expected_n']:.2f} | "
          f"delay={row['bpr_delay_min_per_km']:.2f} min/km | "
          f"road={row.get('road_type', 'N/A')}")


# ============================================================================
# STEP 5: Save queueing results
# ============================================================================
# %%
print("\n" + "=" * 70)
print("STEP 5: Saving queueing model results")
print("=" * 70)

# Save updated cell stats
cell_stats.to_csv(HOTSPOTS_CSV, index=False)  # Update hotspots with queueing data
print(f"  Updated {HOTSPOTS_CSV} with queueing columns")

# Also save a focused queueing results file
queueing_cols = [
    "h3_cell", "police_station", "violation_count", "violations_per_device",
    "lambda_normalized", "mu", "expected_n", "mean_dwell", "dwell_samples",
    "chronic_offender_count", "bpr_delay_min_per_km",
    "hotspot_status", "gi_z_score"
]
existing_cols = [c for c in queueing_cols if c in cell_stats.columns]
cell_stats[existing_cols].to_csv(QUEUEING_CSV, index=False)
print(f"  Saved {QUEUEING_CSV}")


# ============================================================================
# SUMMARY
# ============================================================================
# %%
print("\n" + "=" * 70)
print("SUMMARY — Phase 4 Complete")
print("=" * 70)

confirmed = cell_stats[cell_stats["hotspot_status"].isin(["CONFIRMED_HIGH", "CONFIRMED"])]
if len(confirmed) > 0:
    print(f"""
  Dwell-time proxy:
    Same-day repeat events: {len(dwell_data):,}
    Global mean dwell: {GLOBAL_MEAN_DWELL:.1f} minutes

  M/M/∞ Queueing:
    Mean E[N] (all cells): {cell_stats['expected_n'].mean():.3f}
    Mean E[N] (confirmed hotspots): {confirmed['expected_n'].mean():.3f}

  BPR Delay (confirmed hotspots):
    Mean delay: {confirmed['bpr_delay_min_per_km'].mean():.2f} min/km
    Max delay: {confirmed['bpr_delay_min_per_km'].max():.2f} min/km

  Chronic offenders:
    Cells with chronic offenders: {(cell_stats['chronic_offender_count'] > 0).sum():,}
""")


if __name__ == "__main__":
    print("Phase 4 complete.")
