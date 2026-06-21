# ============================================================================
# prioritization.py — Phase 7: Station-Level Patrol Prioritization
# ============================================================================
# Maps each hotspot to its police station (54 stations) and generates
# actionable patrol recommendations:
#   - Top-N zones per station ranked by CIS × predicted_risk
#   - Recommended patrol shift based on peak violation hours
#   - Plain-language reasoning for each recommendation
# ============================================================================

# %%
import pandas as pd
import numpy as np
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    from pipeline.config import (
        HOTSPOTS_CSV, FORECASTS_CSV, PATROL_CSV, IMPACT_SCORES_CSV,
        OUTPUT_DIR, TOP_N_ZONES_PER_STATION, OSM_ENRICHED_CSV
    )
except ImportError:
    HOTSPOTS_CSV = "data/hotspots.csv"
    FORECASTS_CSV = "data/forecasts.csv"
    PATROL_CSV = "data/patrol_schedule.csv"
    IMPACT_SCORES_CSV = "data/impact_scores.csv"
    OSM_ENRICHED_CSV = "data/osm_enriched.csv"
    OUTPUT_DIR = "data"
    TOP_N_ZONES_PER_STATION = 5


# ============================================================================
# STEP 1: Load data
# ============================================================================
# %%
print("=" * 70)
print("STEP 1: Loading hotspot and forecast data")
print("=" * 70)

cell_stats = pd.read_csv(HOTSPOTS_CSV)
forecasts = pd.read_csv(FORECASTS_CSV)

# Merge forecast risk into cell_stats
cell_stats = cell_stats.merge(
    forecasts[["h3_cell", "predicted_risk", "risk_category",
               "trend_direction", "predicted_violations_next_week"]],
    on="h3_cell", how="left"
)

# Fill missing risk
cell_stats["predicted_risk"] = cell_stats["predicted_risk"].fillna(
    cell_stats["CIS"] * 0.5  # Conservative estimate for unforecasted cells
)

print(f"  Loaded {len(cell_stats):,} cells")
print(f"  Unique police stations: {cell_stats['police_station'].nunique()}")


# ============================================================================
# STEP 2: Compute priority score per cell
# ============================================================================
# %%
print("\n" + "=" * 70)
print("STEP 2: Computing priority scores")
print("=" * 70)

# Priority = CIS × predicted_risk / 100 (both are 0-100+ scale)
cell_stats["priority_score"] = (
    cell_stats["CIS"] * cell_stats["predicted_risk"] / 100
)

print(f"  Priority score distribution:")
print(f"    Mean: {cell_stats['priority_score'].mean():.1f}")
print(f"    Max: {cell_stats['priority_score'].max():.1f}")


# ============================================================================
# STEP 3: Determine recommended patrol shift per cell
# ============================================================================
# %%
print("\n" + "=" * 70)
print("STEP 3: Determining recommended patrol shifts")
print("=" * 70)

# Load violation data to compute per-cell shift distribution
df = pd.read_csv(OSM_ENRICHED_CSV, usecols=["h3_cell", "shift", "hour"])

shift_dist = df.groupby(["h3_cell", "shift"]).size().reset_index(name="count")
dominant_shift = (
    shift_dist.sort_values("count", ascending=False)
    .drop_duplicates("h3_cell", keep="first")
    [["h3_cell", "shift"]]
    .rename(columns={"shift": "recommended_shift"})
)

# Also get peak hour
peak_hours = df.groupby("h3_cell")["hour"].agg(
    lambda x: f"{x.mode().iloc[0]}:00" if len(x.mode()) > 0 else "N/A"
).rename("peak_hour")

cell_stats = cell_stats.merge(dominant_shift, on="h3_cell", how="left")
cell_stats = cell_stats.merge(peak_hours, on="h3_cell", how="left")
cell_stats["recommended_shift"] = cell_stats["recommended_shift"].fillna("evening_shift")

print(f"  Shift distribution:")
print(f"    {cell_stats['recommended_shift'].value_counts().to_dict()}")


# ============================================================================
# STEP 4: Generate patrol schedule per station
# ============================================================================
# %%
print("\n" + "=" * 70)
print("STEP 4: Generating patrol schedule per station")
print("=" * 70)

patrol_records = []

stations = cell_stats["police_station"].unique()
print(f"  Generating schedules for {len(stations)} stations...")

for station in sorted(stations):
    station_cells = cell_stats[cell_stats["police_station"] == station]

    # Rank by priority score within this station
    top_cells = station_cells.nlargest(TOP_N_ZONES_PER_STATION, "priority_score")

    for rank_idx, (_, cell) in enumerate(top_cells.iterrows(), 1):
        # Generate plain-language reasoning
        reasons = []

        # CIS reasoning
        if cell["CIS"] > 70:
            reasons.append(f"High congestion impact (CIS={cell['CIS']:.0f}/100)")
        elif cell["CIS"] > 40:
            reasons.append(f"Moderate congestion impact (CIS={cell['CIS']:.0f}/100)")

        # Trend reasoning
        trend = cell.get("trend_direction", "STABLE")
        if trend == "RISING":
            reasons.append("Violations RISING — proactive enforcement needed")
        elif trend == "FALLING":
            reasons.append("Violations declining — maintain presence")

        # Vehicle mix reasoning
        if cell.get("mean_vehicle_severity", 0) > 3:
            reasons.append(f"Heavy vehicles dominant ({cell.get('dominant_vehicle','N/A')})")

        # Road impact reasoning
        if cell.get("bpr_delay_min_per_km", 0) > 2:
            reasons.append(
                f"Estimated {cell['bpr_delay_min_per_km']:.1f} min/km delay from parking"
            )

        # Peak hour reasoning
        if cell.get("peak_ratio", 0) > 0.5:
            reasons.append(f"Peak hour concentration ({cell['peak_ratio']:.0%})")

        # Chronic offenders
        if cell.get("chronic_offender_count", 0) > 5:
            reasons.append(
                f"{int(cell['chronic_offender_count'])} repeat offenders identified"
            )

        patrol_records.append({
            "police_station": station,
            "h3_cell": cell["h3_cell"],
            "location_name": cell.get("location_name", "N/A"),
            "lat": cell["lat"],
            "lon": cell["lon"],
            "priority_rank": rank_idx,
            "priority_score": round(cell["priority_score"], 1),
            "CIS": round(cell["CIS"], 1),
            "predicted_risk": round(cell.get("predicted_risk", 0), 1),
            "trend_direction": trend,
            "hotspot_status": cell.get("hotspot_status", "N/A"),
            "recommended_shift": cell["recommended_shift"],
            "peak_hour": cell.get("peak_hour", "N/A"),
            "bpr_delay": round(cell.get("bpr_delay_min_per_km", 0), 2),
            "violation_count": int(cell["violation_count"]),
            "reasoning": " | ".join(reasons) if reasons else "Standard priority zone",
        })

patrol_df = pd.DataFrame(patrol_records)
print(f"  Generated {len(patrol_df)} patrol recommendations")
print(f"  Across {patrol_df['police_station'].nunique()} stations")


# ============================================================================
# STEP 5: Save patrol schedule
# ============================================================================
# %%
print("\n" + "=" * 70)
print("STEP 5: Saving patrol schedule")
print("=" * 70)

patrol_df.to_csv(PATROL_CSV, index=False)
print(f"  Saved to: {PATROL_CSV}")

# Also update the main hotspot file with priority info
cell_stats.to_csv(HOTSPOTS_CSV, index=False)


# ============================================================================
# SUMMARY
# ============================================================================
# %%
print("\n" + "=" * 70)
print("SUMMARY — Phase 7 Complete")
print("=" * 70)

print(f"\n  Sample patrol briefs:\n")

for station in patrol_df["police_station"].unique()[:3]:
    station_data = patrol_df[patrol_df["police_station"] == station]
    print(f"  📋 {station} Station:")
    for _, row in station_data.iterrows():
        print(f"    #{row['priority_rank']}. {row['location_name']}")
        print(f"       CIS={row['CIS']} | Risk={row['predicted_risk']} | "
              f"Trend={row['trend_direction']}")
        print(f"       Deploy: {row['recommended_shift']} (peak at {row['peak_hour']})")
        print(f"       {row['reasoning']}")
    print()


if __name__ == "__main__":
    print("Phase 7 complete. Output saved to:", PATROL_CSV)
