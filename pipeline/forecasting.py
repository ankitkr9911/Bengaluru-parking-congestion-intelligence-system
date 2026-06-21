# ============================================================================
# forecasting.py — Phase 6: Hotspot Forecasting
# ============================================================================
# FIX #4: Prophet only on data-dense hotspots; trend classifier for the rest.
#
#   a) For top-N confirmed hotspots with ≥8 weeks of data: run full Prophet
#      time-series forecasting to predict next-week violation counts.
#
#   b) For all other cells: use a simple slope-based trend classifier
#      (rising / stable / falling) computed from recent weekly counts.
#
# Output: per-cell predicted_risk, trend_direction, forecast confidence
# ============================================================================

# %%
import pandas as pd
import numpy as np
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    from pipeline.config import (
        OSM_ENRICHED_CSV, HOTSPOTS_CSV, FORECASTS_CSV, OUTPUT_DIR,
        PROPHET_MIN_WEEKS, PROPHET_TOP_N_HOTSPOTS
    )
except ImportError:
    OSM_ENRICHED_CSV = "data/osm_enriched.csv"
    HOTSPOTS_CSV = "data/hotspots.csv"
    FORECASTS_CSV = "data/forecasts.csv"
    OUTPUT_DIR = "data"
    PROPHET_MIN_WEEKS = 8
    PROPHET_TOP_N_HOTSPOTS = 50


# ============================================================================
# STEP 1: Load data and build weekly time series per cell
# ============================================================================
# %%
print("=" * 70)
print("STEP 1: Building weekly time series per H3 cell")
print("=" * 70)

df = pd.read_csv(OSM_ENRICHED_CSV)
df["created_datetime_ist"] = pd.to_datetime(df["created_datetime_ist"], format="mixed")
df["date"] = pd.to_datetime(df["date"])

cell_stats = pd.read_csv(HOTSPOTS_CSV)

# Create weekly violation counts per cell
df["week_start"] = df["created_datetime_ist"].dt.to_period("W").apply(lambda r: r.start_time)

weekly_counts = (
    df.groupby(["h3_cell", "week_start"])
    .agg(
        weekly_violations=("id", "count"),
        weekly_severity=("row_severity", "mean"),
        weekly_peak_ratio=("is_peak_hour", "mean"),
    )
    .reset_index()
)

# Fill missing weeks with 0 for each cell
all_weeks = pd.date_range(
    weekly_counts["week_start"].min(),
    weekly_counts["week_start"].max(),
    freq="W-MON"
)

print(f"  Total weeks in dataset: {len(all_weeks)}")
print(f"  Week range: {all_weeks[0].strftime('%Y-%m-%d')} → {all_weeks[-1].strftime('%Y-%m-%d')}")
print(f"  Cells with time series: {weekly_counts['h3_cell'].nunique():,}")

# Count weeks per cell
weeks_per_cell = weekly_counts.groupby("h3_cell")["week_start"].count()
print(f"  Cells with ≥{PROPHET_MIN_WEEKS} weeks of data: "
      f"{(weeks_per_cell >= PROPHET_MIN_WEEKS).sum()}")


# ============================================================================
# STEP 2: Identify cells eligible for Prophet
# ============================================================================
# %%
print("\n" + "=" * 70)
print("STEP 2: Selecting cells for Prophet forecasting")
print("=" * 70)

# Eligible = confirmed/emerging hotspot + enough data points
confirmed_cells = set(
    cell_stats[cell_stats["hotspot_status"].isin(["CONFIRMED_HIGH", "CONFIRMED", "EMERGING"])]
    ["h3_cell"]
)

data_rich_cells = set(weeks_per_cell[weeks_per_cell >= PROPHET_MIN_WEEKS].index)

# Take top N by CIS among eligible cells
eligible = cell_stats[
    (cell_stats["h3_cell"].isin(confirmed_cells)) &
    (cell_stats["h3_cell"].isin(data_rich_cells))
].nlargest(PROPHET_TOP_N_HOTSPOTS, "CIS")

prophet_cells = set(eligible["h3_cell"])
print(f"  Prophet-eligible cells: {len(prophet_cells)}")
print(f"  (Confirmed/emerging hotspots with ≥{PROPHET_MIN_WEEKS} weeks of data)")


# ============================================================================
# STEP 3A: Prophet forecasting for top hotspots
# ============================================================================
# %%
print("\n" + "=" * 70)
print("STEP 3A: Running Prophet on top hotspots")
print("=" * 70)

from prophet import Prophet
import logging
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

prophet_results = []

for i, cell_id in enumerate(prophet_cells):
    # Get weekly data for this cell
    cell_data = weekly_counts[weekly_counts["h3_cell"] == cell_id].copy()
    cell_data = cell_data.sort_values("week_start")

    # Fill gaps: complete week range with 0s for missing weeks
    full_range = pd.DataFrame({"week_start": all_weeks})
    cell_data = full_range.merge(cell_data, on="week_start", how="left")
    cell_data["weekly_violations"] = cell_data["weekly_violations"].fillna(0)
    cell_data["h3_cell"] = cell_id

    # Prophet expects columns named 'ds' and 'y'
    prophet_df = pd.DataFrame({
        "ds": cell_data["week_start"],
        "y": cell_data["weekly_violations"],
    })

    try:
        # Fit Prophet (simple: weekly data, no strong seasonality expected)
        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.1,  # Conservative — short time series
        )
        model.fit(prophet_df)

        # Forecast next 2 weeks
        future = model.make_future_dataframe(periods=2, freq="W")
        forecast = model.predict(future)

        # Get the forecast for next week
        next_week = forecast.iloc[-2]  # Second to last = next week
        week_after = forecast.iloc[-1]  # Last = week after next

        # Trend direction from recent change
        recent = forecast.tail(5)
        slope = np.polyfit(range(len(recent)), recent["yhat"].values, 1)[0]

        if slope > 1:
            trend = "RISING"
        elif slope < -1:
            trend = "FALLING"
        else:
            trend = "STABLE"

        prophet_results.append({
            "h3_cell": cell_id,
            "forecast_method": "prophet",
            "predicted_violations_next_week": max(0, round(next_week["yhat"])),
            "predicted_lower": max(0, round(next_week["yhat_lower"])),
            "predicted_upper": max(0, round(next_week["yhat_upper"])),
            "trend_direction": trend,
            "trend_slope": slope,
            "forecast_confidence": 1 - (next_week["yhat_upper"] - next_week["yhat_lower"]) / max(next_week["yhat"], 1),
        })

        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{len(prophet_cells)} cells...")

    except Exception as e:
        # Fallback to simple trend if Prophet fails
        prophet_results.append({
            "h3_cell": cell_id,
            "forecast_method": "fallback_trend",
            "predicted_violations_next_week": int(cell_data["weekly_violations"].tail(4).mean()),
            "predicted_lower": 0,
            "predicted_upper": 0,
            "trend_direction": "UNKNOWN",
            "trend_slope": 0,
            "forecast_confidence": 0.3,
        })

prophet_df_results = pd.DataFrame(prophet_results)
print(f"\n  Prophet forecasting complete:")
print(f"    Cells forecasted: {len(prophet_df_results)}")
print(f"    Trend distribution: {prophet_df_results['trend_direction'].value_counts().to_dict()}")


# ============================================================================
# STEP 3B: Simple trend classifier for remaining cells
# ============================================================================
# For cells not eligible for Prophet: compute a slope-based trend from
# the last 4 weeks of violation counts (rising/stable/falling).
# ============================================================================
# %%
print("\n" + "=" * 70)
print("STEP 3B: Trend classification for remaining cells")
print("=" * 70)

remaining_cells = set(cell_stats["h3_cell"]) - prophet_cells
trend_results = []

for cell_id in remaining_cells:
    cell_data = weekly_counts[weekly_counts["h3_cell"] == cell_id].sort_values("week_start")

    if len(cell_data) < 2:
        trend_results.append({
            "h3_cell": cell_id,
            "forecast_method": "insufficient_data",
            "predicted_violations_next_week": int(cell_data["weekly_violations"].mean()) if len(cell_data) > 0 else 0,
            "predicted_lower": 0,
            "predicted_upper": 0,
            "trend_direction": "UNKNOWN",
            "trend_slope": 0,
            "forecast_confidence": 0.1,
        })
        continue

    # Use last 4 weeks (or all available)
    recent = cell_data.tail(4)
    values = recent["weekly_violations"].values

    # Linear regression slope
    if len(values) >= 2:
        slope = np.polyfit(range(len(values)), values, 1)[0]
    else:
        slope = 0

    # Classify trend
    mean_val = np.mean(values)
    if mean_val > 0:
        relative_slope = slope / mean_val
    else:
        relative_slope = 0

    if relative_slope > 0.1:
        trend = "RISING"
    elif relative_slope < -0.1:
        trend = "FALLING"
    else:
        trend = "STABLE"

    trend_results.append({
        "h3_cell": cell_id,
        "forecast_method": "trend_classifier",
        "predicted_violations_next_week": max(0, int(values[-1] + slope)),
        "predicted_lower": 0,
        "predicted_upper": 0,
        "trend_direction": trend,
        "trend_slope": slope,
        "forecast_confidence": 0.5,
    })

trend_df_results = pd.DataFrame(trend_results)
print(f"  Trend classification complete:")
print(f"    Cells classified: {len(trend_df_results):,}")
print(f"    Trend distribution: {trend_df_results['trend_direction'].value_counts().to_dict()}")


# ============================================================================
# STEP 4: Combine and compute risk scores
# ============================================================================
# %%
print("\n" + "=" * 70)
print("STEP 4: Computing risk scores")
print("=" * 70)

forecasts = pd.concat([prophet_df_results, trend_df_results], ignore_index=True)

# Merge with cell CIS data
forecasts = forecasts.merge(
    cell_stats[["h3_cell", "CIS", "hotspot_status", "police_station",
                "location_name", "lat", "lon"]],
    on="h3_cell", how="left"
)

# Risk score = CIS × trend multiplier
trend_multipliers = {"RISING": 1.3, "STABLE": 1.0, "FALLING": 0.7, "UNKNOWN": 0.8}
forecasts["trend_multiplier"] = forecasts["trend_direction"].map(trend_multipliers)
forecasts["predicted_risk"] = (
    forecasts["CIS"].fillna(0) * forecasts["trend_multiplier"]
).clip(0, 130)  # Can exceed 100 if rising

# Risk category
def risk_category(risk):
    if risk >= 80:
        return "CRITICAL"
    elif risk >= 60:
        return "HIGH"
    elif risk >= 40:
        return "MODERATE"
    elif risk >= 20:
        return "LOW"
    return "MINIMAL"

forecasts["risk_category"] = forecasts["predicted_risk"].apply(risk_category)

print(f"  Risk distribution:")
print(f"    {forecasts['risk_category'].value_counts().to_dict()}")


# ============================================================================
# STEP 5: Save forecasts
# ============================================================================
# %%
print("\n" + "=" * 70)
print("STEP 5: Saving forecasts")
print("=" * 70)

forecasts.to_csv(FORECASTS_CSV, index=False)
print(f"  Saved {FORECASTS_CSV}")
print(f"  Total cells with forecasts: {len(forecasts):,}")


# ============================================================================
# SUMMARY
# ============================================================================
# %%
print("\n" + "=" * 70)
print("SUMMARY — Phase 6 Complete")
print("=" * 70)

critical = forecasts[forecasts["risk_category"] == "CRITICAL"]
print(f"""
  Forecasting:
    Prophet-forecasted (top hotspots): {len(prophet_df_results)}
    Trend-classified (remaining): {len(trend_df_results):,}
    
  Risk breakdown:
    CRITICAL: {(forecasts['risk_category']=='CRITICAL').sum()}
    HIGH: {(forecasts['risk_category']=='HIGH').sum()}
    MODERATE: {(forecasts['risk_category']=='MODERATE').sum()}
    LOW: {(forecasts['risk_category']=='LOW').sum()}
    MINIMAL: {(forecasts['risk_category']=='MINIMAL').sum()}

  Top 10 highest-risk zones:
""")

top_risk = forecasts.nlargest(10, "predicted_risk")
for _, row in top_risk.iterrows():
    print(f"    {row.get('location_name','?'):30s} | "
          f"Risk={row['predicted_risk']:.1f} | "
          f"CIS={row.get('CIS',0):.1f} | "
          f"Trend={row['trend_direction']} | "
          f"Next week: ~{row['predicted_violations_next_week']} violations")


if __name__ == "__main__":
    print("\nPhase 6 complete. Output saved to:", FORECASTS_CSV)
