# ============================================================================
# tools.py — Tool Functions for Agentic Layer
# ============================================================================
# Plain Python functions that query pre-computed pipeline outputs.
# Each function returns a formatted string that agents can interpret.
# Agents NEVER access raw CSVs directly — they call these tools.
# ============================================================================

import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------------------------
# Load pre-computed data at import time (runs once)
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

def _load_csv(filename):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

# Lazy loading — populated on first tool call
_cache = {}

def _get_data(key):
    """Lazy-load and cache dataframes."""
    if key not in _cache or _cache[key].empty:
        mapping = {
            "impact": "impact_scores.csv",
            "hotspots": "hotspots.csv",
            "forecasts": "forecasts.csv",
            "patrol": "patrol_schedule.csv",
            "queueing": "queueing_results.csv",
            "violations": "cleaned_violations.csv",
        }
        _cache[key] = _load_csv(mapping.get(key, ""))
    return _cache[key]


# ============================================================================
# TOOL 1: Query Top Hotspots
# ============================================================================
def query_top_hotspots(n: int = 10, police_station: str = None) -> str:
    """
    Get the top N illegal parking hotspots ranked by Congestion Impact Score.
    Optionally filter by police station name.

    Args:
        n: Number of top hotspots to return (default 10)
        police_station: Filter by station name (optional)

    Returns:
        Formatted string with hotspot details
    """
    df = _get_data("impact")
    if df.empty:
        return "Error: Impact scores data not available. Run the pipeline first."

    if police_station:
        df = df[df["police_station"].str.contains(police_station, case=False, na=False)]
        if df.empty:
            return f"No hotspots found for station '{police_station}'. Check station name."

    top = df.nlargest(n, "CIS")

    header = f"Top {len(top)} Hotspots"
    if police_station:
        header += f" — {police_station} Station"
    header += f"\n{'─' * 60}\n"

    result = header
    for _, row in top.iterrows():
        status_icon = {
            "CONFIRMED_HIGH": "🔴",
            "CONFIRMED": "🟠",
            "EMERGING": "🟡"
        }.get(row.get("hotspot_status", ""), "⚪")

        result += (
            f"\n{status_icon} #{int(row.get('rank', 0))}. "
            f"{row.get('location_name', 'Unknown')}\n"
            f"   CIS: {row['CIS']:.1f}/100 | "
            f"Violations: {int(row.get('violation_count', 0))} | "
            f"Status: {row.get('hotspot_status', 'N/A')}\n"
            f"   BPR Delay: {row.get('bpr_delay_min_per_km', 0):.2f} min/km | "
            f"Vehicles: {row.get('dominant_vehicle', 'N/A')}\n"
        )

    return result


# ============================================================================
# TOOL 2: Get Zone Details (CIS Breakdown)
# ============================================================================
def get_zone_details(location_name: str) -> str:
    """
    Get detailed CIS breakdown for a specific zone.
    Shows WHY the zone scores high or low on congestion impact.

    Args:
        location_name: Name or partial name of the location/zone

    Returns:
        Formatted CIS breakdown with all components
    """
    df = _get_data("impact")
    if df.empty:
        return "Error: Impact scores data not available."

    # Search by location name (fuzzy)
    matches = df[df["location_name"].str.contains(location_name, case=False, na=False)]
    if matches.empty:
        # Try police station
        matches = df[df["police_station"].str.contains(location_name, case=False, na=False)]
    if matches.empty:
        return f"No data found for '{location_name}'. Try a different search term."

    row = matches.nlargest(1, "CIS").iloc[0]

    result = f"""
📍 Zone: {row.get('location_name', 'Unknown')}
   Station: {row.get('police_station', 'N/A')}
   Status: {row.get('hotspot_status', 'N/A')}

📊 Congestion Impact Score: {row['CIS']:.1f}/100 (Rank #{int(row.get('rank', 0))})

   CIS Component Breakdown:
   ┌───────────────────────────┬────────┬────────┐
   │ Component                 │ Score  │ Weight │
   ├───────────────────────────┼────────┼────────┤
   │ Violation Density         │ {row.get('cis_violation_density', 0):.3f}  │  25%   │
   │ Vehicle Severity          │ {row.get('cis_vehicle_severity', 0):.3f}  │  20%   │
   │ Queueing Delay (BPR)      │ {row.get('cis_queueing_delay', 0):.3f}  │  20%   │
   │ Road Capacity Impact      │ {row.get('cis_road_capacity_impact', 0):.3f}  │  15%   │
   │ Temporal Persistence      │ {row.get('cis_temporal_persistence', 0):.3f}  │  10%   │
   │ Peak Hour Ratio           │ {row.get('cis_peak_hour_ratio', 0):.3f}  │  10%   │
   └───────────────────────────┴────────┴────────┘

   Additional Context:
   • Estimated delay: {row.get('bpr_delay_min_per_km', 0):.2f} min/km during peak
   • Expected parked vehicles (E[N]): {row.get('expected_n', 0):.2f}
   • Avg dwell time: {row.get('mean_dwell', 0):.0f} minutes
   • Chronic offenders: {int(row.get('chronic_offender_count', 0))}
   • Peak hour concentration: {row.get('peak_ratio', 0):.0%}
   • Dominant vehicle: {row.get('dominant_vehicle', 'N/A')}
   • Dominant violation: {row.get('dominant_violation', 'N/A')}
   • Road type: {row.get('road_type', 'N/A')} ({int(row.get('lanes', 0))} lanes)
"""
    return result


# ============================================================================
# TOOL 3: Get Temporal Pattern
# ============================================================================
def get_temporal_pattern(location_name: str) -> str:
    """
    Get time-of-day and day-of-week violation patterns for a zone.

    Args:
        location_name: Name or partial name of the location

    Returns:
        Temporal pattern analysis
    """
    df = _get_data("violations")
    if df.empty:
        return "Error: Violation data not available."

    # Find matching rows
    matches = df[
        df["police_station"].str.contains(location_name, case=False, na=False) |
        df["junction_name"].str.contains(location_name, case=False, na=False) |
        df["location"].str.contains(location_name, case=False, na=False)
    ]

    if matches.empty:
        return f"No violation records found for '{location_name}'."

    # Hourly distribution
    hourly = matches["hour"].value_counts().sort_index()
    peak_hour = hourly.idxmax()
    peak_count = hourly.max()

    # Daily distribution
    daily = matches["day_of_week"].value_counts()
    peak_day = daily.idxmax()

    # Peak ratio
    peak_pct = matches["is_peak_hour"].mean() if "is_peak_hour" in matches.columns else 0

    # Weekend ratio
    weekend_pct = matches["is_weekend"].mean() if "is_weekend" in matches.columns else 0

    # Shift distribution
    shift_dist = matches["shift"].value_counts() if "shift" in matches.columns else pd.Series()

    result = f"""
⏰ Temporal Pattern for: {location_name}
   (Based on {len(matches):,} violation records)

   Peak Hour: {peak_hour}:00 ({peak_count} violations)
   Peak Day: {peak_day} ({daily[peak_day]} violations)
   Peak Hour Concentration: {peak_pct:.0%} during rush hours
   Weekend Violations: {weekend_pct:.0%}

   Hourly Distribution (top 5):
"""
    for hour, count in hourly.nlargest(5).items():
        bar = "█" * int(count / hourly.max() * 20)
        result += f"     {hour:02d}:00  {bar} {count}\n"

    if not shift_dist.empty:
        result += "\n   Recommended Patrol Shift:\n"
        best_shift = shift_dist.idxmax()
        result += f"     → {best_shift} ({shift_dist[best_shift]} violations)\n"

    return result


# ============================================================================
# TOOL 4: Get Forecast
# ============================================================================
def get_forecast(location_name: str) -> str:
    """
    Get future risk prediction for a zone.

    Args:
        location_name: Name or partial name of the location

    Returns:
        Forecast with predicted violations and trend
    """
    df = _get_data("forecasts")
    if df.empty:
        return "Error: Forecast data not available."

    matches = df[
        df["location_name"].str.contains(location_name, case=False, na=False) |
        df["police_station"].str.contains(location_name, case=False, na=False)
    ]

    if matches.empty:
        return f"No forecast available for '{location_name}'."

    row = matches.nlargest(1, "predicted_risk").iloc[0]

    trend_icon = {"RISING": "📈", "STABLE": "➡️", "FALLING": "📉"}.get(
        row.get("trend_direction", ""), "❓"
    )

    result = f"""
🔮 Forecast for: {row.get('location_name', location_name)}

   Predicted violations next week: ~{int(row.get('predicted_violations_next_week', 0))}
   Risk Score: {row.get('predicted_risk', 0):.1f}/100
   Risk Category: {row.get('risk_category', 'N/A')}
   Trend: {trend_icon} {row.get('trend_direction', 'N/A')}
   Forecast Method: {row.get('forecast_method', 'N/A')}
   Current CIS: {row.get('CIS', 0):.1f}
"""
    return result


# ============================================================================
# TOOL 5: Get Station Patrol Brief
# ============================================================================
def get_station_patrol_brief(police_station: str) -> str:
    """
    Get the full patrol brief for a specific police station.
    Includes priority zones, recommended shifts, and reasoning.

    Args:
        police_station: Station name (e.g., "Madiwala", "Kodigehalli")

    Returns:
        Actionable patrol brief
    """
    df = _get_data("patrol")
    if df.empty:
        return "Error: Patrol schedule not available."

    matches = df[df["police_station"].str.contains(police_station, case=False, na=False)]
    if matches.empty:
        # List available stations
        all_stations = sorted(df["police_station"].unique())
        return (f"Station '{police_station}' not found. "
                f"Available: {', '.join(all_stations[:10])}...")

    station_name = matches["police_station"].iloc[0]
    result = f"""
📋 PATROL BRIEF — {station_name} Station
{'═' * 50}
Priority zones for enforcement this week:
"""

    for _, row in matches.sort_values("priority_rank").iterrows():
        trend_icon = {"RISING": "📈", "STABLE": "➡️", "FALLING": "📉"}.get(
            row.get("trend_direction", ""), "❓"
        )
        result += f"""
  #{int(row['priority_rank'])}. {row.get('location_name', 'N/A')}
     ├─ CIS: {row['CIS']:.0f} | Risk: {row.get('predicted_risk', 0):.0f} | {trend_icon} {row.get('trend_direction', '')}
     ├─ Deploy: {row.get('recommended_shift', 'N/A')} (peak at {row.get('peak_hour', 'N/A')})
     ├─ Estimated delay: {row.get('bpr_delay', 0):.2f} min/km
     └─ {row.get('reasoning', 'Standard priority')}
"""

    return result


# ============================================================================
# TOOL 6: Get Repeat Offenders
# ============================================================================
def get_repeat_offenders(location_name: str = None, top_n: int = 10) -> str:
    """
    Get vehicles that repeatedly park illegally.

    Args:
        location_name: Optional location filter
        top_n: Number of top offenders to return

    Returns:
        List of repeat offenders with details
    """
    df = _get_data("violations")
    if df.empty:
        return "Error: Violation data not available."

    if location_name:
        df = df[
            df["police_station"].str.contains(location_name, case=False, na=False) |
            df["junction_name"].str.contains(location_name, case=False, na=False)
        ]
        if df.empty:
            return f"No records found for '{location_name}'."

    repeats = df["vehicle_number"].value_counts().head(top_n)

    result = f"🚗 Top {min(top_n, len(repeats))} Repeat Offenders"
    if location_name:
        result += f" at {location_name}"
    result += f"\n{'─' * 50}\n"

    for veh, count in repeats.items():
        veh_data = df[df["vehicle_number"] == veh].iloc[0]
        result += (
            f"\n  {veh} ({veh_data.get('vehicle_type', 'N/A')}): "
            f"{count} violations\n"
        )

    return result


# ============================================================================
# TOOL 7: General Statistics Query
# ============================================================================
def get_general_stats() -> str:
    """
    Get overall system statistics and summary.

    Returns:
        System-wide statistics summary
    """
    impact = _get_data("impact")
    patrol = _get_data("patrol")
    forecasts = _get_data("forecasts")

    if impact.empty:
        return "Error: Data not available. Run the pipeline first."

    confirmed = impact[impact["hotspot_status"].isin(["CONFIRMED_HIGH", "CONFIRMED"])]

    result = f"""
📊 SYSTEM OVERVIEW — Bengaluru Parking Congestion Intelligence
{'═' * 60}

  Total H3 cells analyzed: {len(impact):,}
  Confirmed hotspots: {len(confirmed)}
  Police stations covered: {impact['police_station'].nunique()}

  CIS Statistics:
    Mean: {impact['CIS'].mean():.1f} | Max: {impact['CIS'].max():.1f}
    Cells with CIS > 50: {(impact['CIS'] > 50).sum()}
    Cells with CIS > 75: {(impact['CIS'] > 75).sum()}
"""

    if not forecasts.empty:
        result += f"""
  Risk Forecast:
    CRITICAL risk zones: {(forecasts.get('risk_category', pd.Series()) == 'CRITICAL').sum()}
    HIGH risk zones: {(forecasts.get('risk_category', pd.Series()) == 'HIGH').sum()}
    Rising trend: {(forecasts.get('trend_direction', pd.Series()) == 'RISING').sum()}
"""

    if not patrol.empty:
        result += f"""
  Patrol Schedule:
    Total recommendations: {len(patrol)}
    Stations with briefs: {patrol['police_station'].nunique()}
"""

    return result


# ============================================================================
# TOOL 8: Lookup Offence Code (for Policy/RAG Agent)
# ============================================================================
# Motor Vehicles Act sections mapped to offence codes in the dataset
OFFENCE_CODE_MAP = {
    112: {"name": "WRONG PARKING", "section": "Section 122/177 MVA",
          "description": "Parking in a manner causing obstruction or inconvenience",
          "fine": "₹500 (first offence), ₹1500 (subsequent)"},
    113: {"name": "NO PARKING", "section": "Section 122/177 MVA",
          "description": "Parking in a designated No Parking zone",
          "fine": "₹500 (first offence), ₹1500 (subsequent)"},
    107: {"name": "PARKING IN A MAIN ROAD", "section": "Section 122 MVA",
          "description": "Parking on a main road causing traffic obstruction",
          "fine": "₹500-₹1500"},
    109: {"name": "DOUBLE PARKING", "section": "Section 122/177 MVA",
          "description": "Parking alongside an already parked vehicle, blocking traffic lane",
          "fine": "₹500-₹1500 + towing charges"},
    104: {"name": "PARKING NEAR ROAD CROSSING", "section": "Section 122 MVA",
          "description": "Parking within 30m of a road crossing/intersection",
          "fine": "₹500"},
    105: {"name": "PARKING ON FOOTPATH", "section": "Section 122/177 MVA",
          "description": "Parking on pedestrian footpath",
          "fine": "₹500"},
    106: {"name": "PARKING NEAR TRAFFIC LIGHT", "section": "Section 122 MVA",
          "description": "Parking near traffic light or zebra crossing",
          "fine": "₹500-₹1500"},
    108: {"name": "PARKING OPPOSITE ANOTHER VEHICLE", "section": "Section 122 MVA",
          "description": "Parking opposite to another parked vehicle narrowing road",
          "fine": "₹500"},
    111: {"name": "PARKING NEAR BUSTOP/SCHOOL/HOSPITAL", "section": "Section 122 MVA",
          "description": "Parking near bus stop, school, or hospital entrance",
          "fine": "₹500-₹1500"},
    116: {"name": "DEFECTIVE NUMBER PLATE", "section": "Section 39/192 MVA",
          "description": "Vehicle with non-standard or defective registration plate",
          "fine": "₹5000"},
}

def lookup_offence_code(code: int = None, violation_name: str = None) -> str:
    """
    Look up legal details for a violation offence code or name.

    Args:
        code: Offence code number (e.g., 112)
        violation_name: Violation name to search (e.g., "double parking")

    Returns:
        Legal details including MVA section, description, and fine
    """
    if code and code in OFFENCE_CODE_MAP:
        info = OFFENCE_CODE_MAP[code]
        return (
            f"⚖️ Offence Code {code}: {info['name']}\n"
            f"   Legal basis: {info['section']}\n"
            f"   Description: {info['description']}\n"
            f"   Penalty: {info['fine']}\n"
        )

    if violation_name:
        for c, info in OFFENCE_CODE_MAP.items():
            if violation_name.lower() in info["name"].lower():
                return (
                    f"⚖️ Offence Code {c}: {info['name']}\n"
                    f"   Legal basis: {info['section']}\n"
                    f"   Description: {info['description']}\n"
                    f"   Penalty: {info['fine']}\n"
                )
        return f"No matching offence found for '{violation_name}'."

    # Return all codes
    result = "⚖️ All Offence Codes:\n"
    for c, info in sorted(OFFENCE_CODE_MAP.items()):
        result += f"  {c}: {info['name']} — {info['section']} ({info['fine']})\n"
    return result
