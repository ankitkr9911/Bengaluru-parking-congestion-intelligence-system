# ============================================================================
# config.py — Central configuration for the entire pipeline
# ============================================================================
# All tunable parameters in one place. Nothing is hardcoded in the modules.
# ============================================================================

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Output file paths
CLEANED_CSV       = os.path.join(OUTPUT_DIR, "cleaned_violations.csv")
OSM_ENRICHED_CSV  = os.path.join(OUTPUT_DIR, "osm_enriched.csv")
HOTSPOTS_CSV      = os.path.join(OUTPUT_DIR, "hotspots.csv")
IMPACT_SCORES_CSV = os.path.join(OUTPUT_DIR, "impact_scores.csv")
FORECASTS_CSV     = os.path.join(OUTPUT_DIR, "forecasts.csv")
PATROL_CSV        = os.path.join(OUTPUT_DIR, "patrol_schedule.csv")
QUEUEING_CSV      = os.path.join(OUTPUT_DIR, "queueing_results.csv")

# ---------------------------------------------------------------------------
# Timezone fix (#1 from review)
# All timestamps in the CSV are UTC (+00). Bengaluru is IST (UTC+5:30).
# We convert BEFORE any temporal feature is derived.
# ---------------------------------------------------------------------------
SOURCE_TZ = "UTC"
TARGET_TZ = "Asia/Kolkata"  # IST = UTC+5:30

# ---------------------------------------------------------------------------
# Validation status filtering
# Keep: approved, processing, created1, NaN (unvalidated ≠ bad)
# Drop: rejected, duplicate (confirmed bad)
# ---------------------------------------------------------------------------
DROP_VALIDATION_STATUSES = ["rejected", "duplicate"]

# ---------------------------------------------------------------------------
# Vehicle severity weights
# Bigger vehicles block more road capacity → higher severity
# ---------------------------------------------------------------------------
VEHICLE_SEVERITY = {
    "SCOOTER":             1.0,
    "MOPED":               1.0,
    "MOTOR CYCLE":         1.0,
    "PASSENGER AUTO":      1.5,
    "CAR":                 2.0,
    "JEEP":                2.0,
    "VAN":                 2.5,
    "MAXI-CAB":            3.0,
    "SCHOOL VEHICLE":      3.0,
    "LGV":                 3.5,
    "TEMPO":               3.5,
    "GOODS AUTO":          3.0,
    "MINI LORRY":          4.0,
    "LORRY/GOODS VEHICLE": 4.5,
    "PRIVATE BUS":         5.0,
    "BUS (BMTC/KSRTC)":   5.0,
    "TOURIST BUS":         5.0,
    "FACTORY BUS":         5.0,
    "HGV":                 5.0,
    "TANKER":              6.0,
    "TRACTOR":             4.0,
    "OTHERS":              2.0,
}

# ---------------------------------------------------------------------------
# Violation type severity weights
# How much each violation type contributes to road obstruction
# ---------------------------------------------------------------------------
VIOLATION_SEVERITY = {
    "DOUBLE PARKING":                            1.0,   # Blocks a full lane
    "PARKING IN A MAIN ROAD":                    0.9,   # Arterial obstruction
    "PARKING NEAR ROAD CROSSING":                0.85,
    "PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS": 0.85,
    "PARKING OPPOSITE TO ANOTHER PARKED VEHICLE":0.8,   # Narrows road to 1 lane
    "PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC":   0.75,
    "WRONG PARKING":                             0.6,   # Generic
    "NO PARKING":                                0.5,   # Zone violation
    "PARKING ON FOOTPATH":                       0.3,   # Pedestrian impact, low road impact
    "PARKING OTHER THAN BUS STOP":               0.5,
    "H T V PROHIBITED":                          0.7,
}

# ---------------------------------------------------------------------------
# Peak hour definition (IST)
# Bengaluru typical rush hours
# ---------------------------------------------------------------------------
MORNING_PEAK = (8, 11)   # 8:00 AM – 11:00 AM IST
EVENING_PEAK = (17, 21)  # 5:00 PM – 9:00 PM IST

# ---------------------------------------------------------------------------
# ST-DBSCAN parameters
# eps1: spatial distance in meters
# eps2: temporal distance in minutes
# min_samples: minimum points to form a cluster
# ---------------------------------------------------------------------------
STDBSCAN_EPS1 = 200       # 200 meters spatial radius
STDBSCAN_EPS2 = 120       # 2 hour temporal window (in minutes)
STDBSCAN_MIN_SAMPLES = 10 # Minimum violations to form a cluster

# ---------------------------------------------------------------------------
# H3 hexagonal grid resolution
# Resolution 9 → ~175m edge length (matched to ST-DBSCAN eps1)
# ---------------------------------------------------------------------------
H3_RESOLUTION = 9

# ---------------------------------------------------------------------------
# Spatial statistics significance threshold
# ---------------------------------------------------------------------------
SIGNIFICANCE_LEVEL = 0.05  # p < 0.05 for confirmed hotspots

# ---------------------------------------------------------------------------
# BPR (Bureau of Public Roads) volume-delay function parameters
# travel_time = free_flow_time × (1 + α × (V/C)^β)
# Standard values from transportation engineering
# ---------------------------------------------------------------------------
BPR_ALPHA = 0.15
BPR_BETA = 4.0

# ---------------------------------------------------------------------------
# CIS (Congestion Impact Score) weights
# Subject to sensitivity analysis — see cis_calculation.py
# ---------------------------------------------------------------------------
CIS_WEIGHTS = {
    "violation_density":     0.25,
    "vehicle_severity":      0.20,
    "queueing_delay":        0.20,
    "road_capacity_impact":  0.15,
    "temporal_persistence":  0.10,
    "peak_hour_ratio":       0.10,
}

# ---------------------------------------------------------------------------
# Forecasting parameters
# ---------------------------------------------------------------------------
# Only run full Prophet on cells with this many weekly data points minimum
PROPHET_MIN_WEEKS = 8
# Top N hotspots to run full Prophet on
PROPHET_TOP_N_HOTSPOTS = 50

# ---------------------------------------------------------------------------
# Patrol prioritization
# ---------------------------------------------------------------------------
TOP_N_ZONES_PER_STATION = 5  # Recommend top 5 zones per station

# ---------------------------------------------------------------------------
# OSM road type classification
# Map OSM highway tags → simplified categories
# ---------------------------------------------------------------------------
ROAD_TYPE_MAP = {
    "motorway":       "arterial",
    "motorway_link":  "arterial",
    "trunk":          "arterial",
    "trunk_link":     "arterial",
    "primary":        "arterial",
    "primary_link":   "arterial",
    "secondary":      "collector",
    "secondary_link": "collector",
    "tertiary":       "collector",
    "tertiary_link":  "collector",
    "residential":    "local",
    "living_street":  "local",
    "unclassified":   "local",
    "service":        "local",
}

# Lane capacity per road type (vehicles per hour per lane, approximate)
ROAD_CAPACITY_PER_LANE = {
    "arterial":  1800,
    "collector": 1200,
    "local":      800,
}

# Default lanes when OSM data is missing
DEFAULT_LANES = {
    "arterial":  4,
    "collector": 2,
    "local":     2,
}
