# ============================================================================
# hotspot_detection.py — Phase 3: Dual Hotspot Detection
# ============================================================================
# Two independent methods, cross-validated:
#
#   A) ST-DBSCAN (BallTree implementation) — spatiotemporal clusters that
#      recur in both space (~200m) and time (~2hr window).
#      Uses BallTree to query spatial neighbors first (O(n log n)), then
#      filters by temporal distance — never builds the O(n²) matrix that
#      causes OOM with the st_dbscan library on bursty timestamp data.
#
#   B) Getis-Ord Gi* + Local Moran's I — on H3 hexagonal cells.
#      Gi* identifies statistically significant hotspots (z-scores + p-values).
#      Moran's I classifies cluster types (HH / HL / LH / LL).
#
# Cross-validation:
#   - All 3 agree → CONFIRMED_HIGH
#   - 2 of 3 agree → CONFIRMED
#   - Only 1 fires → EMERGING
#   - None → NOT_HOTSPOT
# ============================================================================

# %%
import sys
import os
import pandas as pd
import numpy as np
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Ensure project root is on sys.path so local st_dbscan folder is found
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from pipeline.config import (
        OSM_ENRICHED_CSV, HOTSPOTS_CSV, OUTPUT_DIR,
        STDBSCAN_EPS1, STDBSCAN_EPS2, STDBSCAN_MIN_SAMPLES,
        H3_RESOLUTION, SIGNIFICANCE_LEVEL
    )
except ImportError:
    OSM_ENRICHED_CSV = "data/osm_enriched.csv"
    HOTSPOTS_CSV = "data/hotspots.csv"
    OUTPUT_DIR = "data"
    STDBSCAN_EPS1 = 200
    STDBSCAN_EPS2 = 120
    STDBSCAN_MIN_SAMPLES = 10
    H3_RESOLUTION = 9
    SIGNIFICANCE_LEVEL = 0.05


# ============================================================================
# STEP 1: Load enriched data
# ============================================================================
# %%
print("=" * 70)
print("STEP 1: Loading OSM-enriched data")
print("=" * 70)

df = pd.read_csv(OSM_ENRICHED_CSV)
df["created_datetime_ist"] = pd.to_datetime(df["created_datetime_ist"], format="mixed")
print(f"  Loaded {len(df):,} rows")


# ============================================================================
# STEP 2: Assign H3 hexagonal grid cells
# ============================================================================
# H3 resolution 9 → ~175m edge length (matched to ST-DBSCAN spatial scale)
# ============================================================================
# %%
print("\n" + "=" * 70)
print("STEP 2: Assigning H3 hexagonal grid cells")
print("=" * 70)

import h3

df["h3_cell"] = df.apply(
    lambda row: h3.latlng_to_cell(row["latitude"], row["longitude"], H3_RESOLUTION),
    axis=1
)

n_cells = df["h3_cell"].nunique()
print(f"  Unique H3 cells (resolution {H3_RESOLUTION}): {n_cells:,}")
print(f"  Avg violations per cell: {len(df) / n_cells:.1f}")

# Get cell center coordinates (for mapping later)
cell_centers = pd.DataFrame(df["h3_cell"].unique(), columns=["h3_cell"])
cell_centers[["cell_lat", "cell_lon"]] = cell_centers["h3_cell"].apply(
    lambda c: pd.Series(h3.cell_to_latlng(c))
)


# ============================================================================
# STEP 3A: ST-DBSCAN via BallTree (memory-efficient)
# ============================================================================
# Root cause of the st_dbscan library OOM:
#   The library builds the full temporal pairwise matrix first — checking
#   which pairs of ALL n points are within eps2=120 min of each other —
#   before applying the spatial filter. With bursty timestamps (batch
#   logging means many rows share near-identical times), this produces
#   O(n²) temporal pairs (275M entries for 100K points) before a single
#   spatial check is done.
#
# BallTree fix: query spatial neighbors (eps1=200m) FIRST for every point.
#   Only ~10-100 candidates per point pass the spatial filter (200m is tiny
#   in a city). Then check which of those are also within 120 min.
#   No global pairwise matrix is ever built. Runs on all 248K rows directly.
# ============================================================================
# %%
print("\n" + "=" * 70)
print("STEP 3A: Running ST-DBSCAN (BallTree — spatial-first, memory-safe)")
print("=" * 70)

from sklearn.neighbors import BallTree

# Convert lat/lon → meters (flat-earth approx, accurate enough for <50km)
center_lat = df["latitude"].mean()
df["x_meters"] = (
    (df["longitude"] - df["longitude"].min())
    * 111320 * np.cos(np.radians(center_lat))
)
df["y_meters"] = (df["latitude"] - df["latitude"].min()) * 111320

# Convert datetime → minutes since dataset start
time_start = df["created_datetime_ist"].min()
df["time_minutes"] = (
    (df["created_datetime_ist"] - time_start).dt.total_seconds() / 60
)

coords_xy = df[["x_meters", "y_meters"]].values.astype(np.float64)
times     = df["time_minutes"].values.astype(np.float64)

# Build spatial index once (O(n log n))
print(f"  Building BallTree on {len(df):,} points...")
tree = BallTree(coords_xy, metric="euclidean")

# Pre-compute spatial neighbor lists for all points
print(f"  Querying spatial neighbors within {STDBSCAN_EPS1}m...")
neighbors_spatial = tree.query_radius(coords_xy, r=STDBSCAN_EPS1)
# neighbors_spatial[i] = array of indices within eps1 metres of point i


def get_st_neighbors(idx):
    """Spatiotemporal neighbors: spatial candidates → temporal filter."""
    sp = neighbors_spatial[idx]
    return sp[np.abs(times[sp] - times[idx]) <= STDBSCAN_EPS2]


# DBSCAN label propagation
print(f"  Running DBSCAN label propagation (eps1={STDBSCAN_EPS1}m, "
      f"eps2={STDBSCAN_EPS2}min, min_samples={STDBSCAN_MIN_SAMPLES})...")

n       = len(df)
labels  = np.full(n, -1, dtype=np.int32)
visited = np.zeros(n, dtype=bool)
cluster_id = 0

for i in range(n):
    if visited[i]:
        continue
    visited[i] = True

    st_nbrs = get_st_neighbors(i)

    if len(st_nbrs) < STDBSCAN_MIN_SAMPLES:
        continue  # noise — may be absorbed later

    # Start a new cluster
    labels[i] = cluster_id
    queue = list(st_nbrs)
    in_queue = set(queue)

    qi = 0
    while qi < len(queue):
        j = queue[qi]
        qi += 1

        if not visited[j]:
            visited[j] = True
            j_nbrs = get_st_neighbors(j)
            if len(j_nbrs) >= STDBSCAN_MIN_SAMPLES:
                for nb in j_nbrs:
                    if nb not in in_queue:
                        in_queue.add(nb)
                        queue.append(nb)

        if labels[j] == -1:
            labels[j] = cluster_id

    cluster_id += 1

    if (i + 1) % 25000 == 0:
        print(f"    Progress: {i+1:,}/{n:,} ({(i+1)/n:.0%}) "
              f"— {cluster_id} clusters found so far...")

df["stdbscan_cluster"] = labels

n_clustered = int((labels >= 0).sum())
n_noise     = int((labels == -1).sum())

print(f"  ST-DBSCAN results:")
print(f"    Clusters found: {cluster_id}")
print(f"    Clustered points: {n_clustered:,} ({n_clustered/n:.1%})")
print(f"    Noise points: {n_noise:,} ({n_noise/n:.1%})")

# H3 cells that contain at least one clustered violation
stdbscan_cells = set(df[df["stdbscan_cluster"] >= 0]["h3_cell"].unique())
print(f"    H3 cells with clusters: {len(stdbscan_cells)}")


# ============================================================================
# STEP 3B: Aggregate violations per H3 cell (for spatial statistics)
# ============================================================================
# %%
print("\n" + "=" * 70)
print("STEP 3B: Aggregating violations per H3 cell")
print("=" * 70)

cell_stats = df.groupby("h3_cell").agg(
    violation_count=("id", "count"),
    unique_vehicles=("vehicle_number", "nunique"),
    unique_devices=("device_id", "nunique"),
    unique_days=("date", "nunique"),
    mean_severity=("row_severity", "mean"),
    peak_hour_count=("is_peak_hour", "sum"),
    mean_vehicle_severity=("vehicle_severity", "mean"),
    dominant_vehicle=("vehicle_type", lambda x: x.mode().iloc[0]),
    dominant_violation=("violation_types_flat", lambda x: x.mode().iloc[0]),
    lat=("latitude", "mean"),
    lon=("longitude", "mean"),
    police_station=("police_station", lambda x: x.mode().iloc[0]),
    has_junction_pct=("has_junction", "mean"),
).reset_index()

# Add road context (mode of road_type per cell)
if "road_type" in df.columns:
    road_mode = df.groupby("h3_cell")["road_type"].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "local"
    ).rename("road_type")
    lanes_mode = df.groupby("h3_cell")["lanes"].agg("median").rename("lanes")
    capacity_mode = df.groupby("h3_cell")["road_capacity"].agg("median").rename("road_capacity")
    cell_stats = cell_stats.merge(road_mode, on="h3_cell", how="left")
    cell_stats = cell_stats.merge(lanes_mode, on="h3_cell", how="left")
    cell_stats = cell_stats.merge(capacity_mode, on="h3_cell", how="left")

# Peak hour ratio
cell_stats["peak_ratio"] = cell_stats["peak_hour_count"] / cell_stats["violation_count"]

# Device-normalized violation count (FIX #2: prevents camera-density bias in λ)
cell_stats["violations_per_device"] = (
    cell_stats["violation_count"] / cell_stats["unique_devices"].clip(lower=1)
)

# ST-DBSCAN flag per cell
cell_stats["has_stdbscan_cluster"] = cell_stats["h3_cell"].isin(stdbscan_cells)

print(f"  Total H3 cells: {len(cell_stats):,}")
print(f"  Violation count range: {cell_stats['violation_count'].min()} – "
      f"{cell_stats['violation_count'].max()}")


# ============================================================================
# STEP 3C: Getis-Ord Gi* hotspot analysis
# ============================================================================
# Identifies statistically significant hot/cold spots with z-scores and
# p-values. Uses device-normalized counts (FIX #2) to avoid camera-density bias.
# ============================================================================
# %%
print("\n" + "=" * 70)
print("STEP 3C: Running Getis-Ord Gi* analysis")
print("=" * 70)

import geopandas as gpd
import libpysal
from esda.getisord import G_Local

# Create GeoDataFrame from cell centers
cell_gdf = gpd.GeoDataFrame(
    cell_stats,
    geometry=gpd.points_from_xy(cell_stats["lon"], cell_stats["lat"]),
    crs="EPSG:4326"
)

# KNN spatial weights (k=8) — more robust than distance-band for irregular spacing
w = libpysal.weights.KNN.from_dataframe(cell_gdf, k=8)
w.transform = "r"  # Row-standardize

# Run Gi* on device-normalized counts
gi_star = G_Local(cell_stats["violations_per_device"].values, w)

cell_stats["gi_z_score"] = gi_star.Zs
cell_stats["gi_p_value"] = gi_star.p_sim

# Significant hotspot: high z-score AND p < threshold
cell_stats["gi_hotspot"] = (
    (cell_stats["gi_z_score"] > 0) &
    (cell_stats["gi_p_value"] < SIGNIFICANCE_LEVEL)
)
cell_stats["gi_coldspot"] = (
    (cell_stats["gi_z_score"] < 0) &
    (cell_stats["gi_p_value"] < SIGNIFICANCE_LEVEL)
)

n_gi_hot  = cell_stats["gi_hotspot"].sum()
n_gi_cold = cell_stats["gi_coldspot"].sum()
print(f"  Gi* results (p < {SIGNIFICANCE_LEVEL}):")
print(f"    Significant hotspots: {n_gi_hot}")
print(f"    Significant coldspots: {n_gi_cold}")
print(f"    Not significant: {len(cell_stats) - n_gi_hot - n_gi_cold}")


# ============================================================================
# STEP 3D: Local Moran's I (LISA)
# ============================================================================
# Quadrants: HH=hotspot cluster, HL=spatial outlier,
#            LH=spatial outlier, LL=coldspot cluster
# ============================================================================
# %%
print("\n" + "=" * 70)
print("STEP 3D: Running Local Moran's I (LISA)")
print("=" * 70)

from esda.moran import Moran_Local

lisa = Moran_Local(cell_stats["violations_per_device"].values, w)

cell_stats["lisa_i"] = lisa.Is
cell_stats["lisa_p"] = lisa.p_sim
cell_stats["lisa_q"] = lisa.q  # 1=HH, 2=LH, 3=LL, 4=HL

quadrant_labels = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
cell_stats["lisa_label"] = cell_stats["lisa_q"].map(quadrant_labels)

cell_stats["lisa_significant"] = cell_stats["lisa_p"] < SIGNIFICANCE_LEVEL
cell_stats["lisa_hh"] = (cell_stats["lisa_q"] == 1) & cell_stats["lisa_significant"]

n_hh = cell_stats["lisa_hh"].sum()
print(f"  LISA results (p < {SIGNIFICANCE_LEVEL}):")
print(f"    HH (hot cluster): {n_hh}")
sig_counts = cell_stats[cell_stats["lisa_significant"]]["lisa_label"].value_counts()
print(f"    All significant: {sig_counts.to_dict()}")


# ============================================================================
# STEP 3E: Cross-validation — combine all three methods
# ============================================================================
# %%
print("\n" + "=" * 70)
print("STEP 3E: Cross-validating hotspot methods")
print("=" * 70)


def classify_hotspot(row):
    """Cross-validate ST-DBSCAN, Gi*, and Moran's I."""
    gi_hot      = row["gi_hotspot"]
    lisa_hot    = row["lisa_hh"]
    stdbscan_hot = row["has_stdbscan_cluster"]

    if gi_hot and lisa_hot and stdbscan_hot:
        return "CONFIRMED_HIGH"
    if sum([gi_hot, lisa_hot, stdbscan_hot]) >= 2:
        return "CONFIRMED"
    if any([gi_hot, lisa_hot, stdbscan_hot]):
        return "EMERGING"
    return "NOT_HOTSPOT"


cell_stats["hotspot_status"] = cell_stats.apply(classify_hotspot, axis=1)

# Combined confidence score (0–1)
cell_stats["hotspot_confidence"] = (
    cell_stats["gi_hotspot"].astype(int) * 0.4 +
    cell_stats["lisa_hh"].astype(int) * 0.3 +
    cell_stats["has_stdbscan_cluster"].astype(int) * 0.3
)

print(f"  Cross-validation results:")
print(f"    {cell_stats['hotspot_status'].value_counts().to_dict()}")


# ============================================================================
# STEP 4: Save hotspot results
# ============================================================================
# %%
print("\n" + "=" * 70)
print("STEP 4: Saving hotspot results")
print("=" * 70)

Path(OUTPUT_DIR).mkdir(exist_ok=True)
cell_stats.to_csv(HOTSPOTS_CSV, index=False)
print(f"  Saved to: {HOTSPOTS_CSV}")

# Merge hotspot status back to violation-level data
df = df.merge(
    cell_stats[["h3_cell", "hotspot_status", "hotspot_confidence", "gi_z_score"]],
    on="h3_cell",
    how="left"
)
df.to_csv(OSM_ENRICHED_CSV, index=False)
print(f"  Updated {OSM_ENRICHED_CSV} with hotspot assignments")


# ============================================================================
# SUMMARY
# ============================================================================
# %%
print("\n" + "=" * 70)
print("SUMMARY — Phase 3 Complete")
print("=" * 70)

confirmed = cell_stats[cell_stats["hotspot_status"].isin(["CONFIRMED_HIGH", "CONFIRMED"])]

print(f"""
  Total H3 cells analyzed: {len(cell_stats):,}

  Hotspot breakdown:
    CONFIRMED_HIGH (all 3 agree): {(cell_stats['hotspot_status']=='CONFIRMED_HIGH').sum()}
    CONFIRMED (2 of 3 agree):     {(cell_stats['hotspot_status']=='CONFIRMED').sum()}
    EMERGING (1 method only):     {(cell_stats['hotspot_status']=='EMERGING').sum()}
    NOT_HOTSPOT:                  {(cell_stats['hotspot_status']=='NOT_HOTSPOT').sum()}

  Top 10 confirmed hotspots by device-normalised density:
""")

if len(confirmed) > 0:
    top10 = confirmed.nlargest(10, "violations_per_device")
    for _, row in top10.iterrows():
        print(f"    {str(row['police_station']):25s} | "
              f"violations={int(row['violation_count']):5d} | "
              f"per_device={row['violations_per_device']:.1f} | "
              f"Gi* z={row['gi_z_score']:.2f} | "
              f"LISA={row['lisa_label']}")


if __name__ == "__main__":
    print("\nPhase 3 complete. Output saved to:", HOTSPOTS_CSV)
