# ============================================================================
# osm_enrichment.py — Phase 2: Road Context Enrichment via OSMnx
# ============================================================================
# Downloads the Bengaluru road network in BULK (one call, ~3 min, no API
# rate limits), then spatial-joins every violation to its nearest road
# segment to extract:
#   - Road type (arterial / collector / local)
#   - Lane count
#   - Road name
#   - Distance to nearest metro station
#   - Distance to nearest commercial area
#
# Uses a bounding-box derived from the data itself (with buffer) instead
# of the full OSM admin polygon — faster and avoids boundary issues.
#
# Run in Colab: Copy each # %% section as a separate cell
# Run locally:  python pipeline/osm_enrichment.py
# ============================================================================

# %% [markdown]
# # Phase 2: OSM Road Context Enrichment

# %%
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, box
import warnings
import os

warnings.filterwarnings("ignore")

try:
    from pipeline.config import (
        CLEANED_CSV, OSM_ENRICHED_CSV, OUTPUT_DIR,
        ROAD_TYPE_MAP, DEFAULT_LANES, ROAD_CAPACITY_PER_LANE
    )
except ImportError:
    CLEANED_CSV = "data/cleaned_violations.csv"
    OSM_ENRICHED_CSV = "data/osm_enriched.csv"
    OUTPUT_DIR = "data"
    ROAD_TYPE_MAP = {
        "motorway": "arterial", "motorway_link": "arterial",
        "trunk": "arterial", "trunk_link": "arterial",
        "primary": "arterial", "primary_link": "arterial",
        "secondary": "collector", "secondary_link": "collector",
        "tertiary": "collector", "tertiary_link": "collector",
        "residential": "local", "living_street": "local",
        "unclassified": "local", "service": "local",
    }
    DEFAULT_LANES = {"arterial": 4, "collector": 2, "local": 2}
    ROAD_CAPACITY_PER_LANE = {"arterial": 1800, "collector": 1200, "local": 800}


# ============================================================================
# STEP 1: Load cleaned data
# ============================================================================
# %%
print("=" * 70)
print("STEP 1: Loading cleaned violation data")
print("=" * 70)

df = pd.read_csv(CLEANED_CSV)
print(f"  Loaded {len(df):,} rows")
print(f"  Lat range: {df['latitude'].min():.4f} – {df['latitude'].max():.4f}")
print(f"  Lon range: {df['longitude'].min():.4f} – {df['longitude'].max():.4f}")


# ============================================================================
# STEP 2: Download Bengaluru road network (BULK — one call)
# ============================================================================
# Uses a bounding box from the data + 0.01° buffer (~1km) for efficiency
# ============================================================================
# %%
print("\n" + "=" * 70)
print("STEP 2: Downloading road network from OpenStreetMap")
print("=" * 70)

import osmnx as ox

# Derive bounding box from violation locations (with buffer)
BUFFER = 0.01  # ~1km buffer around data extent
north = df["latitude"].max() + BUFFER
south = df["latitude"].min() - BUFFER
east = df["longitude"].max() + BUFFER
west = df["longitude"].min() - BUFFER

print(f"  Bounding box: N={north:.4f}, S={south:.4f}, E={east:.4f}, W={west:.4f}")
print(f"  Downloading drivable road network... (this takes ~2-3 min)")

# Download road graph for the bounding box
# OSMnx v2 API: bbox = (west, south, east, north)
G = ox.graph_from_bbox(bbox=(west, south, east, north), network_type="drive")

# Convert to GeoDataFrames
nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)

print(f"  Downloaded {len(edges_gdf):,} road segments")
print(f"  Downloaded {len(nodes_gdf):,} nodes")


# ============================================================================
# STEP 3: Clean road attributes
# ============================================================================
# %%
print("\n" + "=" * 70)
print("STEP 3: Cleaning road attributes")
print("=" * 70)

def clean_highway_tag(val):
    """Extract the primary road type from OSM highway tag.
    Can be a string or list of strings."""
    if isinstance(val, list):
        val = val[0]  # Take the primary type
    return str(val).lower() if pd.notna(val) else "unclassified"

def clean_lanes(val):
    """Parse lane count from OSM data (can be string, list, or number)."""
    if isinstance(val, list):
        val = val[0]
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return np.nan

# Clean the highway and lanes columns
edges_gdf["highway_clean"] = edges_gdf["highway"].apply(clean_highway_tag)
edges_gdf["lanes_clean"] = edges_gdf["lanes"].apply(clean_lanes) if "lanes" in edges_gdf.columns else np.nan

# Map to simplified road type
edges_gdf["road_type"] = edges_gdf["highway_clean"].map(ROAD_TYPE_MAP).fillna("local")

# Fill missing lanes with defaults based on road type
edges_gdf["lanes_clean"] = edges_gdf.apply(
    lambda row: row["lanes_clean"] if pd.notna(row["lanes_clean"])
    else DEFAULT_LANES.get(row["road_type"], 2),
    axis=1
).astype(int)

# Compute road capacity (vehicles/hour)
edges_gdf["road_capacity"] = edges_gdf.apply(
    lambda row: row["lanes_clean"] * ROAD_CAPACITY_PER_LANE.get(row["road_type"], 800),
    axis=1
)

print(f"  Road type distribution:")
print(f"    {edges_gdf['road_type'].value_counts().to_dict()}")
print(f"  Lane distribution:")
print(f"    {edges_gdf['lanes_clean'].value_counts().nlargest(5).to_dict()}")


# ============================================================================
# STEP 4: Spatial join — violations → nearest road segment
# ============================================================================
# %%
print("\n" + "=" * 70)
print("STEP 4: Spatial joining violations to nearest roads")
print("=" * 70)

# Convert violations to GeoDataFrame
violations_gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
    crs="EPSG:4326"
)

# Project to UTM for accurate distance calculations
# Bengaluru is in UTM zone 43N (EPSG:32643)
violations_proj = violations_gdf.to_crs(epsg=32643)
edges_proj = edges_gdf.to_crs(epsg=32643)

# Spatial join to nearest road (this is the key operation)
print(f"  Joining {len(violations_proj):,} violations to nearest road...")
print(f"  (This may take a few minutes for large datasets...)")

joined = gpd.sjoin_nearest(
    violations_proj,
    edges_proj[["geometry", "road_type", "lanes_clean", "road_capacity", "highway_clean"]],
    how="left",
    distance_col="dist_to_road_m"
)

# Remove duplicate rows from spatial join (one violation may match multiple edges)
joined = joined.drop_duplicates(subset=["id"], keep="first")

# Transfer results back to original dataframe
df["road_type"] = joined["road_type"].values
df["lanes"] = joined["lanes_clean"].values
df["road_capacity"] = joined["road_capacity"].values
df["dist_to_road_m"] = joined["dist_to_road_m"].values
df["highway_tag"] = joined["highway_clean"].values

print(f"  Road type assignment:")
print(f"    {df['road_type'].value_counts().to_dict()}")
print(f"  Avg distance to nearest road: {df['dist_to_road_m'].mean():.1f}m")


# ============================================================================
# STEP 5: POI proximity — metro stations, commercial areas
# ============================================================================
# %%
print("\n" + "=" * 70)
print("STEP 5: Computing POI proximity (metro, commercial, schools)")
print("=" * 70)

def compute_nearest_poi_distance(violations_proj, poi_tags, poi_name, bbox_tuple):
    """Download POIs and compute distance from each violation to nearest POI."""
    try:
        # OSMnx v2: features_from_bbox takes bbox=(west, south, east, north)
        pois = ox.features_from_bbox(bbox=bbox_tuple, tags=poi_tags)
        if pois.empty:
            print(f"    No {poi_name} found in area")
            return pd.Series(np.nan, index=violations_proj.index)

        # Get centroids for polygon POIs, keep points as-is
        poi_points = pois.copy()
        poi_points["geometry"] = poi_points["geometry"].centroid
        poi_points = poi_points.to_crs(epsg=32643)

        # For each violation, find distance to nearest POI
        from scipy.spatial import cKDTree

        poi_coords = np.array(
            list(poi_points.geometry.apply(lambda g: (g.x, g.y)))
        )
        viol_coords = np.array(
            list(violations_proj.geometry.apply(lambda g: (g.x, g.y)))
        )

        tree = cKDTree(poi_coords)
        distances, _ = tree.query(viol_coords, k=1)

        print(f"    Found {len(pois)} {poi_name}. Avg distance: {np.mean(distances):.0f}m")
        return pd.Series(distances, index=violations_proj.index)

    except Exception as e:
        print(f"    Warning: Could not fetch {poi_name}: {e}")
        return pd.Series(np.nan, index=violations_proj.index)


# Create bounding box tuple for POI queries (west, south, east, north)
bbox_tuple = (west, south, east, north)

# Metro / Railway stations
df["dist_to_metro_m"] = compute_nearest_poi_distance(
    violations_proj,
    {"railway": ["station", "halt"]},
    "metro/railway stations",
    bbox_tuple
)

# Commercial areas (shops, malls, markets)
df["dist_to_commercial_m"] = compute_nearest_poi_distance(
    violations_proj,
    {"shop": True},
    "commercial POIs",
    bbox_tuple
)

# Schools and hospitals
df["dist_to_school_hospital_m"] = compute_nearest_poi_distance(
    violations_proj,
    {"amenity": ["school", "hospital", "college", "university"]},
    "schools/hospitals",
    bbox_tuple
)

# Create proximity flags (within 300m)
df["near_metro"] = df["dist_to_metro_m"].fillna(9999) < 300
df["near_commercial"] = df["dist_to_commercial_m"].fillna(9999) < 300
df["near_school_hospital"] = df["dist_to_school_hospital_m"].fillna(9999) < 300

print(f"\n  Proximity flags:")
print(f"    Near metro (<300m): {df['near_metro'].sum():,} ({df['near_metro'].mean():.1%})")
print(f"    Near commercial (<300m): {df['near_commercial'].sum():,} ({df['near_commercial'].mean():.1%})")
print(f"    Near school/hospital (<300m): {df['near_school_hospital'].sum():,} ({df['near_school_hospital'].mean():.1%})")


# ============================================================================
# STEP 6: Device density per cell (for FIX #2: λ normalization)
# ============================================================================
# We'll compute device_id count per area later in hotspot_detection.py,
# but we prepare the per-violation device info here.
# ============================================================================
# %%
print("\n" + "=" * 70)
print("STEP 6: Preparing device density metadata")
print("=" * 70)

# Count unique devices per police station (proxy for camera coverage)
station_device_counts = (
    df.groupby("police_station")["device_id"]
    .nunique()
    .rename("station_device_count")
)
df = df.merge(station_device_counts, on="police_station", how="left")

print(f"  Device density by station (top 10):")
for station, count in station_device_counts.nlargest(10).items():
    print(f"    {station}: {count} devices")


# ============================================================================
# STEP 7: Save enriched data
# ============================================================================
# %%
print("\n" + "=" * 70)
print("STEP 7: Saving OSM-enriched dataset")
print("=" * 70)

df.to_csv(OSM_ENRICHED_CSV, index=False)
print(f"  Saved to: {OSM_ENRICHED_CSV}")
print(f"  New columns added: road_type, lanes, road_capacity, dist_to_road_m,")
print(f"    highway_tag, dist_to_metro_m, dist_to_commercial_m,")
print(f"    dist_to_school_hospital_m, near_metro, near_commercial,")
print(f"    near_school_hospital, station_device_count")


# ============================================================================
# SUMMARY
# ============================================================================
# %%
print("\n" + "=" * 70)
print("SUMMARY — Phase 2 Complete")
print("=" * 70)
print(f"""
  Rows:               {len(df):,}
  Road segments used:  {len(edges_gdf):,}
  Road types: {df['road_type'].value_counts().to_dict()}
  Avg lanes: {df['lanes'].mean():.1f}
  Near metro: {df['near_metro'].mean():.1%}
  Near commercial: {df['near_commercial'].mean():.1%}
""")


if __name__ == "__main__":
    print("Phase 2 complete. Output saved to:", OSM_ENRICHED_CSV)
