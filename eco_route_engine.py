"""
eco_route_engine.py
--------------------
Master backend engine for eco-routing in Kolkata.

Features:
  - Loads Kolkata road network (fused_roads.geojson)
  - Predicts emission_factor per edge via CarbonFusionNet ML model
    (falls back to physics formula if model not found)
  - Vehicle count fallback: realistic random values by road type
    (used when no camera image is supplied)
  - Renames carbon_cost -> emission_factor throughout
  - 4 routing strategies: shortest, fastest, lowest emission, balanced
  - Saves route_compare.png
  - Returns JSON-serialisable results for the frontend
"""

import json
import pickle
from scipy.spatial import cKDTree
import math
import os
import random
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import Point, LineString

warnings.filterwarnings("ignore")
import osmnx as ox


# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
KOLKATA_GJ = BASE_DIR / "kolkata" / "fused_roads.geojson"
MODEL_PATH = BASE_DIR / "carbon_fusion_catboost.cbm"

# ── Emission factors (g CO2 / km / vehicle) ───────────────────────────────────
EMISSIONS_G_PER_KM = {
    "car": 120, "motorcycle": 72, "bus": 822,
    "truck": 900, "van": 200, "bicycle": 0, "auto": 85,
}

# ── Realistic fallback vehicle counts by road type ────────────────────────────
FALLBACK_VEHICLE_COUNT = {
    "residential":    (2,  8),
    "tertiary":       (4, 14),
    "secondary":      (8, 20),
    "primary":        (10,25),
    "secondary_link": (4, 11),
    "tertiary_link":  (2,  8),
    "living_street":  (1,  4),
    "unclassified":   (2,  9),
    "busway":         (2,  8),
    "trunk":          (12, 28),
    "trunk_link":     (8, 20),
}
DEFAULT_COUNT_RANGE = (3, 13)

# ── Fallback vehicle-type mix ─────────────────────────────────────────────────
VEHICLE_MIX = {
    "residential":  {"car":0.50,"motorcycle":0.25,"bus":0.02,"truck":0.02,"van":0.05,"bicycle":0.08,"auto":0.08},
    "secondary":    {"car":0.40,"motorcycle":0.18,"bus":0.10,"truck":0.12,"van":0.08,"bicycle":0.04,"auto":0.08},
    "tertiary":     {"car":0.45,"motorcycle":0.22,"bus":0.06,"truck":0.06,"van":0.07,"bicycle":0.06,"auto":0.08},
    "primary":      {"car":0.38,"motorcycle":0.15,"bus":0.12,"truck":0.15,"van":0.08,"bicycle":0.03,"auto":0.09},
    "trunk":        {"car":0.35,"motorcycle":0.12,"bus":0.15,"truck":0.20,"van":0.08,"bicycle":0.02,"auto":0.08},
    "busway":       {"car":0.10,"motorcycle":0.05,"bus":0.70,"truck":0.05,"van":0.05,"bicycle":0.02,"auto":0.03},
}
DEFAULT_MIX = {"car":0.45,"motorcycle":0.20,"bus":0.07,"truck":0.08,"van":0.07,"bicycle":0.06,"auto":0.07}

# ── TomTom Traffic Flow API ───────────────────────────────────────────────────
TOMTOM_FLOW_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
TOMTOM_CACHE_TTL = 120   # seconds — TomTom traffic updates every ~2 min
_tomtom_cache = {}       # { "lat_lon": { "data": {...}, "ts": float } }

# Road capacity estimates (peak-hour vehicles/segment) for vehicle count derivation
ROAD_CAPACITY = {
    "trunk": 250, "trunk_link": 200,
    "primary": 200, "primary_link": 160,
    "secondary": 140, "secondary_link": 110,
    "tertiary": 100, "tertiary_link": 80,
    "residential": 60, "living_street": 30,
    "unclassified": 70, "busway": 50,
}
DEFAULT_ROAD_CAPACITY = 80


def fetch_tomtom_speed(lat: float, lon: float, api_key: str) -> dict:
    """
    Fetch real-time traffic speed for a point from TomTom Flow Segment Data API.
    Returns { current_speed, free_flow_speed, confidence, road_closure } or None on failure.
    Results are cached for TOMTOM_CACHE_TTL seconds.
    """
    cache_key = f"{round(lat, 5)}_{round(lon, 5)}"
    now = time.time()
    cached = _tomtom_cache.get(cache_key)
    if cached and (now - cached["ts"]) < TOMTOM_CACHE_TTL:
        return cached["data"]

    try:
        import requests as _requests
        resp = _requests.get(TOMTOM_FLOW_URL, params={
            "point": f"{lat},{lon}",
            "key": api_key,
            "unit": "KMPH",
        }, timeout=5)
        if resp.status_code != 200:
            return None
        body = resp.json()
        fsd = body.get("flowSegmentData", {})
        result = {
            "current_speed": float(fsd.get("currentSpeed", 0)),
            "free_flow_speed": float(fsd.get("freeFlowSpeed", 0)),
            "confidence": float(fsd.get("confidence", 0)),
            "road_closure": bool(fsd.get("roadClosure", False)),
            "current_travel_time": float(fsd.get("currentTravelTime", 0)),
            "free_flow_travel_time": float(fsd.get("freeFlowTravelTime", 0)),
        }
        _tomtom_cache[cache_key] = {"data": result, "ts": now}
        return result
    except Exception as e:
        print(f"[TOMTOM] API error for ({lat},{lon}): {e}")
        return None


def estimate_vehicle_count_from_speed(current_speed: float, free_flow_speed: float,
                                       highway: str, hour: int = 8) -> dict:
    """
    Estimate vehicle count and breakdown from TomTom speed ratio.
    congestion_ratio = 1 - (current_speed / free_flow_speed)
    vehicle_count ≈ road_capacity × congestion_factor × hour_multiplier
    """
    if free_flow_speed <= 0:
        return fallback_vehicle_count(highway, hour)

    congestion_ratio = max(0, 1 - (current_speed / free_flow_speed))
    # Map congestion to vehicle density: even free-flow has ~10% capacity
    density_factor = 0.10 + congestion_ratio * 0.90

    hw = get_highway_str(highway)
    capacity = ROAD_CAPACITY.get(hw, DEFAULT_ROAD_CAPACITY)

    # Hour-of-day multiplier (same as used in training data)
    hour_mult = {
        0:0.10,1:0.07,2:0.05,3:0.05,4:0.07,5:0.20,
        6:0.60,7:1.00,8:1.00,9:1.50,10:2.50,11:2.50,
        12:2.50,13:2.50,14:1.05,15:1.00,16:1.20,17:1.70,
        18:1.90,19:1.60,20:1.20,21:0.80,22:0.50,23:0.25,
    }.get(hour % 24, 1.0)

    total = max(1, int(round(capacity * density_factor * min(hour_mult, 1.5))))

    # Split by vehicle type mix
    mix = VEHICLE_MIX.get(hw, DEFAULT_MIX)
    types = list(mix.keys())
    probs = np.array([mix[t] for t in types])
    probs /= probs.sum()
    counts = np.random.multinomial(total, probs)
    breakdown = {f"n_{t}": int(c) for t, c in zip(types, counts)}
    co2_per_km = sum(breakdown.get(f"n_{t}", 0) * g
                     for t, g in EMISSIONS_G_PER_KM.items())
    return {"total": total, "co2_per_km_g": co2_per_km, **breakdown}


def fetch_traffic_for_segments(segments: list, api_key: str, hour: int = 8) -> dict:
    """
    Batch-fetch TomTom live traffic for a list of segment dicts.
    Each segment must have: segment_id, midpoint ([lat, lon]), highway.
    Returns { segment_id: { vehicle_info, tomtom_speed, source } }
    """
    results = {}
    fetched = 0
    
    # Cap TomTom live speed queries to at most 10 sampled segments to conserve API credits.
    if len(segments) > 10:
        step = max(1, len(segments) // 10)
        sampled = segments[::step][:10]
    else:
        sampled = segments

    for seg in sampled:
        mid = seg.get("midpoint")
        if not mid or len(mid) != 2:
            continue
        lat, lon = mid
        tt = fetch_tomtom_speed(lat, lon, api_key)
        if tt and tt["current_speed"] > 0:
            vinfo = estimate_vehicle_count_from_speed(
                tt["current_speed"], tt["free_flow_speed"],
                seg.get("highway", "residential"), hour
            )
            results[seg["segment_id"]] = {
                "vehicle_info": vinfo,
                "tomtom_speed": tt,
                "source": "tomtom_live",
            }
            fetched += 1
        # else: segment not in results → will use existing data
    if fetched > 0:
        print(f"[TOMTOM] Fetched live traffic for {fetched}/{len(segments)} segments")
    return results


def enrich_graph_with_live_traffic(
        G: nx.MultiDiGraph,
        route_segments: dict,
        api_key: str,
        hour: int = 8) -> nx.MultiDiGraph:
    """
    Fetch live TomTom traffic for route segments, re-predict emission factors
    with real speed + estimated vehicle count, and return a scaled graph.
    """
    # Collect all unique segments across all routes
    all_segs = {}
    for route_name, segs in route_segments.items():
        for seg in segs:
            sid = seg.get("segment_id")
            if sid and sid not in all_segs and seg.get("midpoint"):
                all_segs[sid] = seg

    if not all_segs:
        return G

    live_data = fetch_traffic_for_segments(list(all_segs.values()), api_key, hour)
    if not live_data:
        return G

    # Build segment_overrides from live data: { "u_v": vehicle_count }
    segment_overrides = {}
    for seg_id, info in live_data.items():
        segment_overrides[seg_id] = info["vehicle_info"]["total"]

    # Also update avg_speed on matching edges in a new graph
    H = nx.MultiDiGraph()
    for node, data in G.nodes(data=True):
        H.add_node(node, **data)

    for u, v, key, data in G.edges(data=True, keys=True):
        new_data = dict(data)
        seg_id = f"{u}_{v}"

        if seg_id in live_data:
            info = live_data[seg_id]
            tt = info["tomtom_speed"]
            vinfo = info["vehicle_info"]

            # Update speed from TomTom
            new_data["avg_speed"] = tt["current_speed"]
            new_data["time"] = new_data["length"] / (tt["current_speed"] * 1000 / 3600) if tt["current_speed"] > 0 else new_data["time"]

            # Update vehicle count
            new_data["vehicle_count"] = vinfo["total"]

            # Re-predict emission factor with real data
            row_dict = {
                "length": new_data["length"],
                "avg_speed_kmph": tt["current_speed"],
                "highway": new_data.get("highway", "residential"),
                "building_density": new_data.get("building_density", 5),
                "vegetation_score": new_data.get("vegetation_score", 2),
                "AQI": new_data.get("AQI", 100),
                "wind_speed_mps": new_data.get("wind_speed_mps", 1),
            }
            new_data["emission_factor"] = predict_emission_factor(row_dict, vinfo, hour=hour)
            new_data["traffic_source"] = "tomtom_live"
        H.add_edge(u, v, key=key, **new_data)

    print(f"[TOMTOM] Enriched graph: {len(live_data)} edges updated with live traffic")
    return H


# ─────────────────────────────────────────────────────────────────────────────
# 1.  VEHICLE COUNT & CO2 HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def get_highway_str(highway_val) -> str:
    """Normalise highway field (may be list-string or plain string)."""
    s = str(highway_val).strip("[]'\" ")
    return s.split(",")[0].strip().strip("'\"") if "," in s else s


def fallback_vehicle_count(highway: str, hour: int = 8) -> dict:
    """
    Generate a realistic vehicle breakdown for a road segment
    when no camera image is available.
    """
    hw = get_highway_str(highway)
    lo, hi = FALLBACK_VEHICLE_COUNT.get(hw, DEFAULT_COUNT_RANGE)

    # Hour-of-day multiplier
    hour_mult = {
        0:0.10,1:0.07,2:0.05,3:0.05,4:0.07,5:0.20,
        6:0.60,7:1.00,8:1.00,9:1.50,10:2.50,11:2.50,
        12:2.50,13:2.50,14:1.05,15:1.00,16:1.20,17:1.70,
        18:1.90,19:1.60,20:1.20,21:0.80,22:0.50,23:0.25,
    }.get(hour % 24, 1.0)

    base = (lo + hi) / 2
    total = max(1, int(round(base * hour_mult + random.gauss(0, (hi - lo) * 0.15))))
    total = max(lo // 2, min(total, hi * 2))

    mix = VEHICLE_MIX.get(hw, DEFAULT_MIX)
    types = list(mix.keys())
    probs = np.array([mix[t] for t in types])
    probs /= probs.sum()
    counts = np.random.multinomial(total, probs)
    breakdown = {f"n_{t}": int(c) for t, c in zip(types, counts)}
    co2_per_km = sum(breakdown.get(f"n_{t}", 0) * g
                     for t, g in EMISSIONS_G_PER_KM.items())
    return {"total": total, "co2_per_km_g": co2_per_km, **breakdown}


def physics_emission_factor(vehicle_count: int, avg_speed: float,
                             length_m: float, co2_per_km: float = None,
                             building_density: int = 5,
                             vegetation_score: int = 2,
                             aqi: int = 100,
                             wind_speed_mps: float = 1.0) -> float:
    """
    Fallback emission_factor calculation (physics formula reverse-engineered
    from real data; corr=0.93 with actual carbon_cost column).
    """
    k = 6.0
    base = (vehicle_count / max(avg_speed, 5)) * length_m * k

    # Congestion extra
    if vehicle_count > 120 and avg_speed < 35:
        ci = (vehicle_count - 120) / 80 * (35 - avg_speed) / 20
        base += base * ci * 0.15

    aqi_pen  = (aqi / 200) * vehicle_count * 0.8
    density  = 1 + building_density * 0.005
    veg_off  = vegetation_score * 0.8
    wind_dispersion = max(0.5, 1 - wind_speed_mps * 0.05) # Higher wind -> more dispersion -> lower localized emission factor

    factor = (base + aqi_pen) * density * wind_dispersion - veg_off
    return max(factor, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  ML MODEL INFERENCE (with graceful fallback)
# ─────────────────────────────────────────────────────────────────────────────
_model_bundle = None   # loaded once

def _load_model_bundle():
    global _model_bundle
    if _model_bundle is not None:
        return _model_bundle
    if not MODEL_PATH.exists():
        return None
    try:
        from catboost import CatBoostRegressor
        model = CatBoostRegressor()
        model.load_model(str(MODEL_PATH))
        _model_bundle = {
            "model": model,
        }
        print("[INFO] CatBoost model loaded successfully.")
        return _model_bundle
    except Exception as e:
        print(f"[WARN] Could not load CatBoost model: {e}. Using physics fallback.")
        return None


def safe_float(val, default=0.0):
    if val is None:
        return default
    if hasattr(val, "__iter__") and not isinstance(val, str):
        items = list(val)
        val = items[0] if items else default
    try:
        import pandas as _pd
        if _pd.isna(val):
            return default
    except Exception:
        pass
    try:
        s_str = str(val).replace("km/h", "").replace(" mph", "").replace("m", "").strip()
        return float(s_str)
    except (ValueError, TypeError):
        return default

def safe_int(val, default=0):
    return int(safe_float(val, float(default)))

def predict_emission_factor(row: dict, vehicle_info: dict, hour: int = None) -> float:
    """
    Predict emission_factor for one road segment.
    Uses ML model if available, otherwise physics formula.
    hour: if provided, used for temporal features; else uses current time.
    """
    bundle = _load_model_bundle()
    if bundle is None:
        return physics_emission_factor(
            vehicle_count=vehicle_info["total"],
            avg_speed=safe_float(row.get("avg_speed_kmph"), 35.0),
            length_m=safe_float(row.get("length"), 100.0),
            co2_per_km=vehicle_info.get("co2_per_km_g"),
            building_density=safe_int(row.get("building_density"), 5),
            vegetation_score=safe_int(row.get("vegetation_score"), 2),
            aqi=safe_int(row.get("AQI"), 100),
        )

    row_input = {
        "length":             safe_float(row.get("length"), 100.0),
        "lanes":              safe_float(row.get("lanes"), 1.0),
        "building_density":   safe_int(row.get("building_density"), 5),
        "vegetation_score":   safe_int(row.get("vegetation_score"), 2),
        "maxspeed":           safe_float(row.get("maxspeed"), 40.0),
        "width":              safe_float(row.get("width"), 8.0),
        "vehicle_count":      vehicle_info["total"],
        "avg_speed_kmph":     safe_float(row.get("avg_speed_kmph"), 35.0),
        "n_car":              vehicle_info.get("n_car", 0),
        "n_motorcycle":       vehicle_info.get("n_motorcycle", 0),
        "n_bus":              vehicle_info.get("n_bus", 0),
        "n_truck":            vehicle_info.get("n_truck", 0),
        "n_van":              vehicle_info.get("n_van", 0),
        "n_bicycle":          vehicle_info.get("n_bicycle", 0),
        "n_auto":             vehicle_info.get("n_auto", 0),
        "co2_per_km_g":       vehicle_info.get("co2_per_km_g", 10000),
        "pm2_5_ugm3":         safe_float(row.get("pm2_5_ugm3"), 40.0),
        "pm10_ugm3":          safe_float(row.get("pm10_ugm3"), 50.0),
        "no2_ugm3":           safe_float(row.get("no2_ugm3"), 1.5),
        "o3_ugm3":            safe_float(row.get("o3_ugm3"), 160.0),
        "so2_ugm3":           safe_float(row.get("so2_ugm3"), 3.0),
        "co_ugm3":            safe_float(row.get("co_ugm3"), 300.0),
        "AQI":                safe_int(row.get("AQI"), 100),
        "openweather_aqi_1to5": safe_int(row.get("openweather_aqi_1to5"), 3),
        "temperature_k":      safe_float(row.get("temperature_k"), 305.0),
        "humidity_pct":       safe_float(row.get("humidity_pct"), 70.0),
        "wind_speed_mps":     safe_float(row.get("wind_speed_mps"), 1.0),
    }
    
    emit = {"n_car": 120, "n_motorcycle": 72, "n_bus": 822,
            "n_truck": 900, "n_van": 200, "n_bicycle": 0, "n_auto": 85}
    for vtype, g in emit.items():
        row_input["co2_" + vtype.replace("n_", "")] = vehicle_info.get(vtype, 0) * g

    row_input["base_cost_feature"] = (
        row_input["vehicle_count"] / max(row_input["avg_speed_kmph"], 5.0)
    ) * row_input["length"]

    # Use actual hour and day-of-week (not hardcoded)
    actual_hour = hour if hour is not None else datetime.now().hour
    actual_dow = datetime.now().weekday()  # 0=Monday
    row_input.update({
        "hour_sin": np.sin(2 * np.pi * actual_hour / 24),
        "hour_cos": np.cos(2 * np.pi * actual_hour / 24),
        "dow_sin":  np.sin(2 * np.pi * actual_dow / 7),
        "dow_cos":  np.cos(2 * np.pi * actual_dow / 7),
        "is_weekend": float(actual_dow >= 5),
    })

    row_input.update({
        "highway":            get_highway_str(row.get("highway", "residential")),
        "weather_description": str(row.get("weather_description", "clear sky")),
        "junction_enc":       str(row.get("junction") or "none"),
        "oneway_enc":         str(row.get("oneway", True)),
        "reversed_enc":       str(row.get("reversed", False)),
    })

    df_input = pd.DataFrame([row_input])

    try:
        log_val = bundle["model"].predict(df_input)[0]
        val = np.expm1(log_val)
        return max(float(val), 1.0)
    except Exception as e:
        print(f"[WARN] ML inference failed ({e}), using physics fallback.")
        return physics_emission_factor(
            vehicle_info["total"],
            float(row.get("avg_speed_kmph", 35)),
            float(row.get("length", 100)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3.  GRAPH BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def download_and_fuse_bbox(olat: float, olon: float, dlat: float, dlon: float, city_name: str = "custom_route"):
    """
    Downloads OSM road network, building footprints, and vegetation polygons for a
    bounding box enclosing (olat, olon) and (dlat, dlon) with a buffer.
    Fetches real-time weather and AQI at the midpoint, fuses them together,
    computes building density, vegetation score, simulated traffic, and saves
    the fused dataset under BASE_DIR / city_name / fused_roads.geojson & fused_roads.csv.
    """
    import osmnx as ox
    city_dir = BASE_DIR / city_name
    city_dir.mkdir(parents=True, exist_ok=True)

    lat_min, lat_max = min(olat, dlat), max(olat, dlat)
    lon_min, lon_max = min(olon, dlon), max(olon, dlon)

    # 0.015 degrees is roughly 1.6km. This is a good routing buffer.
    buffer = 0.015
    lat_min -= buffer
    lat_max += buffer
    lon_min -= buffer
    lon_max += buffer

    print(f"[OSM] Downloading road network for bbox: N={lat_max:.4f}, S={lat_min:.4f}, E={lon_max:.4f}, W={lon_min:.4f}")
    try:
        try:
            # OSMnx 2.x expects bbox=(left, bottom, right, top) i.e. (lon_min, lat_min, lon_max, lat_max)
            G = ox.graph_from_bbox(bbox=(lon_min, lat_min, lon_max, lat_max), network_type="drive")
        except TypeError:
            G = ox.graph_from_bbox(north=lat_max, south=lat_min, east=lon_max, west=lon_min, network_type="drive")
    except Exception as e:
        print(f"[ERROR] Failed to download OSM road network: {e}")
        raise RuntimeError(f"Failed to fetch road network from OpenStreetMap. Error: {e}")

    # Project graph to EPSG:3857 for metric distance buffers
    G_proj = ox.project_graph(G, to_crs="EPSG:3857")
    nodes, edges = ox.graph_to_gdfs(G_proj)

    # Fetch buildings
    tags_b = {"building": True}
    print("[OSM] Downloading building footprints...")
    try:
        try:
            buildings = ox.features_from_bbox(bbox=(lon_min, lat_min, lon_max, lat_max), tags=tags_b)
        except TypeError:
            buildings = ox.features_from_bbox(north=lat_max, south=lat_min, east=lon_max, west=lon_min, tags=tags_b)
        if not buildings.empty:
            buildings = buildings.to_crs(epsg=3857)
    except Exception as e:
        print(f"[WARN] Failed to download buildings: {e}. Using empty building footprint dataset.")
        buildings = gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:3857")

    # Fetch vegetation
    tags_v = {
        "landuse": ["forest", "grass", "meadow"], 
        "natural": ["wood", "tree", "grassland"],
        "leisure": ["park", "garden"]
    }
    print("[OSM] Downloading vegetation polygons...")
    try:
        try:
            vegetation = ox.features_from_bbox(bbox=(lon_min, lat_min, lon_max, lat_max), tags=tags_v)
        except TypeError:
            vegetation = ox.features_from_bbox(north=lat_max, south=lat_min, east=lon_max, west=lon_min, tags=tags_v)
        if not vegetation.empty:
            vegetation = vegetation.to_crs(epsg=3857)
    except Exception as e:
        print(f"[WARN] Failed to download vegetation: {e}. Using empty vegetation dataset.")
        vegetation = gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:3857")

    # Compute building density (50m buffer) using vectorized spatial join
    print("[FUSION] Computing building density (vectorized)...")
    building_counts = [0] * len(edges)
    if not buildings.empty and "geometry" in buildings.columns:
        try:
            buffered_edges = gpd.GeoDataFrame(geometry=edges.geometry.buffer(50), crs=edges.crs).reset_index()
            joined = gpd.sjoin(buildings, buffered_edges, how="inner", predicate="intersects")
            if not joined.empty:
                counts = joined["index_right"].value_counts()
                building_counts = [int(counts.get(i, 0)) for i in range(len(edges))]
        except Exception as e:
            print(f"[WARN] Vectorized building density failed ({e}). Falling back to loop.")
            try:
                buildings_sindex = buildings.sindex
                building_counts = []
                for idx, road in edges.iterrows():
                    try:
                        buffer_geom = road.geometry.buffer(50)
                        possible_matches_index = list(buildings_sindex.intersection(buffer_geom.bounds))
                        possible_matches = buildings.iloc[possible_matches_index]
                        nearby_buildings = possible_matches[possible_matches.intersects(buffer_geom)]
                        building_counts.append(len(nearby_buildings))
                    except Exception:
                        building_counts.append(0)
            except Exception:
                building_counts = [0] * len(edges)
    edges["building_density"] = building_counts

    # Compute vegetation score (50m buffer) using vectorized spatial join
    print("[FUSION] Computing vegetation score (vectorized)...")
    veg_counts = [0] * len(edges)
    if not vegetation.empty and "geometry" in vegetation.columns:
        try:
            buffered_edges = gpd.GeoDataFrame(geometry=edges.geometry.buffer(50), crs=edges.crs).reset_index()
            joined = gpd.sjoin(vegetation, buffered_edges, how="inner", predicate="intersects")
            if not joined.empty:
                counts = joined["index_right"].value_counts()
                veg_counts = [int(counts.get(i, 0)) for i in range(len(edges))]
        except Exception as e:
            print(f"[WARN] Vectorized vegetation score failed ({e}). Falling back to loop.")
            try:
                vegetation_sindex = vegetation.sindex
                veg_counts = []
                for idx, road in edges.iterrows():
                    try:
                        buffer_geom = road.geometry.buffer(50)
                        possible_matches_index = list(vegetation_sindex.intersection(buffer_geom.bounds))
                        possible_matches = vegetation.iloc[possible_matches_index]
                        veg = possible_matches[possible_matches.intersects(buffer_geom)]
                        veg_counts.append(len(veg))
                    except Exception:
                        veg_counts.append(0)
            except Exception:
                veg_counts = [0] * len(edges)
    edges["vegetation_score"] = veg_counts

    # Fetch weather and AQI
    mid_lat = (olat + dlat) / 2
    mid_lon = (olon + dlon) / 2
    api_key = os.environ.get("OPEN_WEATHER_API_KEY", "c87754e82558c2df5352f5a899078d0d")

    print(f"[API] Fetching weather & AQI for coordinates: {mid_lat:.4f}, {mid_lon:.4f}")
    wind_speed = 3.0
    wind_direction = 180
    try:
        import requests as _requests
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={mid_lat}&lon={mid_lon}&appid={api_key}"
        resp = _requests.get(weather_url, timeout=5).json()
        if "wind" in resp:
            wind_speed = float(resp["wind"].get("speed", 3.0))
            wind_direction = int(resp["wind"].get("deg", 180))
    except Exception as e:
        print(f"[WARN] Weather API failed: {e}. Using defaults (wind_speed=3.0, wind_direction=180).")

    edges["wind_speed_mps"] = wind_speed
    edges["wind_direction"] = wind_direction
    edges["temperature_k"] = 300.0
    edges["humidity_pct"] = 65.0
    edges["openweather_aqi_1to5"] = 3

    AQI = 100
    try:
        import requests as _requests
        aqi_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={mid_lat}&lon={mid_lon}&appid={api_key}"
        aqi_data = _requests.get(aqi_url, timeout=5).json()
        pm25 = aqi_data["list"][0]["components"]["pm2_5"]
        if pm25 <= 12.0:
            AQI = (50 / 12.0) * pm25
        elif pm25 <= 35.4:
            AQI = 50 + ((100 - 50) / (35.4 - 12.1)) * (pm25 - 12.1)
        elif pm25 <= 55.4:
            AQI = 100 + ((150 - 100) / (55.4 - 35.5)) * (pm25 - 35.5)
        elif pm25 <= 150.4:
            AQI = 150 + ((200 - 150) / (150.4 - 55.5)) * (pm25 - 55.5)
        else:
            AQI = 200 + ((300 - 200) / (250.4 - 150.5)) * min(pm25 - 150.5, 99.9)
        AQI = int(AQI)
    except Exception as e:
        print(f"[WARN] AQI API failed: {e}. Using default AQI=100.")
    edges["AQI"] = AQI

    # Simulate traffic data
    edges["vehicle_count"] = np.random.randint(10, 80, len(edges))

    avg_speeds = []
    for idx, row in edges.iterrows():
        maxspeed = row.get("maxspeed")
        if maxspeed:
            try:
                if isinstance(maxspeed, list):
                    maxspeed = maxspeed[0]
                speed_val = float(str(maxspeed).replace("km/h", "").replace(" mph", "").strip())
            except Exception:
                speed_val = 40.0
        else:
            speed_val = 40.0
        avg_speeds.append(max(5.0, speed_val - np.random.randint(5, 15)))
    edges["avg_speed_kmph"] = avg_speeds

    # Compute initial emission factor
    emission_factor = 0.12
    edges["carbon_cost"] = (
        edges["length"] * emission_factor * edges["vehicle_count"]
        + edges["building_density"] * 5
        - edges["vegetation_score"] * 3
        + edges["AQI"] * 0.2
    )

    # Save to file
    print("[FUSION] Saving dynamically fused network...")
    edges_to_save = edges.copy()
    if "u" not in edges_to_save.columns:
        edges_to_save = edges_to_save.reset_index()

    if "index" in edges_to_save.columns:
        edges_to_save = edges_to_save.drop(columns=["index"])

    geojson_path = city_dir / "fused_roads.geojson"
    csv_path = city_dir / "fused_roads.csv"

    edges_to_save.to_file(str(geojson_path), driver="GeoJSON")
    edges_to_save.to_csv(str(csv_path), index=False)
    print(f"[FUSION] Dynamic fusion successful: saved files under {city_dir}")


GRAPH_CACHE = {}
KDTREE_CACHE = {}

def get_city_dir(city: str) -> Path:
    """Get directory path for a city. Check cities/city first, then fallback to root/city."""
    city_key = city.lower().strip()
    cities_dir = BASE_DIR / "cities" / city_key
    if cities_dir.exists():
        return cities_dir
    local_dir = BASE_DIR / city_key
    if local_dir.exists():
        return local_dir
    return cities_dir

def safe_scalar(val, default=None):
    """Return a scalar even if val is a list/array (GeoJSON multi-value columns)."""
    if val is None:
        return default
    if hasattr(val, "__iter__") and not isinstance(val, str):
        items = list(val)
        return items[0] if items else default
    try:
        if pd.isna(val):
            return default
    except (TypeError, ValueError):
        pass
    return val

def rebuild_graph_from_geojson(geojson_path: Path, use_ml: bool, hour: int, use_csv_emission: bool) -> nx.MultiDiGraph:
    """Load GeoJSON and reconstruct the NetworkX MultiDiGraph, populating node coordinates."""
    print(f"[INFO] Parsing GeoJSON: {geojson_path}")
    gdf = gpd.read_file(str(geojson_path))
    print(f"[INFO] {len(gdf):,} road segments loaded. Extracting node coordinates...")
    
    # Project edge geometries to EPSG:4326 to get node lat/lon coordinates
    gdf_gps = gdf.to_crs("EPSG:4326")
    
    G = nx.MultiDiGraph()
    
    for idx, row in gdf.iterrows():
        u = row["u"]
        v = row["v"]
        
        # Extract GPS coordinates for nodes (start and end coordinates of LineString in EPSG:4326)
        geom_gps = gdf_gps.at[idx, "geometry"]
        if geom_gps and geom_gps.geom_type == "LineString":
            coords_gps = list(geom_gps.coords)
            if coords_gps:
                G.add_node(u, lon=coords_gps[0][0], lat=coords_gps[0][1])
                G.add_node(v, lon=coords_gps[-1][0], lat=coords_gps[-1][1])
                
        hw  = get_highway_str(safe_scalar(row.get("highway"), "residential"))
        spd_raw = safe_scalar(row.get("avg_speed_kmph"), 30)
        spd = max(float(spd_raw) if spd_raw is not None else 30.0, 5.0)
        lng_raw = safe_scalar(row.get("length"), 50)
        lng = float(lng_raw) if lng_raw is not None else 50.0
        
        # emission_factor
        cc_raw = safe_scalar(row.get("carbon_cost"))
        if use_csv_emission and cc_raw is not None:
            try:
                emission_factor = float(cc_raw)
            except (ValueError, TypeError):
                emission_factor = None
        else:
            emission_factor = None
            
        if emission_factor is None or emission_factor <= 0:
            vinfo = fallback_vehicle_count(hw, hour)
            emission_factor = predict_emission_factor(dict(row), vinfo)
            
        vc_raw = safe_scalar(row.get("vehicle_count"), 60)
        vc = int(float(vc_raw)) if vc_raw is not None else 60
        
        bd_raw = safe_scalar(row.get("building_density"), 5)
        vs_raw = safe_scalar(row.get("vegetation_score"), 2)
        aq_raw = safe_scalar(row.get("AQI"), 100)
        ws_raw = safe_scalar(row.get("wind_speed_mps"), 1)
        
        edge_data = {
            "length":           lng,
            "emission_factor":  emission_factor,
            "vehicle_count":    vc,
            "avg_speed":        spd,
            "time":             lng / (spd * 1000 / 3600),
            "highway":          hw,
            "name":             str(safe_scalar(row.get("name"), "") or ""),
            "geometry":         row.get("geometry"),
            "building_density": int(float(bd_raw)) if bd_raw is not None else 5,
            "vegetation_score": int(float(vs_raw)) if vs_raw is not None else 2,
            "AQI":              int(float(aq_raw)) if aq_raw is not None else 100,
            "wind_speed_mps":   float(ws_raw) if ws_raw is not None else 1.0,
        }
        G.add_edge(u, v, **edge_data)
        
    print(f"[INFO] Completed graph building. Nodes: {G.number_of_nodes():,}, Edges: {G.number_of_edges():,}")
    return G

def get_or_build_kdtree(city_name: str, G: nx.MultiDiGraph):
    """Get or build scipy.spatial.cKDTree for a city's graph nodes in lat/lon coordinates."""
    global KDTREE_CACHE
    city_key = city_name.lower().strip()
    if city_key in KDTREE_CACHE:
        return KDTREE_CACHE[city_key]
        
    node_coords = []
    node_ids = []
    for n, data in G.nodes(data=True):
        lat = data.get("lat") or data.get("y")
        lon = data.get("lon") or data.get("x")
        if lat is not None and lon is not None:
            node_coords.append((lat, lon))
            node_ids.append(n)
            
    if node_coords:
        kdtree = cKDTree(node_coords)
        KDTREE_CACHE[city_key] = (kdtree, node_ids)
        print(f"[INFO] Built KDTree for {city_name} with {len(node_ids):,} nodes.")
        return kdtree, node_ids
        
    return None, None

def build_emission_graph(city: str = "kolkata",
                          use_ml: bool = True,
                          hour: int = 8,
                          use_csv_emission: bool = True,
                          vehicle_override: int = None) -> nx.MultiDiGraph:
    """
    Build or retrieve a cached directed road graph where edge weight = emission_factor.
    """
    global GRAPH_CACHE
    city_key = city.lower().strip()
    
    # Check if base graph exists in memory
    if city_key in GRAPH_CACHE:
        base_G = GRAPH_CACHE[city_key]
    else:
        # Load from disk (gpickle or geojson)
        city_dir = get_city_dir(city_key)
        gpickle_path = city_dir / "fused_roads.gpickle"
        geojson_path = city_dir / "fused_roads.geojson"
        
        if gpickle_path.exists():
            print(f"[INFO] Loading cached graph from binary: {gpickle_path}")
            try:
                with open(gpickle_path, "rb") as f:
                    base_G = pickle.load(f)
                print(f"[INFO] Loaded binary cache with {base_G.number_of_nodes():,} nodes.")
            except Exception as e:
                print(f"[WARN] Failed to load binary cache: {e}. Rebuilding from GeoJSON.")
                base_G = None
        else:
            base_G = None
            
        if base_G is None:
            if not geojson_path.exists():
                print(f"[WARN] No network data found for city '{city}' at '{geojson_path}'.")
                return nx.MultiDiGraph()
                
            base_G = rebuild_graph_from_geojson(geojson_path, use_ml, hour, use_csv_emission)
            
            # Save to binary gpickle cache
            try:
                city_dir.mkdir(parents=True, exist_ok=True)
                with open(gpickle_path, "wb") as f:
                    pickle.dump(base_G, f)
                print(f"[INFO] Saved graph to binary cache: {gpickle_path}")
            except Exception as e:
                print(f"[WARN] Failed to save binary cache: {e}")
                
        GRAPH_CACHE[city_key] = base_G
        
    # Apply vehicle override (YOLO overrides) in-memory if set
    if vehicle_override is not None:
        G = base_G.copy()
        for u, v, data in G.edges(data=True):
            stored_vc = data.get("vehicle_count", 60)
            if stored_vc > 0:
                scale = vehicle_override / stored_vc
                data["emission_factor"] *= max(0.3, min(scale, 3.0))
                data["vehicle_count"] = vehicle_override
        return G
        
    return base_G


# ─────────────────────────────────────────────────────────────────────────────
# 4.  ROUTING STRATEGIES
# ─────────────────────────────────────────────────────────────────────────────
def _safe_route(G, origin, dest, weight):
    try:
        return nx.shortest_path(G, origin, dest, weight=weight)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def compute_all_routes(G, origin, destination):
    """Run all 4 routing strategies. Returns dict of {strategy_name: route}."""
    # Add composite weight for balanced routing
    for u, v, data in G.edges(data=True):
        n_ef  = data["emission_factor"] / 5000
        n_t   = data["time"] / 300
        n_d   = data["length"] / 1000
        data["composite"] = 0.5 * n_ef + 0.3 * n_t + 0.2 * n_d

    strategies = {
        "Shortest Distance":    ("length",         "#3B82F6"),
        "Fastest Time":         ("time",            "#F59E0B"),
        "Lowest Emission":      ("emission_factor", "#10B981"),
        "Balanced":             ("composite",       "#8B5CF6"),
    }
    routes = {}
    for name, (weight, _) in strategies.items():
        r = _safe_route(G, origin, destination, weight)
        if r:
            routes[name] = r
    return routes, strategies


def route_stats(G, route, vehicle_override=None, vehicle_type="car"):
    """Compute stats for a route. Returns dict."""
    dist = time_ = ef = veh = 0
    trip_co2 = 0.0
    speeds = []
    
    # Emission factor per km for user's vehicle
    user_emissions_base = EMISSIONS_G_PER_KM.get(vehicle_type, 120)
    if vehicle_type == "ev" or vehicle_type == "bicycle":
        user_emissions_base = 0.0
        
    for i in range(len(route) - 1):
        u, v = route[i], route[i + 1]
        if v not in G[u]:
            continue
        ed = min(G[u][v].values(), key=lambda x: x["emission_factor"])

        emission = ed["emission_factor"]
        vc = ed.get("vehicle_count", 60)

        if vehicle_override is not None and emission > 0 and vc > 0:
            scale = vehicle_override / vc
            emission *= max(0.3, min(scale, 3.0))
            veh_to_add = vehicle_override
        else:
            veh_to_add = vc

        length_m = ed["length"]
        dist  += length_m
        ef    += emission
        veh   += veh_to_add
        
        # Calculate vehicle travel speed
        avg_spd = ed.get("avg_speed", 30)
        
        # Speed adjustment based on vehicle type
        if vehicle_type == "motorcycle":
            travel_speed = min(avg_spd * 1.15, 60.0) # motorcycles filter through traffic
        elif vehicle_type == "bicycle":
            travel_speed = 15.0 # steady bicycle speed
        elif vehicle_type in ["bus", "truck"]:
            travel_speed = min(avg_spd * 0.8, 40.0) # slower, capped speed
        else: # car, ev
            travel_speed = avg_spd
            
        travel_speed = max(travel_speed, 5.0) # never less than 5 km/h
        speeds.append(travel_speed)
        
        # Time on this segment (seconds)
        seg_time = length_m / (travel_speed * 1000 / 3600)
        time_ += seg_time
        
        # Trip CO2 emissions for this segment
        # Congestion factor based on speed
        if travel_speed < 15.0:
            congestion_factor = 1.8
        elif travel_speed < 30.0:
            congestion_factor = 1.3
        else:
            congestion_factor = 1.0
            
        seg_co2 = (length_m / 1000.0) * user_emissions_base * congestion_factor
        trip_co2 += seg_co2
        
    dist_km = dist / 1000.0
    time_min = time_ / 60.0
    
    return {
        "distance_km":    round(dist_km, 3),
        "time_min":       round(time_min, 2),
        "emission_factor": round(ef, 1),
        "trip_co2_g":     round(trip_co2, 1),
        "avg_vehicles":   round(veh / max(len(route) - 1, 1), 1),
        "avg_speed_kmh":  round(sum(speeds) / max(len(speeds), 1), 1),
        "segments":       len(route) - 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4b.  SCALED GRAPH BUILDER (for route overrides / segment overrides)
# ─────────────────────────────────────────────────────────────────────────────
MAJOR_HIGHWAYS = {
    "primary", "secondary", "trunk", "tertiary",
    "primary_link", "secondary_link", "trunk_link", "tertiary_link",
}


def get_route_edge_scales(G, route: list, override_vehicles: int) -> dict:
    """
    For every edge (u,v) in a route, compute a scale factor based on:
        scale = override_vehicles / stored_vehicle_count
    Returns { (u,v): scale_factor }
    """
    scales = {}
    for i in range(len(route) - 1):
        u, v = route[i], route[i + 1]
        if v not in G[u]:
            continue
        ed = min(G[u][v].values(), key=lambda x: x["emission_factor"])
        stored_vc = max(ed.get("vehicle_count", 60), 1)
        scale = max(0.3, min(override_vehicles / stored_vc, 5.0))
        scales[(u, v)] = scale
    return scales


def build_scaled_graph_with_overrides(
        G: nx.MultiDiGraph,
        segment_overrides: Optional[dict] = None,
        route_edge_scales: Optional[dict] = None) -> nx.MultiDiGraph:
    """
    Return a NEW graph where emission_factor values are scaled by:
      - segment_overrides : { "u_v" : detected_vehicle_count }
      - route_edge_scales : { (u,v) : scale_factor }  (from per-route upload)

    This is the correct fix: Dijkstra on this graph finds genuinely
    different paths (not just different stats).
    """
    H = nx.MultiDiGraph()

    # Copy all nodes
    for node, data in G.nodes(data=True):
        H.add_node(node, **data)

    for u, v, key, data in G.edges(data=True, keys=True):
        new_ef = data["emission_factor"]
        stored_vc = max(data.get("vehicle_count", 60), 1)
        target_vc = stored_vc
        has_override = False

        # Segment-level override takes priority
        seg_id = f"{u}_{v}"
        if segment_overrides and seg_id in segment_overrides:
            target_vc = segment_overrides[seg_id]
            has_override = True

        # Route-level override (only if no segment override for this edge)
        elif route_edge_scales and (u, v) in route_edge_scales:
            target_vc = int(stored_vc * route_edge_scales[(u, v)])
            has_override = True

        new_data = dict(data)
        if has_override:
            scale = target_vc / max(stored_vc, 1)
            new_data["emission_factor"] = new_ef * max(0.3, min(scale, 10.0))
            new_data["vehicle_count"] = target_vc
        else:
            new_data["emission_factor"] = new_ef

        # Also update the composite weight so balanced routing is affected too
        H.add_edge(u, v, key=key, **new_data)

    return H


def get_route_segments(G, route: list, top_n: int = 30) -> list:
    """
    Extract the top `top_n` highest-emission named road segments along a route.
    Only considers major highway types (primary / secondary / trunk / tertiary).
    Returns a list of dicts with segment metadata for the frontend UI.
    """
    try:
        from pyproj import Transformer
        _tr = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    except Exception:
        _tr = None

    segments = []
    seen_names: set = set()

    for i in range(len(route) - 1):
        u, v = route[i], route[i + 1]
        if v not in G[u]:
            continue
        ed = min(G[u][v].values(), key=lambda x: x["emission_factor"])

        hw = ed.get("highway", "residential")
        if hw not in MAJOR_HIGHWAYS:
            continue

        raw_name = (ed.get("name") or "").strip()
        seg_id   = f"{u}_{v}"

        # Deduplicate by road name (keep the highest-emission occurrence)
        dedup_key = raw_name if raw_name else seg_id
        if raw_name and dedup_key in seen_names:
            continue
        seen_names.add(dedup_key)

        # Midpoint lat/lon for map marker
        midpoint = None
        seg_coords = []
        geom = ed.get("geometry")
        if geom and geom.geom_type == "LineString" and _tr:
            try:
                mid = geom.interpolate(0.5, normalized=True)
                lon_, lat_ = _tr.transform(mid.x, mid.y)
                midpoint = [round(lat_, 6), round(lon_, 6)]
                
                # Extract coordinates
                for x, y in geom.coords:
                    lon_c, lat_c = _tr.transform(x, y)
                    seg_coords.append([round(lat_c, 6), round(lon_c, 6)])
            except Exception:
                pass

        # Determine speed category
        avg_spd = float(ed.get("avg_speed", 30))
        if avg_spd < 15.0:
            speed_category = "congested"
        elif avg_spd < 30.0:
            speed_category = "slow"
        elif avg_spd < 45.0:
            speed_category = "moderate"
        else:
            speed_category = "free"

        segments.append({
            "segment_id":      seg_id,
            "edge_u":          u,
            "edge_v":          v,
            "name":            raw_name or f"{hw.replace('_', ' ').title()} Segment",
            "highway":         hw,
            "emission_factor": round(float(ed.get("emission_factor", 0)), 1),
            "vehicle_count":   int(ed.get("vehicle_count", 60)),
            "length_m":        round(float(ed.get("length", 0)), 1),
            "avg_speed_kmh":   round(avg_spd, 1),
            "midpoint":        midpoint,
            "coords":          seg_coords,
            "speed_category":  speed_category,
        })

    segments.sort(key=lambda s: s["emission_factor"], reverse=True)
    return segments[:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# 5.  NEAREST NODE FINDER
# ─────────────────────────────────────────────────────────────────────────────
def nearest_node(G, city: str, lat: float, lon: float) -> int:
    """Find graph node closest to (lat, lon) using cached scipy.spatial.cKDTree."""
    city_key = city.lower().strip()
    kdtree, node_ids = get_or_build_kdtree(city_key, G)
    if kdtree is not None:
        dist, idx = kdtree.query((lat, lon))
        return int(node_ids[idx])
        
    # Fallback to linear scan of nodes in G if KDTree not built
    best_node = None
    best_dist = float("inf")
    for n, data in G.nodes(data=True):
        n_lat = data.get("lat") or data.get("y")
        n_lon = data.get("lon") or data.get("x")
        if n_lat is not None and n_lon is not None:
            d = (n_lat - lat) ** 2 + (n_lon - lon) ** 2
            if d < best_dist:
                best_dist = d
                best_node = n
                
    if best_node is not None:
        return int(best_node)
        
    # Final fallback: return first node
    if G.number_of_nodes() == 0:
        return None
    return list(G.nodes)[0]


# ─────────────────────────────────────────────────────────────────────────────
# 6.  ROUTE COMPARISON PLOT
# ─────────────────────────────────────────────────────────────────────────────
def plot_route_comparison(G, routes: dict, strategies: dict,
                          stats: dict, origin, destination,
                          save_path: str = "route_compare_kolkata.png"):
    """Generate a rich route comparison figure."""
    fig = plt.figure(figsize=(18, 10), facecolor="#0F172A")

    # ── Map panel (left) ──────────────────────────────────────────────────
    ax_map = fig.add_axes([0.01, 0.05, 0.55, 0.90])
    ax_map.set_facecolor("#1E293B")
    ax_map.tick_params(colors="#94A3B8")
    for spine in ax_map.spines.values():
        spine.set_edgecolor("#334155")

    # Draw all background roads (grey)
    drawn = set()
    for u, v, data in G.edges(data=True):
        geom = data.get("geometry")
        if geom and (u, v) not in drawn:
            if geom.geom_type == "LineString":
                xs, ys = geom.xy
                ax_map.plot(xs, ys, color="#334155", linewidth=0.4, alpha=0.6, zorder=1)
            drawn.add((u, v))

    # Draw each route
    legend_patches = []
    for name, (weight, color) in strategies.items():
        route = routes.get(name)
        if not route:
            continue
        xs_all, ys_all = [], []
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            if v not in G[u]:
                continue
            ed = min(G[u][v].values(), key=lambda x: x["emission_factor"])
            geom = ed.get("geometry")
            if geom and geom.geom_type == "LineString":
                xs, ys = geom.xy
                ax_map.plot(xs, ys, color=color, linewidth=3.5,
                            alpha=0.92, zorder=5, solid_capstyle="round")
                xs_all.extend(xs); ys_all.extend(ys)

        st = stats.get(name, {})
        lbl = (f"{name}\n"
               f"  {st.get('distance_km','?')} km | "
               f"{st.get('time_min','?')} min | "
               f"EF={st.get('emission_factor','?'):.0f}")
        legend_patches.append(mpatches.Patch(color=color, label=lbl))

    # Origin / destination markers
    def get_node_coords(node_id):
        for u, v, data in G.edges(data=True):
            geom = data.get("geometry")
            if geom and geom.geom_type == "LineString":
                if u == node_id:
                    return geom.coords[0]
                if v == node_id:
                    return geom.coords[-1]
        return None

    oc = get_node_coords(origin)
    dc = get_node_coords(destination)
    if oc:
        ax_map.scatter(*oc, s=200, c="#22D3EE", zorder=10, edgecolors="white", linewidths=1.5)
        ax_map.annotate("  START", oc, color="#22D3EE", fontsize=9, fontweight="bold", zorder=11)
    if dc:
        ax_map.scatter(*dc, s=200, c="#F43F5E", zorder=10, edgecolors="white", linewidths=1.5)
        ax_map.annotate("  END", dc, color="#F43F5E", fontsize=9, fontweight="bold", zorder=11)

    city_title = Path(save_path).stem.replace("route_compare_", "").title()
    ax_map.set_title(f"{city_title} Eco-Routing — Route Comparison", color="white",
                     fontsize=14, fontweight="bold", pad=10)
    ax_map.legend(handles=legend_patches, loc="lower left",
                  facecolor="#1E293B", edgecolor="#475569",
                  labelcolor="white", fontsize=8.5)
    ax_map.set_aspect("equal")

    # ── Stats panel (right) ───────────────────────────────────────────────
    ax_stats = fig.add_axes([0.58, 0.05, 0.40, 0.90])
    ax_stats.set_facecolor("#1E293B")
    ax_stats.axis("off")

    ax_stats.text(0.5, 0.97, "Route Statistics", color="white",
                  fontsize=15, fontweight="bold", ha="center", va="top",
                  transform=ax_stats.transAxes)
    ax_stats.text(0.5, 0.92, f"{city_title} Road Network  |  Emission Factor = g CO\u2082/segment",
                  color="#94A3B8", fontsize=8, ha="center", va="top",
                  transform=ax_stats.transAxes)

    # Table
    col_labels = ["Strategy", "Dist (km)", "Time (min)", "Emission\nFactor", "Segments"]
    col_widths  = [0.32, 0.17, 0.17, 0.17, 0.14]
    y = 0.85
    # header row
    x = 0.01
    for lbl, w in zip(col_labels, col_widths):
        ax_stats.text(x + w / 2, y, lbl, color="#94A3B8", fontsize=8.5,
                      fontweight="bold", ha="center", va="center",
                      transform=ax_stats.transAxes)
        x += w
    ax_stats.plot([0.01, 0.99], [y - 0.03, y - 0.03], color="#475569",
                  linewidth=0.8, transform=ax_stats.transAxes, clip_on=False)
    y -= 0.07

    best_ef = min((stats[n]["emission_factor"] for n in stats), default=0)
    for name, (weight, color) in strategies.items():
        st = stats.get(name)
        if not st:
            continue
        bg_color = "#1E3A2F" if st["emission_factor"] == best_ef else "#1E293B"
        rect = plt.Rectangle((0.0, y - 0.055), 1.0, 0.075,
                               facecolor=bg_color, transform=ax_stats.transAxes,
                               clip_on=False, zorder=0)
        ax_stats.add_patch(rect)

        vals = [name, st["distance_km"], st["time_min"],
                f"{st['emission_factor']:,.0f}", st["segments"]]
        x = 0.01
        for val, w in zip(vals, col_widths):
            ax_stats.text(x + w / 2, y - 0.015, str(val),
                          color=color, fontsize=9, ha="center", va="center",
                          transform=ax_stats.transAxes, fontweight="bold")
            x += w
        y -= 0.085

    # Savings callout
    y -= 0.02
    ax_stats.plot([0.01, 0.99], [y, y], color="#475569",
                  linewidth=0.8, transform=ax_stats.transAxes, clip_on=False)
    y -= 0.06

    if "Shortest Distance" in stats and "Lowest Emission" in stats:
        saved_ef = (stats["Shortest Distance"]["emission_factor"]
                    - stats["Lowest Emission"]["emission_factor"])
        saved_pct = saved_ef / max(stats["Shortest Distance"]["emission_factor"], 1) * 100
        ax_stats.text(0.5, y, f"Eco-route saves {saved_ef:,.0f} g CO\u2082  ({saved_pct:.1f}%)",
                      color="#10B981", fontsize=12, fontweight="bold",
                      ha="center", va="center", transform=ax_stats.transAxes)
        y -= 0.06
        extra_km = (stats["Lowest Emission"]["distance_km"]
                    - stats["Shortest Distance"]["distance_km"])
        extra_min = (stats["Lowest Emission"]["time_min"]
                     - stats["Shortest Distance"]["time_min"])
        ax_stats.text(0.5, y,
                      f"Extra distance: +{extra_km:.2f} km  |  Extra time: +{extra_min:.1f} min",
                      color="#94A3B8", fontsize=9, ha="center", va="center",
                      transform=ax_stats.transAxes)

    # Emission factor breakdown bar chart
    y -= 0.10
    names_  = list(stats.keys())
    efs_    = [stats[n]["emission_factor"] for n in names_]
    colors_ = [strategies[n][1] for n in names_]
    bars_ax = fig.add_axes([0.595, y - 0.16, 0.385, 0.18])
    bars_ax.set_facecolor("#0F172A")
    bars = bars_ax.barh(names_, efs_, color=colors_, edgecolor="none", height=0.55)
    bars_ax.set_xlabel("Emission Factor (g CO\u2082)", color="#94A3B8", fontsize=8)
    bars_ax.tick_params(colors="#94A3B8", labelsize=8)
    for spine in bars_ax.spines.values():
        spine.set_edgecolor("#334155")
    for bar, val in zip(bars, efs_):
        bars_ax.text(bar.get_width() + max(efs_) * 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{val:,.0f}", color="white", fontsize=7.5, va="center")
    bars_ax.set_title("Emission Factor by Strategy", color="white", fontsize=9, pad=4)

    plt.savefig(save_path, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[INFO] Route comparison saved -> {save_path}")
    return save_path

def find_matching_cached_dynamic_city(olat: float, olon: float, dlat: float, dlon: float) -> Optional[str]:
    """Find a cached dynamic city name whose bounding box covers the requested points."""
    cities_dir = BASE_DIR / "cities"
    if not cities_dir.exists():
        return None
    for folder in cities_dir.iterdir():
        if folder.is_dir() and folder.name.startswith("custom_"):
            parts = folder.name.split("_")
            if len(parts) == 5:
                try:
                    lat_min = float(parts[1])
                    lon_min = float(parts[2])
                    lat_max = float(parts[3])
                    lon_max = float(parts[4])
                    # Check if requested points are inside the cached bbox with a buffer margin
                    margin = 0.002
                    if (lat_min + margin <= olat <= lat_max - margin and
                        lon_min + margin <= olon <= lon_max - margin and
                        lat_min + margin <= dlat <= lat_max - margin and
                        lon_min + margin <= dlon <= lon_max - margin):
                        return folder.name
                except ValueError:
                    continue
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MAIN PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────
def run_eco_routing(origin_lat: float, origin_lon: float,
                     dest_lat: float,   dest_lon: float,
                     hour: int = 8,
                     city: str = "kolkata",
                     plot_path: str = None,
                     use_ml: bool = True,
                     vehicle_override: int = None,
                     route_overrides: dict = None,
                     segment_overrides: dict = None,
                     tomtom_api_key: str = None,
                     vehicle_type: str = "car") -> dict:
    """
    Full pipeline:
      1. Load / cache graph (dynamically download if city is "custom_route")
      2. Find nearest graph nodes to (lat,lon) pairs
      3. Adjust edge travel times based on vehicle type
      4. Compute 4 routes on base graph first (needed to extract route edges for scaling)
      5. If route_overrides or segment_overrides present, build a scaled copy of the
         graph and re-run Dijkstra — this physically changes the route paths on the map.
      6. Generate comparison plot
      7. Return JSON-serialisable result dict including per-route segment metadata.
    """
    if plot_path is None:
        plot_path = f"route_compare_{city}.png"

    city_key = city.lower().strip()
    
    # Boundary check for offline maps to prevent auto-switching or snapping way out of bounds
    if city_key != "custom_route" and not city_key.startswith("custom_"):
        # Load the graph to do snaps and check distances
        G = build_emission_graph(city=city, use_ml=use_ml, vehicle_override=vehicle_override)
        if G.number_of_nodes() == 0:
            return {"error": f"No network data found for city: {city}. Upload data first."}
            
        origin_node = nearest_node(G, city, origin_lat, origin_lon)
        dest_node   = nearest_node(G, city, dest_lat, dest_lon)
        
        if origin_node is None or dest_node is None:
            return {"error": "Coordinates snap query failed. No nodes found in the map graph."}
            
        # Verify snap distances
        origin_data = G.nodes[origin_node]
        dest_data = G.nodes[dest_node]
        olat_snap = origin_data.get("lat") or origin_data.get("y")
        olon_snap = origin_data.get("lon") or origin_data.get("x")
        dlat_snap = dest_data.get("lat") or dest_data.get("y")
        dlon_snap = dest_data.get("lon") or dest_data.get("x")
        
        if olat_snap is not None and olon_snap is not None and dlat_snap is not None and dlon_snap is not None:
            dist_origin = math.sqrt((olat_snap - origin_lat)**2 + (olon_snap - origin_lon)**2)
            dist_dest = math.sqrt((dlat_snap - dest_lat)**2 + (dlon_snap - dest_lon)**2)
            # 0.035 degrees is approx 3.8 km. If either is larger, it's outside the offline map region
            if dist_origin > 0.035 or dist_dest > 0.035:
                return {"error": f"This location is outside the {city.title()} offline map. Please select Dynamic India."}
    else:
        # Dynamic India routing
        # Check if coordinates span too far (cross-country routing block)
        lat_min, lat_max = min(origin_lat, dest_lat), max(origin_lat, dest_lat)
        lon_min, lon_max = min(origin_lon, dest_lon), max(origin_lon, dest_lon)
        if (lat_max - lat_min) > 0.35 or (lon_max - lon_min) > 0.35:
            return {"error": f"The distance between coordinates is too large for local dynamic routing (bounding box spans "
                             f"{(lat_max - lat_min):.2f}° x {(lon_max - lon_min):.2f}°). Max allowed is 0.35° (approx 38 km). "
                             f"Please select coordinates within the same city."}
                             
        # Find if we already have a cached bbox graph covering these points
        matched_city = find_matching_cached_dynamic_city(origin_lat, origin_lon, dest_lat, dest_lon)
        if matched_city:
            print(f"[INFO] Found matching cached dynamic city: {matched_city}")
            city = matched_city
            city_key = matched_city
            G = build_emission_graph(city=city, use_ml=use_ml, vehicle_override=vehicle_override)
        else:
            # We must download and fuse a new bbox
            buffer = 0.015
            lat_min_buf = lat_min - buffer
            lat_max_buf = lat_max + buffer
            lon_min_buf = lon_min - buffer
            lon_max_buf = lon_max + buffer
            new_city_name = f"custom_{lat_min_buf:.4f}_{lon_min_buf:.4f}_{lat_max_buf:.4f}_{lon_max_buf:.4f}"
            print(f"[INFO] No cached dynamic city found. Downloading new bounding box under: {new_city_name}")
            try:
                download_and_fuse_bbox(origin_lat, origin_lon, dest_lat, dest_lon, city_name=new_city_name)
                city = new_city_name
                city_key = new_city_name
                G = build_emission_graph(city=city, use_ml=use_ml, vehicle_override=vehicle_override)
            except Exception as e:
                return {"error": f"Failed to download and process OSM data: {str(e)}"}

    if G.number_of_nodes() == 0:
        return {"error": f"No network data found for city: {city}. Upload data first."}

    print(f"[INFO] Finding nearest nodes to origin ({origin_lat}, {origin_lon})...")
    origin_node = nearest_node(G, city, origin_lat, origin_lon)
    print(f"[INFO] Finding nearest nodes to dest ({dest_lat}, {dest_lon})...")
    dest_node   = nearest_node(G, city, dest_lat, dest_lon)
    print(f"[INFO] Origin node: {origin_node}  |  Dest node: {dest_node}")

    if origin_node == dest_node:
        return {"error": "Origin and destination are the same node. Move them further apart."}

    # ── Step 1: Base routing (always needed) ──────────────────────────────────
    base_routes, strategies = compute_all_routes(G, origin_node, dest_node)
    if not base_routes:
        return {"error": "No path found between these two points. Try different coordinates."}

    # ── Step 1b: Enrich with TomTom live traffic if API key is available ──────
    if tomtom_api_key and not route_overrides and not segment_overrides:
        # Get route segments for all base routes
        base_route_segments = {}
        for name, route in base_routes.items():
            base_route_segments[name] = get_route_segments(G, route, top_n=30)
        # Fetch live traffic and build enriched graph
        enriched_G = enrich_graph_with_live_traffic(G, base_route_segments, tomtom_api_key, hour)
        if enriched_G.number_of_edges() > 0:
            enriched_routes, strategies = compute_all_routes(enriched_G, origin_node, dest_node)
            if enriched_routes:
                G = enriched_G
                base_routes = enriched_routes
                print("[INFO] Routes recomputed with TomTom live traffic data.")

    # ── Step 2: Build scaled graph if overrides are present ───────────────────
    # This is the key fix: Dijkstra is re-run on a modified graph so the actual
    # path coordinates change (not just the displayed emission stats).
    active_graph = G
    routes       = base_routes

    if route_overrides or segment_overrides:
        # Translate route-level overrides into edge-level scales
        combined_edge_scales: dict = {}
        if route_overrides:
            for rname, override_vc in route_overrides.items():
                base_route = base_routes.get(rname)
                if base_route:
                    edge_scales = get_route_edge_scales(G, base_route, override_vc)
                    combined_edge_scales.update(edge_scales)

        H = build_scaled_graph_with_overrides(
            G,
            segment_overrides=segment_overrides,
            route_edge_scales=combined_edge_scales,
        )
        # Recompute composite weights on scaled graph
        new_routes, strategies = compute_all_routes(H, origin_node, dest_node)
        if new_routes:
            active_graph = H
            routes       = new_routes
            print(f"[INFO] Re-routed on scaled graph with "
                  f"{len(route_overrides or {})} route-overrides and "
                  f"{len(segment_overrides or {})} segment-overrides.")

    # Adjust edge properties in G/active_graph based on vehicle_type
    for u, v, key, data in active_graph.edges(data=True, keys=True):
        avg_spd = data.get("avg_speed", 30)
        length_m = data["length"]
        
        # Speed adjustment based on vehicle type
        if vehicle_type == "motorcycle":
            travel_speed = min(avg_spd * 1.15, 60.0)
        elif vehicle_type == "bicycle":
            travel_speed = 15.0
        elif vehicle_type in ["bus", "truck"]:
            travel_speed = min(avg_spd * 0.8, 40.0)
        else:
            travel_speed = avg_spd
            
        travel_speed = max(travel_speed, 5.0)
        data["time"] = length_m / (travel_speed * 1000 / 3600)

    # ── Step 3: Compute stats (no extra vehicle_override needed — baked in) ───
    all_stats: dict = {}
    for name, route in routes.items():
        all_stats[name] = route_stats(active_graph, route, vehicle_type=vehicle_type)

    # Best route (lowest emission_factor)
    best_name = min(all_stats, key=lambda n: all_stats[n]["emission_factor"])

    # ── Step 4: Plot ──────────────────────────────────────────────────────────
    plot_route_comparison(active_graph, routes, strategies, all_stats,
                          origin_node, dest_node, save_path=plot_path)

    # ── Step 5: Build GeoJSON for each route (for frontend map) ───────────────
    from pyproj import Transformer
    _tr = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    routes_geojson: dict = {}
    for name, route in routes.items():
        coords = []
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            if v not in active_graph[u]:
                continue
            ed = min(active_graph[u][v].values(), key=lambda x: x["emission_factor"])
            geom = ed.get("geometry")
            if geom and geom.geom_type == "LineString":
                for x, y in geom.coords:
                    lon_, lat_ = _tr.transform(x, y)
                    coords.append([lat_, lon_])
        routes_geojson[name] = coords

    # ── Step 6: Extract per-route segment metadata for the upload UI ──────────
    route_segments: dict = {}
    for name, route in routes.items():
        route_segments[name] = get_route_segments(active_graph, route, top_n=30)

    return {
        "origin_node":    origin_node,
        "dest_node":      dest_node,
        "strategies":     {k: v[1] for k, v in strategies.items()},
        "stats":          all_stats,
        "best_route":     best_name,
        "routes_latlng":  routes_geojson,
        "plot_path":      plot_path,
        "route_segments": route_segments,   # NEW: per-route segment metadata
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8.  CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, pprint
    parser = argparse.ArgumentParser(description="Eco-routing engine for Kolkata")
    parser.add_argument("--origin-lat", type=float, default=22.5680,
                        help="Origin latitude  (default: near Esplanade)")
    parser.add_argument("--origin-lon", type=float, default=88.3512,
                        help="Origin longitude")
    parser.add_argument("--dest-lat",   type=float, default=22.5756,
                        help="Destination latitude  (default: near Sealdah)")
    parser.add_argument("--dest-lon",   type=float, default=88.3697,
                        help="Destination longitude")
    parser.add_argument("--hour",       type=int,   default=8)
    parser.add_argument("--plot",       type=str,   default="route_compare_kolkata.png")
    parser.add_argument("--no-ml",      action="store_true")
    args = parser.parse_args()

    result = run_eco_routing(
        args.origin_lat, args.origin_lon,
        args.dest_lat,   args.dest_lon,
        hour=args.hour,
        plot_path=args.plot,
        use_ml=not args.no_ml,
    )

    if "error" in result:
        print(f"\n[ERROR] {result['error']}")
        sys.exit(1)

    print("\n" + "=" * 55)
    print("  ECO-ROUTING RESULTS — Kolkata")
    print("=" * 55)
    for name, st in result["stats"].items():
        marker = " <-- BEST ECO" if name == result["best_route"] else ""
        print(f"\n  {name}{marker}")
        print(f"    Distance     : {st['distance_km']} km")
        print(f"    Time         : {st['time_min']} min")
        print(f"    Emission Factor : {st['emission_factor']:,.1f} g CO2")
        print(f"    Avg vehicles : {st['avg_vehicles']}")
    print(f"\n  Plot saved: {result['plot_path']}")
    print("=" * 55)
