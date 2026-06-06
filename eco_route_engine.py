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
import math
import os
import random
import sys
import warnings
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


def predict_emission_factor(row: dict, vehicle_info: dict) -> float:
    """
    Predict emission_factor for one road segment.
    Uses ML model if available, otherwise physics formula.
    """
    bundle = _load_model_bundle()
    if bundle is None:
        return physics_emission_factor(
            vehicle_count=vehicle_info["total"],
            avg_speed=float(row.get("avg_speed_kmph", 35)),
            length_m=float(row.get("length", 100)),
            co2_per_km=vehicle_info.get("co2_per_km_g"),
            building_density=int(row.get("building_density", 5)),
            vegetation_score=int(row.get("vegetation_score", 2)),
            aqi=int(row.get("AQI", 100)),
        )

    row_input = {
        "length":             float(row.get("length", 100)),
        "lanes":              float(row.get("lanes") or 1),
        "building_density":   int(row.get("building_density", 5)),
        "vegetation_score":   int(row.get("vegetation_score", 2)),
        "maxspeed":           float(row.get("maxspeed") or 40),
        "width":              float(row.get("width") or 8),
        "vehicle_count":      vehicle_info["total"],
        "avg_speed_kmph":     float(row.get("avg_speed_kmph", 35)),
        "n_car":              vehicle_info.get("n_car", 0),
        "n_motorcycle":       vehicle_info.get("n_motorcycle", 0),
        "n_bus":              vehicle_info.get("n_bus", 0),
        "n_truck":            vehicle_info.get("n_truck", 0),
        "n_van":              vehicle_info.get("n_van", 0),
        "n_bicycle":          vehicle_info.get("n_bicycle", 0),
        "n_auto":             vehicle_info.get("n_auto", 0),
        "co2_per_km_g":       vehicle_info.get("co2_per_km_g", 10000),
        "pm2_5_ugm3":         float(row.get("pm2_5_ugm3", 40)),
        "pm10_ugm3":          float(row.get("pm10_ugm3", 50)),
        "no2_ugm3":           float(row.get("no2_ugm3", 1.5)),
        "o3_ugm3":            float(row.get("o3_ugm3", 160)),
        "so2_ugm3":           float(row.get("so2_ugm3", 3)),
        "co_ugm3":            float(row.get("co_ugm3", 300)),
        "AQI":                int(row.get("AQI", 100)),
        "openweather_aqi_1to5": int(row.get("openweather_aqi_1to5", 3)),
        "temperature_k":      float(row.get("temperature_k", 305)),
        "humidity_pct":       float(row.get("humidity_pct", 70)),
        "wind_speed_mps":     float(row.get("wind_speed_mps", 1)),
    }
    
    emit = {"n_car": 120, "n_motorcycle": 72, "n_bus": 822,
            "n_truck": 900, "n_van": 200, "n_bicycle": 0, "n_auto": 85}
    for vtype, g in emit.items():
        row_input["co2_" + vtype.replace("n_", "")] = vehicle_info.get(vtype, 0) * g

    row_input["base_cost_feature"] = (
        row_input["vehicle_count"] / max(row_input["avg_speed_kmph"], 5.0)
    ) * row_input["length"]

    hour = 8
    dow = 1
    row_input.update({
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin":  np.sin(2 * np.pi * dow / 7),
        "dow_cos":  np.cos(2 * np.pi * dow / 7),
        "is_weekend": 0.0,
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
_graph_cache = {}

def build_emission_graph(city: str = "kolkata",
                          use_ml: bool = True,
                          hour: int = 8,
                          use_csv_emission: bool = True,
                          vehicle_override: int = None) -> nx.MultiDiGraph:
    """
    Build a directed road graph where edge weight = emission_factor.

    use_csv_emission=True  -> use the pre-computed carbon_cost column directly
                               (fastest, no ML call needed for each edge)
    use_csv_emission=False -> call ML model / physics formula per edge
    vehicle_override       -> if set (from YOLO detection), use this count
                               instead of CSV or random fallback for ALL edges
    """
    global _graph_cache
    geojson_path = BASE_DIR / city / "fused_roads.geojson"
    city_key = city.lower()

    if city_key in _graph_cache and vehicle_override is None:
        return _graph_cache[city_key]

    if not geojson_path.exists():
        print(f"[WARN] No graph file found at {geojson_path}. Returning empty graph.")
        return nx.MultiDiGraph()

    print(f"[INFO] Loading road network from: {geojson_path}")
    gdf = gpd.read_file(str(geojson_path))
    print(f"[INFO] {len(gdf):,} road segments loaded.")

    G = nx.MultiDiGraph()

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

    for idx, row in gdf.iterrows():
        u = row["u"]
        v = row["v"]

        hw  = get_highway_str(safe_scalar(row.get("highway"), "residential"))
        spd_raw = safe_scalar(row.get("avg_speed_kmph"), 30)
        spd = max(float(spd_raw) if spd_raw is not None else 30.0, 5.0)
        lng_raw = safe_scalar(row.get("length"), 50)
        lng = float(lng_raw) if lng_raw is not None else 50.0

        # ── emission_factor ────────────────────────────────────────────────
        cc_raw = safe_scalar(row.get("carbon_cost"))
        if use_csv_emission and cc_raw is not None:
            try:
                emission_factor = float(cc_raw)
            except (ValueError, TypeError):
                emission_factor = None
        else:
            emission_factor = None

        if emission_factor is None or emission_factor <= 0:
            if vehicle_override is not None:
                # Use YOLO-detected vehicle count instead of random fallback
                vinfo = fallback_vehicle_count(hw, hour)
                vinfo["total"] = vehicle_override
            else:
                vinfo = fallback_vehicle_count(hw, hour)
            emission_factor = predict_emission_factor(dict(row), vinfo)

        # ── vehicle count for display ──────────────────────────────────────
        vc_raw = safe_scalar(row.get("vehicle_count"), 60)
        vc = int(float(vc_raw)) if vc_raw is not None else 60

        # If vehicle_override is set and we have a CSV emission, scale it
        # by the ratio of detected vehicles to stored vehicle count
        if vehicle_override is not None and emission_factor > 0:
            stored_vc = int(float(vc_raw)) if vc_raw is not None else 60
            if stored_vc > 0:
                scale = vehicle_override / stored_vc
                emission_factor *= max(0.3, min(scale, 3.0))  # clamp [0.3x, 3x]

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

    print(f"[INFO] Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    if vehicle_override is None:
        _graph_cache[city_key] = G
    return G


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


def route_stats(G, route, vehicle_override=None):
    """Compute stats for a route. Returns dict."""
    dist = time_ = ef = veh = 0
    speeds = []
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

        dist  += ed["length"]
        ef    += emission
        veh   += veh_to_add
        speeds.append(ed["avg_speed"])
        
    dist_km = dist / 1000.0
    time_min = (dist_km / 40.0) * 60.0
    
    return {
        "distance_km":    round(dist_km, 3),
        "time_min":       round(time_min, 2),
        "emission_factor": round(ef, 1),
        "avg_vehicles":   round(veh / max(len(route) - 1, 1), 1),
        "avg_speed_kmh":  40.0,
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
        geom = ed.get("geometry")
        if geom and geom.geom_type == "LineString" and _tr:
            try:
                mid = geom.interpolate(0.5, normalized=True)
                lon_, lat_ = _tr.transform(mid.x, mid.y)
                midpoint = [round(lat_, 6), round(lon_, 6)]
            except Exception:
                pass

        segments.append({
            "segment_id":      seg_id,
            "edge_u":          u,
            "edge_v":          v,
            "name":            raw_name or f"{hw.replace('_', ' ').title()} Segment",
            "highway":         hw,
            "emission_factor": round(float(ed.get("emission_factor", 0)), 1),
            "vehicle_count":   int(ed.get("vehicle_count", 60)),
            "length_m":        round(float(ed.get("length", 0)), 1),
            "avg_speed_kmh":   round(float(ed.get("avg_speed", 30)), 1),
            "midpoint":        midpoint,
        })

    segments.sort(key=lambda s: s["emission_factor"], reverse=True)
    return segments[:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# 5.  NEAREST NODE FINDER
# ─────────────────────────────────────────────────────────────────────────────
def nearest_node(G, city: str, lat: float, lon: float) -> int:
    """Find graph node closest to (lat, lon) using Euclidean approx."""
    geojson_path = BASE_DIR / city / "fused_roads.geojson"
    if not geojson_path.exists():
        if G.number_of_nodes() == 0:
            return None
        # fallback to first node
        return list(G.nodes)[0]

    gdf = gpd.read_file(str(geojson_path)).to_crs("EPSG:4326")
    nodes_u = gdf[["u", "geometry"]].copy()
    nodes_u["cx"] = nodes_u.geometry.apply(
        lambda g: g.coords[0][0] if g.geom_type == "Point" else g.centroid.x)
    nodes_u["cy"] = nodes_u.geometry.apply(
        lambda g: g.coords[0][1] if g.geom_type == "Point" else g.centroid.y)

    best_node = None
    best_dist = float("inf")
    for _, r in nodes_u.drop_duplicates("u").iterrows():
        d = (r["cx"] - lon) ** 2 + (r["cy"] - lat) ** 2
        if d < best_dist:
            best_dist = d
            best_node = r["u"]
    return int(best_node)


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
                     segment_overrides: dict = None) -> dict:
    """
    Full pipeline:
      1. Load / cache Kolkata graph
      2. Find nearest graph nodes to (lat,lon) pairs
      3. Compute 4 routes on base graph first (needed to extract route edges for scaling)
      4. If route_overrides or segment_overrides present, build a scaled copy of the
         graph and re-run Dijkstra — this physically changes the route paths on the map.
      5. Generate comparison plot
      6. Return JSON-serialisable result dict including per-route segment metadata.

    vehicle_override  : scalar used to scale ALL edges (legacy global override)
    route_overrides   : { route_name: detected_vehicle_count }  (per-route upload)
    segment_overrides : { "u_v": detected_vehicle_count }       (per-segment upload)
    """
    if plot_path is None:
        plot_path = f"route_compare_{city}.png"

    G = build_emission_graph(city=city, use_ml=use_ml, vehicle_override=vehicle_override)
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

    # ── Step 3: Compute stats (no extra vehicle_override needed — baked in) ───
    all_stats: dict = {}
    for name, route in routes.items():
        all_stats[name] = route_stats(active_graph, route)

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
