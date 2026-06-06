"""
generate_synthetic_data.py
---------------------------
Generates realistic synthetic road-segment data for Delhi modelled after
the real `fused_roads.csv` dataset.

Key additions over the original data:
  - Dynamic vehicle_count driven by road type + time-of-day profile
  - Vehicle-type breakdown  (n_cars, n_motorcycles, n_buses, n_trucks,
                             n_vans, n_bicycles, n_autos)
  - CO2 per km for the segment  (co2_per_km_g)
  - Richer, more realistic carbon_cost target built from:
      vehicle density × length × emission factor × congestion penalty
      + AQI air-quality penalty + road-type base cost
  - Varied weather snapshots (not just one weather snapshot)
  - Time-of-day and day-of-week encoded

Usage:
    python generate_synthetic_data.py                     # 10 000 rows → synthetic_delhi_roads.csv
    python generate_synthetic_data.py --rows 50000        # 50 000 rows
    python generate_synthetic_data.py --seed 99           # fixed seed
    python generate_synthetic_data.py --output mydata.csv
"""

import argparse
import random
import numpy as np
import pandas as pd

# ── Reproducibility ────────────────────────────────────────────────────────
SEED = 42

# ── Emissions: g CO2 per km per vehicle (IPCC / EEA values) ───────────────
EMISSIONS_G_PER_KM = {
    "car":        120,
    "motorcycle":  72,
    "bus":        822,
    "truck":      900,
    "van":        200,
    "bicycle":      0,
    "auto":        85,   # CNG auto-rickshaw (common in Delhi)
}

# ── Vehicle-type mix per road type (fractions that sum to 1) ──────────────
#   Based on Delhi traffic surveys (approx.)
VEHICLE_MIX = {
    "residential":    {"car": 0.50, "motorcycle": 0.25, "bus": 0.02,
                       "truck": 0.02, "van": 0.05, "bicycle": 0.08, "auto": 0.08},
    "tertiary":       {"car": 0.45, "motorcycle": 0.22, "bus": 0.06,
                       "truck": 0.06, "van": 0.07, "bicycle": 0.06, "auto": 0.08},
    "secondary":      {"car": 0.40, "motorcycle": 0.18, "bus": 0.10,
                       "truck": 0.12, "van": 0.08, "bicycle": 0.04, "auto": 0.08},
    "secondary_link": {"car": 0.42, "motorcycle": 0.20, "bus": 0.08,
                       "truck": 0.10, "van": 0.08, "bicycle": 0.04, "auto": 0.08},
    "tertiary_link":  {"car": 0.42, "motorcycle": 0.22, "bus": 0.06,
                       "truck": 0.06, "van": 0.08, "bicycle": 0.08, "auto": 0.08},
    "living_street":  {"car": 0.35, "motorcycle": 0.28, "bus": 0.01,
                       "truck": 0.01, "van": 0.04, "bicycle": 0.18, "auto": 0.13},
    "unclassified":   {"car": 0.45, "motorcycle": 0.20, "bus": 0.05,
                       "truck": 0.08, "van": 0.07, "bicycle": 0.07, "auto": 0.08},
    "busway":         {"car": 0.10, "motorcycle": 0.05, "bus": 0.70,
                       "truck": 0.05, "van": 0.05, "bicycle": 0.02, "auto": 0.03},
}
DEFAULT_MIX = VEHICLE_MIX["secondary"]

# ── Traffic multipliers by time-of-day (hour 0-23) ────────────────────────
HOUR_MULTIPLIER = {
    0: 0.10, 1: 0.07, 2: 0.05, 3: 0.05, 4: 0.07, 5: 0.20,
    6: 0.60, 7: 1.30, 8: 1.80, 9: 1.50, 10: 1.10, 11: 1.05,
    12: 1.10, 13: 1.15, 14: 1.05, 15: 1.00, 16: 1.20, 17: 1.70,
    18: 1.90, 19: 1.60, 20: 1.20, 21: 0.80, 22: 0.50, 23: 0.25,
}

# ── Base vehicle counts per road type (peak-hour vehicles/segment) ─────────
BASE_VEHICLE_COUNT = {
    "residential":    {"mean":  60, "std": 25},
    "tertiary":       {"mean": 100, "std": 40},
    "secondary":      {"mean": 140, "std": 50},
    "secondary_link": {"mean": 110, "std": 40},
    "tertiary_link":  {"mean":  80, "std": 30},
    "living_street":  {"mean":  30, "std": 15},
    "unclassified":   {"mean":  70, "std": 30},
    "busway":         {"mean":  50, "std": 20},
}
DEFAULT_BASE = {"mean": 80, "std": 35}

# ── Speed ranges per road type (km/h) ─────────────────────────────────────
SPEED_RANGE = {
    "residential":    (15, 45),
    "tertiary":       (20, 55),
    "secondary":      (25, 65),
    "secondary_link": (20, 55),
    "tertiary_link":  (15, 50),
    "living_street":  (10, 30),
    "unclassified":   (15, 50),
    "busway":         (20, 60),
}
DEFAULT_SPEED = (15, 55)

# ── Delhi road names pool ──────────────────────────────────────────────────
ROAD_NAMES = [
    "Janpath", "Akbar Road", "Motilal Nehru Marg", "Shahjahan Road",
    "Panchsheel Marg", "Pandara Road", "Kartavya Path", "Outer Circle",
    "Middle Circle", "Radial Road 1", "Radial Road 7", "Raisina Road",
    "Barakhamba Road", "Mandi House Chowk", "Firoz Shah Road",
    "Kasturba Gandhi Marg", "Humayun Road", "Kautilya Marg",
    "Chandragupta Marg", "Kumaraswami Kamaraj Marg", "Copernicus Marg",
    "Red Cross Road", "Sansad Marg", "Nyaya Marg", "Vijay Chowk",
    "Man Singh Road", "Rafi Marg", "Windsor Place", "Prithviraj Road",
    "Maulana Azad Road", "Panchkuian Road", "Baba Kharak Singh Marg",
    "Mother Teresa Crescent", "Sardar Patel Marg", "Max Mueller Marg",
    "San Martin Marg", "Vardhman Marg", "Fifth Avenue Road",
    "Doctor Rajendra Prasad Road", "Jaswant Singh Road",
    None,  # unnamed roads
]

WEATHER_SCENARIOS = [
    {"weather_description": "clear sky",      "temp_delta": 0,   "hum_delta":  0,  "aqi_delta":   0},
    {"weather_description": "few clouds",     "temp_delta":-1,   "hum_delta":  5,  "aqi_delta":   5},
    {"weather_description": "scattered clouds","temp_delta":-2,  "hum_delta": 10,  "aqi_delta":  10},
    {"weather_description": "broken clouds",  "temp_delta":-3,   "hum_delta": 15,  "aqi_delta":  15},
    {"weather_description": "light rain",     "temp_delta":-5,   "hum_delta": 30,  "aqi_delta": -10},
    {"weather_description": "moderate rain",  "temp_delta":-7,   "hum_delta": 50,  "aqi_delta": -20},
    {"weather_description": "haze",           "temp_delta":+2,   "hum_delta": 20,  "aqi_delta":  40},
    {"weather_description": "smoke",          "temp_delta":+3,   "hum_delta": 15,  "aqi_delta":  80},
    {"weather_description": "mist",           "temp_delta":-2,   "hum_delta": 40,  "aqi_delta":   5},
    {"weather_description": "thunderstorm",   "temp_delta":-8,   "hum_delta": 60,  "aqi_delta": -15},
    {"weather_description": "dust",           "temp_delta":+5,   "hum_delta":  5,  "aqi_delta": 120},
    {"weather_description": "fog",            "temp_delta":-4,   "hum_delta": 70,  "aqi_delta":  20},
]

HIGHWAY_TYPES = [
    "residential", "residential", "residential", "residential",   # most common
    "secondary", "secondary", "secondary",
    "tertiary", "tertiary",
    "secondary_link",
    "tertiary_link",
    "living_street", "living_street",
    "unclassified",
    "busway",
]

JUNCTION_TYPES = ["roundabout", None, None, None, None]  # mostly non-junction
LANES_BY_TYPE  = {
    "residential": [None, None, 1, 2],
    "secondary":   [2, 2, 3, 4],
    "tertiary":    [None, 1, 2, 3],
    "secondary_link": [None, 1, 2],
    "tertiary_link":  [None, 1],
    "living_street":  [None, 1],
    "unclassified":   [None, 1, 2],
    "busway":          [1, 2, 2],
}


def _base_pollutants(aqi_level: int, weather_delta: int, rng: np.random.Generator):
    """Return realistic pollutant readings correlated with AQI."""
    # Delhi baseline (moderate pollution)
    base_pm25  = 35 + (aqi_level - 2) * 12 + weather_delta * 0.5
    base_pm10  = base_pm25 * 1.18
    base_no2   = 0.8  + (aqi_level - 2) * 0.4
    base_o3    = 160  - (aqi_level - 1) * 15
    base_so2   = 2.5  + (aqi_level - 2) * 1.2
    base_co    = 250  + (aqi_level - 2) * 60

    jitter = lambda v, pct: max(0, v * (1 + rng.uniform(-pct, pct)))
    return {
        "pm2_5_ugm3":         round(jitter(base_pm25, 0.15), 2),
        "pm10_ugm3":          round(jitter(base_pm10, 0.12), 2),
        "no2_ugm3":           round(jitter(base_no2,  0.20), 2),
        "o3_ugm3":            round(jitter(base_o3,   0.10), 2),
        "so2_ugm3":           round(jitter(base_so2,  0.25), 2),
        "co_ugm3":            round(jitter(base_co,   0.15), 2),
    }


def _aqi_from_aqi5(aqi5: int, rng: np.random.Generator) -> int:
    """Convert 1-5 OpenWeather AQI to US AQI approximation."""
    mapping = {1: (0, 50), 2: (51, 100), 3: (101, 150), 4: (151, 200), 5: (201, 300)}
    lo, hi = mapping.get(aqi5, (51, 100))
    return int(rng.integers(lo, hi + 1))


def compute_vehicle_breakdown(total_vehicles: int, highway: str, rng: np.random.Generator):
    """Split total_vehicles into per-type counts using road-type mix."""
    mix = VEHICLE_MIX.get(highway, DEFAULT_MIX)
    types = list(mix.keys())
    probs = np.array([mix[t] for t in types])
    probs = probs / probs.sum()  # ensure sums to 1

    # Multinomial draw
    counts = rng.multinomial(total_vehicles, probs)
    return {f"n_{t}": int(c) for t, c in zip(types, counts)}


def compute_co2_per_km(breakdown: dict) -> float:
    """Return total CO2 g per km for all vehicles on this segment."""
    total = 0.0
    for vtype, factor in EMISSIONS_G_PER_KM.items():
        total += breakdown.get(f"n_{vtype}", 0) * factor
    return round(total, 2)


def compute_carbon_cost(
    length_m: float,
    vehicle_count: int,
    avg_speed: float,
    co2_per_km: float,
    aqi: int,
    highway: str,
    building_density: int,
    vegetation_score: int,
    rng: np.random.Generator,
) -> float:
    """
    Physics-inspired carbon cost formula (calibrated to match real data pattern).

    carbon_cost = base_emission_cost
                + congestion_penalty
                + aqi_penalty
                + road_type_premium
                + noise

    base_emission_cost:   (vehicle_count / avg_speed) * length_m * k
        ~ g CO2 delivered on this segment at this speed
        k ≈ 6 (reverse-engineered from real data: mean ratio = 6.02)

    congestion_penalty:   extra cost when density is high & speed is low
    aqi_penalty:          roads with worse air quality carry higher social cost
    road_type_premium:    arterials (secondary/tertiary) have higher base cost
    """
    # Base cost (dominant term, corr=0.93 with real data)
    k = rng.normal(6.0, 0.15)  # tight variability — keeps signal learnable
    k = max(4.0, k)
    base = (vehicle_count / max(avg_speed, 5)) * length_m * k

    # Congestion penalty: kicks in when >120 vehicles AND speed <35 km/h
    congestion = 0.0
    if vehicle_count > 120 and avg_speed < 35:
        congestion_index = (vehicle_count - 120) / 80 * (35 - avg_speed) / 20
        congestion = base * congestion_index * rng.uniform(0.05, 0.25)

    # AQI social cost (higher AQI = greater health externality)
    aqi_penalty = (aqi / 200) * vehicle_count * 0.8

    # Road-type premium (secondary > tertiary > residential)
    road_premium = {"secondary": 1.10, "secondary_link": 1.05,
                    "tertiary": 1.03, "tertiary_link": 1.01,
                    "busway": 1.15}.get(highway, 1.0)

    # Urban density penalty (dense areas → more stop-start → more emissions)
    density_factor = 1 + building_density * 0.005

    # Vegetation acts as partial offset (urban forests reduce local heat)
    veg_offset = vegetation_score * 0.8

    total = (base + congestion + aqi_penalty) * road_premium * density_factor - veg_offset
    total = max(total, 5.0)  # floor

    # Small Gaussian noise to avoid perfectly deterministic target
    total *= rng.normal(1.0, 0.04)
    return round(max(total, 1.0), 6)


def generate_row(osmid_counter: int, rng: np.random.Generator) -> dict:
    """Generate one synthetic road-segment row."""
    highway = rng.choice(HIGHWAY_TYPES)

    # ── Road geometry ──────────────────────────────────────────────────────
    # Length drawn from log-normal to match the real distribution
    length = float(np.exp(rng.normal(4.5, 1.1)))   # mean ≈ 90 m, range 5-3000
    length = round(np.clip(length, 5.0, 3500.0), 8)

    oneway   = rng.choice([True, False], p=[0.65, 0.35])
    reversed_ = rng.choice([True, False], p=[0.3, 0.7])
    junction = rng.choice(JUNCTION_TYPES)
    lanes_pool = LANES_BY_TYPE.get(highway, [None, 1, 2])
    lanes    = rng.choice(lanes_pool)
    maxspeed = rng.choice([None, None, None, 30, 40, 50, 60])
    access   = rng.choice([None, None, None, "no", "private"])
    ref      = rng.choice([None, None, "5", "NH8", "SH1"])
    width    = rng.choice([None, None, 6, 8, 10, 12, 15])
    bridge   = rng.choice([None, None, "yes"])
    landuse  = rng.choice([None, None, "residential", "commercial", "industrial"])

    building_density = int(rng.integers(0, 30))
    vegetation_score = int(rng.integers(0, 25))

    # ── Time of day & day of week ─────────────────────────────────────────
    hour = int(rng.integers(0, 24))
    dow  = int(rng.integers(0, 7))   # 0=Monday
    is_weekend = int(dow >= 5)

    # ── Vehicle count (dynamic) ───────────────────────────────────────────
    base_cfg = BASE_VEHICLE_COUNT.get(highway, DEFAULT_BASE)
    time_mult = HOUR_MULTIPLIER[hour]
    weekend_mult = 0.75 if is_weekend else 1.0
    raw_count = rng.normal(base_cfg["mean"], base_cfg["std"]) * time_mult * weekend_mult
    vehicle_count = int(np.clip(round(raw_count), 1, 400))

    # ── Average speed (inversely related to count) ────────────────────────
    spd_lo, spd_hi = SPEED_RANGE.get(highway, DEFAULT_SPEED)
    # Speed decreases with congestion
    congestion_ratio = min(vehicle_count / (base_cfg["mean"] * 2), 1.0)
    effective_lo = spd_lo
    effective_hi = spd_hi - congestion_ratio * (spd_hi - spd_lo) * 0.5
    avg_speed = float(round(rng.uniform(effective_lo, max(effective_lo + 2, effective_hi)), 1))
    avg_speed = max(5.0, avg_speed)

    # ── Vehicle breakdown + CO2 ───────────────────────────────────────────
    breakdown = compute_vehicle_breakdown(vehicle_count, highway, rng)
    co2_per_km = compute_co2_per_km(breakdown)

    # ── Weather ───────────────────────────────────────────────────────────
    scenario = rng.choice(WEATHER_SCENARIOS)
    base_temp  = 312.08 + rng.normal(0, 3)
    base_hum   = 23     + rng.normal(0, 8)
    base_wind  = 0.39   + rng.uniform(0, 3)
    wind_dir   = int(rng.integers(0, 360))

    temperature_k  = round(base_temp + scenario["temp_delta"], 2)
    humidity_pct   = round(np.clip(base_hum + scenario["hum_delta"], 5, 99), 1)
    wind_speed_mps = round(base_wind, 2)

    # ── AQI ──────────────────────────────────────────────────────────────
    # AQI correlated with vehicle density + weather
    base_aqi5 = rng.choice([1, 2, 3, 4, 4, 5], p=[0.05, 0.15, 0.30, 0.30, 0.15, 0.05])
    aqi5_adj = base_aqi5 + (1 if scenario["aqi_delta"] > 30 else 0)
    aqi5 = int(np.clip(aqi5_adj, 1, 5))
    aqi  = _aqi_from_aqi5(aqi5, rng)

    pollutants = _base_pollutants(aqi5, scenario["aqi_delta"], rng)

    # ── Carbon cost (target variable) ────────────────────────────────────
    carbon_cost = compute_carbon_cost(
        length_m=length,
        vehicle_count=vehicle_count,
        avg_speed=avg_speed,
        co2_per_km=co2_per_km,
        aqi=aqi,
        highway=highway,
        building_density=building_density,
        vegetation_score=vegetation_score,
        rng=rng,
    )

    road_name = rng.choice(ROAD_NAMES)

    # Explicit physics feature (dominant predictor, corr=0.93 with real data)
    base_cost_feature = (vehicle_count / max(avg_speed, 5)) * length

    return {
        # ── Original schema columns ────────────────────────────────────────
        "osmid":                  osmid_counter,
        "highway":                highway,
        "name":                   road_name,
        "oneway":                 oneway,
        "reversed":               reversed_,
        "length":                 length,
        "lanes":                  lanes,
        "junction":               junction,
        "ref":                    ref,
        "maxspeed":               maxspeed,
        "access":                 access,
        "width":                  width,
        "bridge":                 bridge,
        "landuse":                landuse,
        "building_density":       building_density,
        "vegetation_score":       vegetation_score,
        "vehicle_count":          vehicle_count,
        "avg_speed_kmph":         avg_speed,
        "wind_speed_mps":         wind_speed_mps,
        "wind_dir_deg":           wind_dir,
        "temperature_k":          temperature_k,
        "humidity_pct":           humidity_pct,
        "weather_description":    scenario["weather_description"],
        "weather_source":         "synthetic",
        **pollutants,
        "openweather_aqi_1to5":   aqi5,
        "AQI":                    aqi,
        "aqi_source":             "synthetic",
        # ── New enrichment columns ─────────────────────────────────────────
        "hour_of_day":            hour,
        "day_of_week":            dow,
        "is_weekend":             is_weekend,
        **breakdown,                              # n_car, n_motorcycle, …
        "co2_per_km_g":           co2_per_km,     # total CO2 g/km from all vehicles
        "base_cost_feature":      round(base_cost_feature, 4),  # physics: count/speed*length
        # ── Target ────────────────────────────────────────────────────────
        "carbon_cost":            carbon_cost,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Delhi road data for ML training")
    parser.add_argument("--rows",   type=int,   default=30_000,                    help="Number of rows to generate")
    parser.add_argument("--seed",   type=int,   default=SEED,                      help="Random seed")
    parser.add_argument("--output", type=str,   default="synthetic_delhi_roads.csv", help="Output CSV path")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    print(f"[INFO] Generating {args.rows:,} synthetic road segments (seed={args.seed})…")
    rows = [generate_row(i + 1_000_000, rng) for i in range(args.rows)]

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)

    print(f"[INFO] Saved -> {args.output}  ({len(df):,} rows x {len(df.columns)} columns)")
    print("\n--- Vehicle count stats ---")
    print(df["vehicle_count"].describe().to_string())
    print("\n--- CO2/km stats (g) ---")
    print(df["co2_per_km_g"].describe().to_string())
    print("\n--- carbon_cost stats ---")
    print(df["carbon_cost"].describe().to_string())
    print("\n--- Vehicle type totals ---")
    vcols = [c for c in df.columns if c.startswith("n_")]
    print(df[vcols].sum().to_string())
    print("\n--- Highway distribution ---")
    print(df["highway"].value_counts().to_string())
    print("\n--- Weather distribution ---")
    print(df["weather_description"].value_counts().to_string())
    print(f"\n[DONE] Load with:  df = pd.read_csv('{args.output}')")


if __name__ == "__main__":
    main()
