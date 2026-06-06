import argparse
import math
import os
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, Pool

warnings.filterwarnings("ignore")

ROAD_FEATURES = ["length", "lanes", "building_density", "vegetation_score",
                 "maxspeed", "width"]

TRAFFIC_FEATURES = ["vehicle_count", "avg_speed_kmph",
                    "n_car", "n_motorcycle", "n_bus",
                    "n_truck", "n_van", "n_bicycle", "n_auto"]

EMISSION_FEATURES = ["co2_per_km_g",
                     "co2_car", "co2_motorcycle", "co2_bus",
                     "co2_truck", "co2_van", "co2_auto",
                     "base_cost_feature"]

ENV_FEATURES = ["pm2_5_ugm3", "pm10_ugm3", "no2_ugm3", "o3_ugm3",
                "so2_ugm3", "co_ugm3", "AQI", "openweather_aqi_1to5",
                "temperature_k", "humidity_pct", "wind_speed_mps"]

TEMPORAL_FEATURES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]

CATEGORICAL_COLS = ["highway", "weather_description", "junction_enc",
                    "oneway_enc", "reversed_enc"]

TARGET = "carbon_cost"

EMIT = {"n_car": 120, "n_motorcycle": 72, "n_bus": 822,
        "n_truck": 900, "n_van": 200, "n_bicycle": 0, "n_auto": 85}

# ──────────────────────────────────────────────────────────────────────────────
# 2.  DATA LOADING & PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────
def load_and_preprocess(synth_path: str, real_path: str = None,
                        use_real: bool = False):
    dfs = [pd.read_csv(synth_path)]

    if use_real and real_path and Path(real_path).exists():
        real = pd.read_csv(real_path)
        for col in ["n_car","n_motorcycle","n_bus","n_truck",
                    "n_van","n_bicycle","n_auto","co2_per_km_g",
                    "hour_of_day","day_of_week","is_weekend"]:
            if col not in real.columns:
                if col == "hour_of_day":
                    real[col] = 8
                elif col == "day_of_week":
                    real[col] = 1
                elif col == "is_weekend":
                    real[col] = 0
                elif col == "co2_per_km_g":
                    real[col] = real["vehicle_count"] * 120
                else:
                    real[col] = (real["vehicle_count"] * 0.14).astype(int)
        dfs.append(real)
        print(f"[INFO] Loaded real data: {len(real):,} rows")

    df = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Total rows after merge: {len(df):,}")

    for vtype, g_per_km in EMIT.items():
        col_name = "co2_" + vtype.replace("n_", "")
        df[col_name] = df[vtype].fillna(0) * g_per_km

    df["base_cost_feature"] = (
        df["vehicle_count"] / df["avg_speed_kmph"].clip(lower=5)
    ) * df["length"]

    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)

    all_num = (ROAD_FEATURES + TRAFFIC_FEATURES + EMISSION_FEATURES +
               ENV_FEATURES + TEMPORAL_FEATURES)
    for col in all_num:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median() if df[col].dtype != object else 0)
        else:
            df[col] = 0.0

    df["junction_enc"]  = df["junction"].fillna("none").astype(str)
    df["oneway_enc"]    = df["oneway"].astype(str)
    df["reversed_enc"]  = df["reversed"].astype(str)
    df["weather_description"] = df["weather_description"].fillna("clear sky").astype(str)
    df["highway"]       = df["highway"].fillna("residential").astype(str)

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df[TARGET] = np.log1p(df[TARGET])

    features = all_num + CATEGORICAL_COLS
    return df[features], df[TARGET], all_num


def metrics_str(preds_log, trues_log):
    p = np.expm1(preds_log)
    t = np.expm1(trues_log)
    rmse = math.sqrt(mean_squared_error(t, p))
    mae  = mean_absolute_error(t, p)
    r2   = r2_score(t, p)
    mask = t > 1.0
    mape = np.mean(np.abs((t[mask] - p[mask]) / t[mask])) * 100 if mask.sum() > 0 else 0
    return rmse, mae, r2, mape

def plot_pred_vs_actual(preds_log, trues_log, save_path="pred_vs_actual_catboost.png"):
    p = np.expm1(preds_log)
    t = np.expm1(trues_log)
    cap = np.percentile(t, 99)
    mask = t < cap
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.scatter(t[mask], p[mask], alpha=0.25, s=6, color="#4C72B0")
    lo, hi = t[mask].min(), t[mask].max()
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect")
    ax.set_xlabel("Actual Carbon Cost (g CO2)");  ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Actual (CatBoost)")
    ax.legend();  ax.grid(alpha=0.3)

    ax = axes[1]
    residuals = p[mask] - t[mask]
    ax.hist(residuals, bins=80, color="#55A868", edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="red", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Residual (Predicted - Actual)")
    ax.set_ylabel("Count");  ax.set_title("Residual Distribution")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved prediction plot -> {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synth",      default="synthetic_delhi_roads.csv")
    parser.add_argument("--real",       default="delhi/fused_roads.csv")
    parser.add_argument("--also-real",  action="store_true",
                        help="Merge real data into training set")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--lr",         type=float, default=0.05)
    parser.add_argument("--depth",      type=int, default=6)
    parser.add_argument("--out-dir",    default=".")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)

    print("\n[1/4] Loading data…")
    X, y, numeric_cols = load_and_preprocess(
        synth_path=args.synth,
        real_path=args.real,
        use_real=args.also_real,
    )
    print(f"{len(X):,} rows, {X.shape[1]} columns")

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.20, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)
    print(f"      Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    print("\n[2/4] Initializing CatBoost…")
    model = CatBoostRegressor(
        iterations=args.iterations,
        learning_rate=args.lr,
        depth=args.depth,
        loss_function="RMSE",
        eval_metric="RMSE",
        cat_features=CATEGORICAL_COLS,
        verbose=100,
        random_seed=42,
        task_type="CPU"
    )

    train_pool = Pool(X_train, y_train, cat_features=CATEGORICAL_COLS)
    val_pool = Pool(X_val, y_val, cat_features=CATEGORICAL_COLS)

    print(f"\n[3/4] Training CatBoost for up to {args.iterations} iterations…")
    model.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=50,
        use_best_model=True
    )

    print(f"\n[4/4] Evaluating on test set…")
    test_preds = model.predict(X_test)
    rmse, mae, r2, mape = metrics_str(test_preds, y_test.values)
    print("\n")
    print("CATBOOST TEST SET RESULTS (original scale)")
    print(f"RMSE  : {rmse:>12.2f}  g CO2")
    print(f"MAE   : {mae:>12.2f}  g CO2")
    print(f"R2    : {r2:>12.4f}")
    print(f"MAPE  : {mape:>11.2f}%")

    model_path = str(out / "carbon_fusion_catboost.cbm")
    model.save_model(model_path)
    print(f"[INFO] Model saved -> {model_path}")

    plot_pred_vs_actual(test_preds, y_test.values,
                        save_path=str(out / "pred_vs_actual_catboost.png"))

if __name__ == "__main__":
    main()
