"""
carbon_cost_model.py
--------------------
Multi-Branch Fusion Deep Learning model for Carbon Cost prediction.

Architecture  (CarbonFusionNet):
  Branch 1 – Road geometry   : length, lanes, building_density, vegetation_score
  Branch 2 – Traffic         : vehicle_count, avg_speed, n_car … n_auto
  Branch 3 – Emissions       : co2_per_km_g, vehicle-type CO2 contributions
  Branch 4 – Environment/AQI : pm2.5, pm10, NO2, O3, SO2, CO, AQI, temp, humidity, wind
  Branch 5 – Temporal        : hour_of_day (cyclic), day_of_week (cyclic), is_weekend
  Categorical embeddings      : highway type, weather description, junction type

Each branch is processed independently by Residual Blocks, then all branches
are fused via a learned Gated Attention mechanism before the regression head.

Target : log1p(carbon_cost)  →  inversed back to g CO2 at prediction time.

Usage:
    # Train on synthetic data
    python carbon_cost_model.py

    # Train + use real data too
    python carbon_cost_model.py --also-real

    # Quick smoke-test (5 epochs)
    python carbon_cost_model.py --epochs 5 --batch-size 256
"""

import argparse
import math
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless – saves to file
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# ──────────────────────────────────────────────────────────────────────────────
# 1.  FEATURE DEFINITIONS
# ──────────────────────────────────────────────────────────────────────────────
ROAD_FEATURES = ["length", "lanes", "building_density", "vegetation_score",
                 "maxspeed", "width"]

TRAFFIC_FEATURES = ["vehicle_count", "avg_speed_kmph",
                    "n_car", "n_motorcycle", "n_bus",
                    "n_truck", "n_van", "n_bicycle", "n_auto"]

EMISSION_FEATURES = ["co2_per_km_g",
                     # weighted CO2 contributions per type
                     "co2_car", "co2_motorcycle", "co2_bus",
                     "co2_truck", "co2_van", "co2_auto",
                     # dominant physics predictor (corr=0.93 with carbon_cost)
                     "base_cost_feature"]

ENV_FEATURES = ["pm2_5_ugm3", "pm10_ugm3", "no2_ugm3", "o3_ugm3",
                "so2_ugm3", "co_ugm3", "AQI", "openweather_aqi_1to5",
                "temperature_k", "humidity_pct", "wind_speed_mps"]

TEMPORAL_FEATURES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]

CATEGORICAL_COLS = ["highway", "weather_description", "junction_enc",
                    "oneway_enc", "reversed_enc"]

TARGET = "carbon_cost"

# CO2 g/km per vehicle type (same as in generator)
EMIT = {"n_car": 120, "n_motorcycle": 72, "n_bus": 822,
        "n_truck": 900, "n_van": 200, "n_bicycle": 0, "n_auto": 85}

# ──────────────────────────────────────────────────────────────────────────────
# 2.  DATA LOADING & PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────
def load_and_preprocess(synth_path: str, real_path: str = None,
                        use_real: bool = False):
    """Load CSV(s), engineer features, encode categoricals."""
    dfs = [pd.read_csv(synth_path)]

    if use_real and real_path and Path(real_path).exists():
        real = pd.read_csv(real_path)
        # Fill vehicle-type columns that real data lacks
        for col in ["n_car","n_motorcycle","n_bus","n_truck",
                    "n_van","n_bicycle","n_auto","co2_per_km_g",
                    "hour_of_day","day_of_week","is_weekend"]:
            if col not in real.columns:
                if col == "hour_of_day":
                    real[col] = 8          # assume morning peak for real data
                elif col == "day_of_week":
                    real[col] = 1
                elif col == "is_weekend":
                    real[col] = 0
                elif col == "co2_per_km_g":
                    real[col] = real["vehicle_count"] * 120  # car-only approx
                else:
                    # distribute vehicle_count equally across types
                    real[col] = (real["vehicle_count"] * 0.14).astype(int)
        dfs.append(real)
        print(f"[INFO] Loaded real data: {len(real):,} rows")

    df = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Total rows after merge: {len(df):,}")

    # ── Derived CO2 contributions per vehicle type ─────────────────────────
    for vtype, g_per_km in EMIT.items():
        col_name = "co2_" + vtype.replace("n_", "")
        df[col_name] = df[vtype].fillna(0) * g_per_km

    # ── Dominant physics predictor (works on both real & synthetic data) ───
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
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median()
                       if df[col].dtype != object else 0)
        else:
            df[col] = 0.0

    cat_encoders = {}
    df["junction_enc"]  = df["junction"].fillna("none").astype(str)
    df["oneway_enc"]    = df["oneway"].astype(str)
    df["reversed_enc"]  = df["reversed"].astype(str)
    df["weather_description"] = df["weather_description"].fillna("clear sky")
    df["highway"]       = df["highway"].fillna("residential")

    for col in ["highway", "weather_description", "junction_enc",
                "oneway_enc", "reversed_enc"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        cat_encoders[col] = le

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df[TARGET] = np.log1p(df[TARGET])

    return df, cat_encoders


class CarbonDataset(Dataset):
    def __init__(self, df: pd.DataFrame, scalers: dict = None, fit: bool = False):
        self.scalers = scalers if scalers else {}
        branch_cols = {
            "road":     ROAD_FEATURES,
            "traffic":  TRAFFIC_FEATURES,
            "emission": EMISSION_FEATURES,
            "env":      ENV_FEATURES,
            "temporal": TEMPORAL_FEATURES,
        }
        self.X_branches = {}
        for name, cols in branch_cols.items():
            arr = df[cols].values.astype(np.float32)
            if fit:
                sc = StandardScaler()
                arr = sc.fit_transform(arr)
                self.scalers[name] = sc
            elif name in self.scalers:
                arr = self.scalers[name].transform(arr)
            self.X_branches[name] = torch.tensor(arr, dtype=torch.float32)

        # Categorical integers
        self.X_cat = torch.tensor(
            df[CATEGORICAL_COLS].values.astype(np.int64), dtype=torch.long)

        self.y = torch.tensor(df[TARGET].values.astype(np.float32),
                              dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            {k: v[idx] for k, v in self.X_branches.items()},
            self.X_cat[idx],
            self.y[idx],
        )


class ResidualBlock(nn.Module):
    """Pre-activation residual block with optional dimensionality change."""
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.25):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.fc1   = nn.Linear(in_dim, out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.fc2   = nn.Linear(out_dim, out_dim)
        self.act   = nn.GELU()
        self.drop  = nn.Dropout(dropout)
        self.skip  = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        x = self.act(self.fc1(self.norm1(x)))
        x = self.drop(x)
        x = self.fc2(self.norm2(x))
        return self.act(x + residual)


class BranchEncoder(nn.Module):
    """Encodes one feature branch into a fixed-size embedding."""
    def __init__(self, in_dim: int, hidden: int, out_dim: int,
                 n_blocks: int = 2, dropout: float = 0.25):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.GELU()]
        for _ in range(n_blocks):
            layers.append(ResidualBlock(hidden, hidden, dropout))
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GatedFusion(nn.Module):
    """
    Learns a soft attention gate over N branch embeddings.
    Output = sum_i( gate_i * branch_i ), where gates sum to 1.
    """
    def __init__(self, n_branches: int, embed_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(n_branches * embed_dim, n_branches),
            nn.Softmax(dim=-1)
        )
        self.n = n_branches
        self.d = embed_dim

    def forward(self, branches: list):
        # branches: list of (B, embed_dim) tensors
        cat = torch.cat(branches, dim=-1)            # (B, n*d)
        gates = self.gate(cat).unsqueeze(-1)         # (B, n, 1)
        stack = torch.stack(branches, dim=1)         # (B, n, d)
        fused = (gates * stack).sum(dim=1)           # (B, d)
        return fused


class CarbonFusionNet(nn.Module):
    """
    Multi-branch fusion network for carbon cost regression.

    Branches:
      road (6) → 128
      traffic (9) → 128
      emission (7) → 128
      env (11) → 128
      temporal (5) → 64 → projected to 128
      categorical embeddings → 128

    All branches → GatedFusion → Regression head
    """
    def __init__(self, cat_vocab_sizes: list, embed_dim_each: int = 8,
                 branch_out: int = 128, dropout: float = 0.25):
        super().__init__()

        # Branch encoders
        self.road_enc   = BranchEncoder(len(ROAD_FEATURES),     128, branch_out, n_blocks=3)
        self.traf_enc   = BranchEncoder(len(TRAFFIC_FEATURES),  128, branch_out, n_blocks=3)
        self.emit_enc   = BranchEncoder(len(EMISSION_FEATURES), 128, branch_out, n_blocks=3)
        self.env_enc    = BranchEncoder(len(ENV_FEATURES),      128, branch_out, n_blocks=3)
        self.temp_enc   = BranchEncoder(len(TEMPORAL_FEATURES),  64, branch_out, n_blocks=2)

        # Categorical embeddings  (one per categorical column)
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size + 1, embed_dim_each)
            for vocab_size in cat_vocab_sizes
        ])
        cat_total = len(cat_vocab_sizes) * embed_dim_each
        self.cat_proj = nn.Sequential(
            nn.Linear(cat_total, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, branch_out),
            nn.GELU(),
        )

        # Gated fusion across 6 branches
        n_branches = 6
        self.fusion = GatedFusion(n_branches, branch_out)

        # Final regression head
        fused_dim = branch_out
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualBlock(256, 256, dropout),
            ResidualBlock(256, 128, dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, branches: dict, cats: torch.Tensor):
        b_road  = self.road_enc(branches["road"])
        b_traf  = self.traf_enc(branches["traffic"])
        b_emit  = self.emit_enc(branches["emission"])
        b_env   = self.env_enc(branches["env"])
        b_temp  = self.temp_enc(branches["temporal"])

        embs = [self.embeddings[i](cats[:, i]) for i in range(len(self.embeddings))]
        b_cat = self.cat_proj(torch.cat(embs, dim=-1))

        fused = self.fusion([b_road, b_traf, b_emit, b_env, b_temp, b_cat])
        return self.head(fused)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  TRAINING UTILITIES
# ──────────────────────────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_loss = float("inf")
        self.stop      = False

    def __call__(self, val_loss: float):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


def train_epoch(model, loader, optimiser, criterion, scaler=None):
    model.train()
    total_loss = 0.0
    for branches, cats, y in loader:
        branches = {k: v.to(DEVICE) for k, v in branches.items()}
        cats, y  = cats.to(DEVICE), y.to(DEVICE)

        optimiser.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                pred = model(branches, cats)
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimiser)
            scaler.update()
        else:
            pred = model(branches, cats)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    preds, trues = [], []
    for branches, cats, y in loader:
        branches = {k: v.to(DEVICE) for k, v in branches.items()}
        cats, y  = cats.to(DEVICE), y.to(DEVICE)
        pred = model(branches, cats)
        total_loss += criterion(pred, y).item() * len(y)
        preds.extend(pred.cpu().numpy().flatten())
        trues.extend(y.cpu().numpy().flatten())
    loss = total_loss / len(loader.dataset)
    return loss, np.array(preds), np.array(trues)


def metrics_str(preds_log, trues_log):
    """Compute metrics in original scale (undo log1p)."""
    p = np.expm1(preds_log)
    t = np.expm1(trues_log)
    rmse = math.sqrt(mean_squared_error(t, p))
    mae  = mean_absolute_error(t, p)
    r2   = r2_score(t, p)
    # MAPE (exclude near-zero)
    mask = t > 1.0
    mape = np.mean(np.abs((t[mask] - p[mask]) / t[mask])) * 100 if mask.sum() > 0 else 0
    return rmse, mae, r2, mape

def plot_training_curves(train_losses, val_losses, save_path="training_curves.png"):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(train_losses, label="Train Loss (MSE)", linewidth=2)
    ax.plot(val_losses,   label="Val Loss (MSE)",   linewidth=2)
    ax.set_xlabel("Epoch");  ax.set_ylabel("MSE (log-scale)")
    ax.set_title("CarbonFusionNet – Training Curves")
    ax.legend();  ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved training curve -> {save_path}")


def plot_pred_vs_actual(preds_log, trues_log, save_path="pred_vs_actual.png"):
    p = np.expm1(preds_log)
    t = np.expm1(trues_log)
    # Cap for readability
    cap = np.percentile(t, 99)
    mask = t < cap
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.scatter(t[mask], p[mask], alpha=0.25, s=6, color="#4C72B0")
    lo, hi = t[mask].min(), t[mask].max()
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect")
    ax.set_xlabel("Actual Carbon Cost (g CO2)");  ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Actual")
    ax.legend();  ax.grid(alpha=0.3)

    # Residuals
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


# ──────────────────────────────────────────────────────────────────────────────
# 6.  INFERENCE HELPER
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_carbon_cost(model, df_row: dict, scalers: dict,
                        cat_encoders: dict) -> float:
    """
    Predict carbon_cost (g CO2) for a single road segment.

    df_row : dict with the same keys used during training.
    Returns: float (carbon cost in grams CO2 per km)
    """
    model.eval()
    row = pd.DataFrame([df_row])

    # Derive CO2 contributions
    for vtype, g in EMIT.items():
        col = "co2_" + vtype.replace("n_", "")
        row[col] = row.get(vtype, 0) * g

    # Cyclic temporal
    row["hour_sin"] = np.sin(2 * np.pi * row["hour_of_day"] / 24)
    row["hour_cos"] = np.cos(2 * np.pi * row["hour_of_day"] / 24)
    row["dow_sin"]  = np.sin(2 * np.pi * row["day_of_week"] / 7)
    row["dow_cos"]  = np.cos(2 * np.pi * row["day_of_week"] / 7)

    # Fill missing columns
    all_num = (ROAD_FEATURES + TRAFFIC_FEATURES + EMISSION_FEATURES +
               ENV_FEATURES + TEMPORAL_FEATURES)
    for col in all_num:
        if col not in row.columns:
            row[col] = 0.0

    # Encode categoricals
    row["junction_enc"] = row["junction"].fillna("none").astype(str)
    row["oneway_enc"]   = row["oneway"].astype(str)
    row["reversed_enc"] = row["reversed"].astype(str)
    for col in CATEGORICAL_COLS:
        le = cat_encoders.get(col)
        if le:
            val = str(row[col].iloc[0])
            try:
                row[col] = le.transform([val])[0]
            except ValueError:
                row[col] = 0   # unseen category → index 0

    # Build tensors
    branches = {}
    for name, cols in [("road", ROAD_FEATURES), ("traffic", TRAFFIC_FEATURES),
                       ("emission", EMISSION_FEATURES), ("env", ENV_FEATURES),
                       ("temporal", TEMPORAL_FEATURES)]:
        arr = row[cols].values.astype(np.float32)
        if name in scalers:
            arr = scalers[name].transform(arr)
        branches[name] = torch.tensor(arr, dtype=torch.float32).to(DEVICE)

    cats = torch.tensor(row[CATEGORICAL_COLS].values.astype(np.int64),
                        dtype=torch.long).to(DEVICE)

    log_pred = model(branches, cats).item()
    return float(np.expm1(log_pred))


# ──────────────────────────────────────────────────────────────────────────────
# 7.  MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synth",      default="synthetic_delhi_roads.csv")
    parser.add_argument("--real",       default="delhi/fused_roads.csv")
    parser.add_argument("--also-real",  action="store_true",
                        help="Merge real data into training set")
    parser.add_argument("--epochs",     type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--dropout",    type=float, default=0.25)
    parser.add_argument("--patience",   type=int, default=20)
    parser.add_argument("--out-dir",    default=".")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)

    # ── Load & preprocess ────────────────────────────────────────────────
    print("\n[1/5] Loading data…")
    df, cat_encoders = load_and_preprocess(
        synth_path=args.synth,
        real_path=args.real,
        use_real=args.also_real,
    )
    print(f"      {len(df):,} rows, {df.shape[1]} columns")

    # ── Train / val / test split ─────────────────────────────────────────
    train_df, temp_df = train_test_split(df, test_size=0.20, random_state=42)
    val_df,   test_df = train_test_split(temp_df, test_size=0.50, random_state=42)
    print(f"      Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

    # ── Datasets & loaders ───────────────────────────────────────────────
    print("\n[2/5] Building datasets…")
    train_ds = CarbonDataset(train_df, fit=True)
    scalers  = train_ds.scalers
    val_ds   = CarbonDataset(val_df,  scalers=scalers)
    test_ds  = CarbonDataset(test_df, scalers=scalers)

    nw = min(4, os.cpu_count() or 1)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True,  num_workers=0, pin_memory=False)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size * 2,
                          shuffle=False, num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size * 2,
                          shuffle=False, num_workers=0)

    # ── Model ────────────────────────────────────────────────────────────
    print("\n[3/5] Building model…")
    cat_vocab_sizes = [
        int(df[col].max()) + 1 for col in CATEGORICAL_COLS
    ]
    model = CarbonFusionNet(
        cat_vocab_sizes=cat_vocab_sizes,
        embed_dim_each=8,
        branch_out=128,
        dropout=args.dropout,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      Parameters: {n_params:,}")
    print(f"      Architecture summary:\n{model}")

    # ── Optimiser & schedule ─────────────────────────────────────────────
    criterion = nn.HuberLoss(delta=1.0)   # robust to outliers vs MSE
    optimiser = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimiser,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_dl),
        pct_start=0.1,
        anneal_strategy="cos",
    )
    early_stop = EarlyStopping(patience=args.patience)
    use_amp    = DEVICE.type == "cuda"
    amp_scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # ── Training loop ────────────────────────────────────────────────────
    print(f"\n[4/5] Training for up to {args.epochs} epochs…")
    print(f"      {'Epoch':>5}  {'Train':>10}  {'Val':>10}  "
          f"{'RMSE':>10}  {'MAE':>10}  {'R2':>8}  {'MAPE':>8}  LR")
    print("      " + "-" * 78)

    best_val_loss = float("inf")
    train_losses, val_losses = [], []
    best_state = None

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_dl, optimiser, criterion, amp_scaler)
        scheduler.step()
        vl_loss, vp, vt = eval_epoch(model, val_dl, criterion)

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        rmse, mae, r2, mape = metrics_str(vp, vt)
        lr = optimiser.param_groups[0]["lr"]

        print(f"      {epoch:5d}  {tr_loss:10.5f}  {vl_loss:10.5f}  "
              f"{rmse:10.1f}  {mae:10.1f}  {r2:8.4f}  {mape:7.2f}%  {lr:.2e}")

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        early_stop(vl_loss)
        if early_stop.stop:
            print(f"\n      [Early stopping at epoch {epoch}]")
            break

    # ── Final evaluation ─────────────────────────────────────────────────
    print(f"\n[5/5] Evaluating on test set…")
    model.load_state_dict(best_state)
    model.to(DEVICE)

    _, test_preds, test_trues = eval_epoch(model, test_dl, criterion)
    rmse, mae, r2, mape = metrics_str(test_preds, test_trues)

    print("\n  ============================================")
    print("   TEST SET RESULTS (original scale g CO2)")
    print("  ============================================")
    print(f"   RMSE  : {rmse:>12.2f}  g CO2")
    print(f"   MAE   : {mae:>12.2f}  g CO2")
    print(f"   R2    : {r2:>12.4f}")
    print(f"   MAPE  : {mape:>11.2f}%")
    print("  ============================================\n")

    # ── Save artefacts ───────────────────────────────────────────────────
    model_path = str(out / "carbon_fusion_net.pt")
    torch.save({
        "model_state":    best_state,
        "cat_vocab_sizes": cat_vocab_sizes,
        "cat_encoders":   cat_encoders,
        "scalers":        scalers,
        "dropout":        args.dropout,
        "branch_out":     128,
        "embed_dim_each": 8,
    }, model_path)
    print(f"[INFO] Model saved -> {model_path}")

    plot_training_curves(train_losses, val_losses,
                         save_path=str(out / "training_curves.png"))
    plot_pred_vs_actual(test_preds, test_trues,
                        save_path=str(out / "pred_vs_actual.png"))

    # ── Quick demo inference ─────────────────────────────────────────────
    print("\n[DEMO] Predicting carbon cost for a sample road segment…")
    sample = {
        "length": 250.0, "lanes": 2, "building_density": 18,
        "vegetation_score": 3, "maxspeed": 40, "width": 10,
        "vehicle_count": 26, "avg_speed_kmph": 42,
        "n_car": 13, "n_motorcycle": 7, "n_bus": 1, "n_truck": 1,
        "n_van": 2, "n_bicycle": 1, "n_auto": 1,
        "co2_per_km_g": 3000,
        "pm2_5_ugm3": 41.08, "pm10_ugm3": 47.82, "no2_ugm3": 1.22,
        "o3_ugm3": 171.4, "so2_ugm3": 3.52, "co_ugm3": 309.39,
        "AQI": 114, "openweather_aqi_1to5": 4,
        "temperature_k": 312.08, "humidity_pct": 23,
        "wind_speed_mps": 0.39,
        "hour_of_day": 8, "day_of_week": 1, "is_weekend": 0,
        "highway": "residential", "weather_description": "clear sky",
        "junction": None, "oneway": True, "reversed": False,
    }

    pred = predict_carbon_cost(model, sample, scalers, cat_encoders)
    print(f"   Input  : 26 vehicles (13 cars, 7 motorcycles, 1 bus, 1 truck…)")
    print(f"   Road   : residential, 250 m, speed 42 km/h, AQI=114")
    print(f"   Predicted carbon cost : {pred:,.1f} g CO2")
    print(f"   (Real data reference  :   885.7 g CO2)")


if __name__ == "__main__":
    main()
