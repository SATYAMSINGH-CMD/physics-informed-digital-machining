"""
Systematic Research Ablation Study: Data Efficiency & Physics Regularization
Investigates:
1. Physics-Informed Inductive Bias vs Pure Data-Driven Learning
2. Performance under Severe Data Scarcity (10%, 25%, 50%, 75%, 100% Training Data)
3. Cross-Tool Generalization under Distribution Shift (GroupKFold vs Random CV)
"""

import os
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

BASE_DIR = r"D:\tony dataset"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

MASTER_CSV = os.path.join(BASE_DIR, "all_datasets_features_12_master.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_COLUMNS_12 = [
    "kurtosis_1st_derivative",
    "ar_coeff_2",
    "d2_energy",
    "skewness_1st_derivative",
    "multi_axis_energy_asymmetry",
    "d3_d4_subband_energy_ratio",
    "coherence_at_dominant_resonant",
    "impulse_factor",
    "cross_axis_peak_delay",
    "cross_spectral_centroid",
    "phase_space_ellipsicity",
    "bivariate_orbit_radius_ratio",
]

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


class SimpleNet(nn.Module):
    def __init__(self, in_dim=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def physics_loss(preds, depths, bounds, lambda_p=0.35):
    if lambda_p <= 0.0:
        return torch.tensor(0.0, device=preds.device)
    delta = depths - bounds
    v_chatter = torch.clamp(delta, min=0.0) * (1.0 - preds)
    v_stable = torch.clamp(-delta, min=0.0) * preds
    return lambda_p * torch.mean(v_chatter + v_stable)


def train_eval_nn(X_tr, y_tr, d_tr, b_tr, X_val, y_val, lambda_p=0.0, epochs=60, batch_size=64):
    model = SimpleNet(in_dim=X_tr.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.004, weight_decay=1e-4)
    bce = nn.BCELoss()

    X_t = torch.tensor(X_tr, dtype=torch.float32)
    y_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    d_t = torch.tensor(d_tr, dtype=torch.float32).unsqueeze(1)
    b_t = torch.tensor(b_tr, dtype=torch.float32).unsqueeze(1)

    ds = TensorDataset(X_t, y_t, d_t, b_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for bx, by, bd, bb in loader:
            optimizer.zero_grad()
            p = model(bx)
            l_bce = bce(p, by)
            l_phy = physics_loss(p, bd, bb, lambda_p)
            loss = l_bce + l_phy
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds_val = model(torch.tensor(X_val, dtype=torch.float32)).numpy().flatten()

    y_pred = (preds_val >= 0.5).astype(int)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    auc = roc_auc_score(y_val, preds_val) if len(np.unique(y_val)) > 1 else 0.5
    return acc, f1, auc


def run_ablation_study():
    print("=" * 80)
    print("SYSTEMATIC RESEARCH ABLATION STUDY: DATA SCARCITY & PHYSICS REGULARIZATION")
    print("=" * 80)

    df = pd.read_csv(MASTER_CSV)
    for f in FEATURE_COLUMNS_12:
        df[f] = df[f].replace([np.inf, -np.inf], np.nan).fillna(df[f].median())

    X = df[FEATURE_COLUMNS_12].values
    y = df['label'].values
    depths = df['axial_depth_m'].values
    bounds = df['boundary_m'].values
    groups = df['dataset_id'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    fractions = [0.10, 0.25, 0.50, 0.75, 1.00]
    results = {
        "fractions": fractions,
        "pure_nn_group": [],
        "pinn_group": [],
        "xgb_group": [],
        "pure_nn_random": [],
        "pinn_random": [],
        "xgb_random": [],
    }

    gkf = GroupKFold(n_splits=5)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for frac in fractions:
        print(f"\n[*] Evaluating Data Tier: {int(frac*100)}% Training Data...")

        # 1. GroupKFold (Cross-Tool Generalization)
        pinn_g_accs, nn_g_accs, xgb_g_accs = [], [], []
        for tr_idx, val_idx in gkf.split(X_scaled, y, groups=groups):
            if frac < 1.0:
                np.random.seed(RANDOM_STATE)
                sub_size = int(len(tr_idx) * frac)
                sub_tr_idx = np.random.choice(tr_idx, size=sub_size, replace=False)
            else:
                sub_tr_idx = tr_idx

            X_tr, y_tr = X_scaled[sub_tr_idx], y[sub_tr_idx]
            d_tr, b_tr = depths[sub_tr_idx], bounds[sub_tr_idx]
            X_val, y_val = X_scaled[val_idx], y[val_idx]

            # Pure NN (No physics: lambda=0.0)
            acc_nn, _, _ = train_eval_nn(X_tr, y_tr, d_tr, b_tr, X_val, y_val, lambda_p=0.0, epochs=50)
            # PINN (With physics loss: lambda=0.40)
            acc_pinn, _, _ = train_eval_nn(X_tr, y_tr, d_tr, b_tr, X_val, y_val, lambda_p=0.40, epochs=50)
            # XGBoost
            xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE, n_jobs=-1, eval_metric="logloss")
            xgb.fit(X_tr, y_tr)
            acc_xgb = accuracy_score(y_val, xgb.predict(X_val))

            nn_g_accs.append(acc_nn)
            pinn_g_accs.append(acc_pinn)
            xgb_g_accs.append(acc_xgb)

        # 2. Random CV
        pinn_r_accs, nn_r_accs, xgb_r_accs = [], [], []
        for tr_idx, val_idx in skf.split(X_scaled, y):
            if frac < 1.0:
                np.random.seed(RANDOM_STATE)
                sub_size = int(len(tr_idx) * frac)
                sub_tr_idx = np.random.choice(tr_idx, size=sub_size, replace=False)
            else:
                sub_tr_idx = tr_idx

            X_tr, y_tr = X_scaled[sub_tr_idx], y[sub_tr_idx]
            d_tr, b_tr = depths[sub_tr_idx], bounds[sub_tr_idx]
            X_val, y_val = X_scaled[val_idx], y[val_idx]

            acc_nn, _, _ = train_eval_nn(X_tr, y_tr, d_tr, b_tr, X_val, y_val, lambda_p=0.0, epochs=50)
            acc_pinn, _, _ = train_eval_nn(X_tr, y_tr, d_tr, b_tr, X_val, y_val, lambda_p=0.40, epochs=50)
            xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE, n_jobs=-1, eval_metric="logloss")
            xgb.fit(X_tr, y_tr)
            acc_xgb = accuracy_score(y_val, xgb.predict(X_val))

            nn_r_accs.append(acc_nn)
            pinn_r_accs.append(acc_pinn)
            xgb_r_accs.append(acc_xgb)

        m_pinn_g = np.mean(pinn_g_accs) * 100
        m_nn_g = np.mean(nn_g_accs) * 100
        m_xgb_g = np.mean(xgb_g_accs) * 100
        delta_pinn = m_pinn_g - m_nn_g

        results["pinn_group"].append(m_pinn_g)
        results["pure_nn_group"].append(m_nn_g)
        results["xgb_group"].append(m_xgb_g)
        results["pinn_random"].append(np.mean(pinn_r_accs) * 100)
        results["pure_nn_random"].append(np.mean(nn_r_accs) * 100)
        results["xgb_random"].append(np.mean(xgb_r_accs) * 100)

        print(f"    GroupKFold -> Pure NN: {m_nn_g:.2f}% | PINN: {m_pinn_g:.2f}% (Δ = +{delta_pinn:.2f}%) | XGBoost: {m_xgb_g:.2f}%")

    # Generate Summary DataFrame
    df_ablation = pd.DataFrame({
        "Training_Data_Pct": [f"{int(f*100)}%" for f in fractions],
        "GroupKFold_Pure_NN": [f"{v:.2f}%" for v in results["pure_nn_group"]],
        "GroupKFold_PINN": [f"{v:.2f}%" for v in results["pinn_group"]],
        "GroupKFold_PINN_Advantage": [f"+{p - n:.2f}%" for p, n in zip(results["pinn_group"], results["pure_nn_group"])],
        "GroupKFold_XGBoost": [f"{v:.2f}%" for v in results["xgb_group"]],
        "RandomCV_PINN": [f"{v:.2f}%" for v in results["pinn_random"]],
    })

    csv_path = os.path.join(BASE_DIR, "research_ablation_results.csv")
    json_path = os.path.join(BASE_DIR, "research_ablation_results.json")
    df_ablation.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("FINAL RESEARCH ABLATION MATRIX (DATA EFFICIENCY):")
    print("=" * 80)
    print(df_ablation.to_string(index=False))
    print("=" * 80)

    # Plot Publication Figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    pct_axis = [int(f * 100) for f in fractions]

    # Subplot 1: Cross-Tool Generalization vs Data Scarcity (GroupKFold)
    ax1 = axes[0]
    ax1.plot(pct_axis, results["pinn_group"], 'o-', color='#D4F04D', lw=3, markersize=8, label='PINN (Altintaş-Budak Loss)', markeredgecolor='#111111', markeredgewidth=2)
    ax1.plot(pct_axis, results["pure_nn_group"], 's--', color='#EF4444', lw=2, markersize=7, label='Pure Data-Driven NN (No Physics)')
    ax1.plot(pct_axis, results["xgb_group"], '^:', color='#38BDF8', lw=2, markersize=7, label='XGBoost Baseline')
    ax1.set_xlabel("Available Training Data (%)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Cross-Tool Accuracy (%) [GroupKFold]", fontsize=11, fontweight='bold')
    ax1.set_title("A. Cross-Tool Generalization under Data Scarcity", fontsize=12, fontweight='bold')
    ax1.set_xticks(pct_axis)
    ax1.set_ylim(70, 95)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc='lower right')

    # Subplot 2: Physics Inductive Bias Advantage
    ax2 = axes[1]
    advantage = [p - n for p, n in zip(results["pinn_group"], results["pure_nn_group"])]
    bars = ax2.bar([str(p) + "%" for p in pct_axis], advantage, color='#D4F04D', edgecolor='#30302D', width=0.45)
    ax2.set_xlabel("Training Data Partition", fontsize=11, fontweight='bold')
    ax2.set_ylabel("PINN Accuracy Gain over Pure NN (+%)", fontsize=11, fontweight='bold')
    ax2.set_title("B. Empirical Value of Physics Inductive Bias vs Data Volume", fontsize=12, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.15, f"+{yval:.2f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plot_path = os.path.join(BASE_DIR, "data_efficiency_ablation_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[+] Saved scientific ablation plot to: {plot_path}")


if __name__ == "__main__":
    run_ablation_study()
