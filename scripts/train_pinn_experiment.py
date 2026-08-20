import os
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = r"D:\tony dataset"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

MODELS_DIR = os.path.join(BASE_DIR, "models")
MASTER_CSV = os.path.join(BASE_DIR, "all_datasets_features_12_master.csv")
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

def main():
    print("=" * 80)
    print("TRAINING PHYSICS-INFORMED NEURAL NETWORK (PINN)")
    print("=" * 80)

    import torch
    import torch.nn as nn
    from tony_dataset.pinn_model import MachiningPINN

    df = pd.read_csv(MASTER_CSV)
    
    # Fill any NaNs
    for f in FEATURE_COLUMNS_12:
        df[f] = df[f].replace([np.inf, -np.inf], np.nan).fillna(df[f].median())
        
    X = df[FEATURE_COLUMNS_12].values
    y = df['label'].values
    depths = df['axial_depth_m'].values
    boundaries = df['boundary_m'].values

    # Scale features
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler_12.joblib"))
    X_scaled = scaler.transform(X)

    # Train-test split (80/20)
    indices = np.arange(len(df))
    np.random.seed(42)
    np.random.shuffle(indices)
    split_idx = int(0.8 * len(df))
    tr_idx, val_idx = indices[:split_idx], indices[split_idx:]

    pinn = MachiningPINN(
        input_dim=12,
        hidden_dims=(64, 32, 16),
        lambda_physics=0.35,
        learning_rate=0.003
    )

    print(f"\n[*] Training PINN across {len(tr_idx)} samples for 120 epochs...")
    history = pinn.train_pinn(
        X_scaled[tr_idx],
        y[tr_idx],
        depths[tr_idx],
        boundaries[tr_idx],
        epochs=120,
        batch_size=128
    )

    # Validation metrics
    preds = pinn.predict(X_scaled[val_idx])
    probs = pinn.predict_proba(X_scaled[val_idx])
    acc = np.mean(preds == y[val_idx])
    
    from sklearn.metrics import roc_auc_score, f1_score
    val_f1 = f1_score(y[val_idx], preds)
    val_auc = roc_auc_score(y[val_idx], probs)

    print(f"\n[+] PINN Training Complete:")
    print(f"    Validation Accuracy: {acc*100:.2f}%")
    print(f"    Validation F1-Score: {val_f1:.4f}")
    print(f"    Validation ROC-AUC:  {val_auc:.4f}")

    # Export PyTorch weights
    pt_path = os.path.join(MODELS_DIR, "pinn_model.pt")
    torch.save(pinn.model.state_dict(), pt_path)
    print(f"    [+] Saved PyTorch model: {pt_path}")

    # Export ONNX model
    onnx_path = os.path.join(MODELS_DIR, "pinn_chatter.onnx")
    pinn.export_onnx(onnx_path)

    # Plot PINN Loss Curve
    plt.figure(figsize=(9, 5))
    plt.plot(history['bce_loss'], label='Data Loss (BCE)', color='#3b82f6', lw=2)
    plt.plot(history['phys_loss'], label='Physics Regularization Loss', color='#f59e0b', lw=2)
    plt.plot(history['total_loss'], label='Total Combined Loss', color='#10b981', lw=2.5)
    plt.xlabel('Epoch', fontsize=11, fontweight='bold')
    plt.ylabel('Loss', fontsize=11, fontweight='bold')
    plt.title('Physics-Informed Neural Network (PINN) Loss Convergence', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    loss_plot_path = os.path.join(BASE_DIR, "pinn_loss_convergence.png")
    plt.savefig(loss_plot_path, dpi=300)
    plt.close()
    print(f"    [+] Saved PINN loss convergence plot: {loss_plot_path}")

if __name__ == "__main__":
    main()
