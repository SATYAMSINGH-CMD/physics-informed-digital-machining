"""
Dual XGBoost Classification Experiment & Diagnostic Evaluation
Project: Digital Machining Database (Chatter vs. Stable Classification)
Dataset: digital_machining_features_12.csv

This script trains and evaluates:
1. Model 1: XGBoost on all 12 candidate features.
2. Model 2: XGBoost on 7 selected features.

Methodological constraints & safeguards:
- No metadata leakage (omega_rpm, axial_depth_m, boundary_m, file excluded).
- Stratified 80/20 train/test split with identical indices for both models.
- Hyperparameters are a conservative baseline tailored for small N (~80-100), not tuned on test data.
- Train-set metrics, 5-fold CV (training partition only), and untouched test metrics are reported.
- Evaluation plots, metrics JSON/CSV, and models are saved to D:\\tony dataset.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)
from xgboost import XGBClassifier

# Set seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Paths
BASE_DIR = r"D:\tony dataset"
CSV_PATH = os.path.join(BASE_DIR, "digital_machining_features_12.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. Load and Verify Dataset
# ---------------------------------------------------------
print("=" * 80)
print("DIGITAL MACHINING CHATTER CLASSIFICATION: XGBOOST BENCHMARK")
print("=" * 80)

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV file not found at: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
print(f"\n[1] Loaded Dataset: {CSV_PATH}")
print(f"    Total Rows (Samples): {df.shape[0]}")
print(f"    Total Columns:        {df.shape[1]}")

# Missing and Inf check
missing_counts = df.isnull().sum().to_dict()
has_missing = any(v > 0 for v in missing_counts.values())
numeric_cols = df.select_dtypes(include=[np.number]).columns
has_inf = np.isinf(df[numeric_cols].values).any()

print(f"    Missing Values:       {'Found issues: ' + str(missing_counts) if has_missing else '0 (Clean)'}")
print(f"    Infinite Values:      {'Found Inf' if has_inf else '0 (Clean)'}")

# Class distribution
label_counts = df['label'].value_counts().to_dict()
label_pcts = (df['label'].value_counts(normalize=True) * 100).to_dict()
print(f"    Class Distribution:   0 (STABLE) = {label_counts.get(0, 0)} ({label_pcts.get(0, 0):.2f}%), "
      f"1 (CHATTER) = {label_counts.get(1, 0)} ({label_pcts.get(1, 0):.2f}%)")

# ---------------------------------------------------------
# 2. Define Feature Subsets & Prevent Leakage
# ---------------------------------------------------------
FEATURES_12 = [
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
    "bivariate_orbit_radius_ratio"
]

FEATURES_7 = [
    "kurtosis_1st_derivative",
    "ar_coeff_2",
    "d2_energy",
    "skewness_1st_derivative",
    "multi_axis_energy_asymmetry",
    "d3_d4_subband_energy_ratio",
    "coherence_at_dominant_resonant"
]

# Verify feature presence in dataframe
for f in FEATURES_12:
    if f not in df.columns:
        raise ValueError(f"Required feature '{f}' missing from CSV columns!")

# Verify metadata columns are isolated
METADATA_COLS = ["file", "omega_rpm", "axial_depth_m", "boundary_m", "label"]
print("\n[2] Feature Verification & Leakage Safeguards:")
print(f"    Model 1 (Full):       {len(FEATURES_12)} features")
print(f"    Model 2 (Selected):   {len(FEATURES_7)} features")
print(f"    Excluded Metadata:    {METADATA_COLS} (None included in predictor matrix X)")

# ---------------------------------------------------------
# 3. Train / Test Partitioning (80/20 Stratified)
# ---------------------------------------------------------
# Perform train_test_split on indices to guarantee identical split across both models
indices = np.arange(len(df))
y = df['label'].values

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)

y_train = y[train_idx]
y_test = y[test_idx]

train_c0 = np.sum(y_train == 0)
train_c1 = np.sum(y_train == 1)
test_c0 = np.sum(y_test == 0)
test_c1 = np.sum(y_test == 1)

print("\n[3] Train/Test Partition (train_test_split, test_size=0.20, stratify=y, random_state=42):")
print(f"    Training Set: {len(train_idx)} samples | Class 0 (STABLE): {train_c0} ({train_c0/len(train_idx)*100:.1f}%), Class 1 (CHATTER): {train_c1} ({train_c1/len(train_idx)*100:.1f}%)")
print(f"    Test Set:     {len(test_idx)} samples | Class 0 (STABLE): {test_c0} ({test_c0/len(test_idx)*100:.1f}%), Class 1 (CHATTER): {test_c1} ({test_c1/len(test_idx)*100:.1f}%)")

# Matrices for Model 1 (12 features)
X12_train = df.iloc[train_idx][FEATURES_12]
X12_test = df.iloc[test_idx][FEATURES_12]

# Matrices for Model 2 (7 features)
X7_train = df.iloc[train_idx][FEATURES_7]
X7_test = df.iloc[test_idx][FEATURES_7]

# ---------------------------------------------------------
# 4. XGBoost Model Configuration
# ---------------------------------------------------------
# Note: XGBoost uses tree-based splitting along orthogonal axes, making it invariant
# to monotonic transformations. StandardScaler provides no split advantage.
# Parameters below are a deliberately conservative baseline tailored for N ~ 80-100,
# using shallow depth, conservative learning rate, subsampling, and L1/L2 regularization
# to mitigate small-sample overfitting. They are NOT tuned on the test set.
xgb_params = {
    'n_estimators': 60,
    'max_depth': 3,
    'learning_rate': 0.05,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'eval_metric': 'logloss',
    'random_state': RANDOM_STATE,
    'use_label_encoder': False
}

print("\n[4] XGBoost Model Configuration (Deliberately Conservative Baseline):")
for k, v in xgb_params.items():
    print(f"    - {k}: {v}")

# ---------------------------------------------------------
# 5. Training, Cross-Validation & Test Evaluation Function
# ---------------------------------------------------------
def run_model_pipeline(model_name, X_train, y_train, X_test, y_test, feature_names):
    print("\n" + "=" * 60)
    print(f"RUNNING PIPELINE: {model_name} ({len(feature_names)} Features)")
    print("=" * 60)
    
    # 1. Instantiate Model
    model = XGBClassifier(**xgb_params)
    
    # 2. Stratified 5-Fold Cross-Validation on Training Partition Only
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    cv_results = cross_validate(
        XGBClassifier(**xgb_params),
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=False
    )
    
    cv_acc_mean = cv_results['test_accuracy'].mean()
    cv_acc_std = cv_results['test_accuracy'].std()
    cv_f1_mean = cv_results['test_f1'].mean()
    cv_f1_std = cv_results['test_f1'].std()
    cv_roc_mean = cv_results['test_roc_auc'].mean()
    cv_roc_std = cv_results['test_roc_auc'].std()
    
    # 3. Train Final Model on Full Training Partition
    model.fit(X_train, y_train)
    
    # 4. Training Set Evaluation (for diagnosing overfitting)
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    
    train_acc = accuracy_score(y_train, y_train_pred)
    train_prec = precision_score(y_train, y_train_pred, zero_division=0)
    train_rec = recall_score(y_train, y_train_pred, zero_division=0)
    train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
    train_roc = roc_auc_score(y_train, y_train_proba)
    
    # 5. Held-Out 20% Test Set Evaluation (strictly evaluated once)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    test_acc = accuracy_score(y_test, y_test_pred)
    test_prec = precision_score(y_test, y_test_pred, zero_division=0)
    test_rec = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    test_roc = roc_auc_score(y_test, y_test_proba)
    test_cm = confusion_matrix(y_test, y_test_pred)
    test_report = classification_report(y_test, y_test_pred, target_names=["STABLE (0)", "CHATTER (1)"], digits=4)
    
    # 6. Feature Importances (Gain and Weight)
    booster = model.get_booster()
    score_gain = booster.get_score(importance_type='gain')
    score_weight = booster.get_score(importance_type='weight')
    
    # Create complete feature importance dataframe
    feat_imp = []
    for f in feature_names:
        feat_imp.append({
            'feature': f,
            'importance_gain': score_gain.get(f, 0.0),
            'importance_weight': score_weight.get(f, 0.0)
        })
    df_feat_imp = pd.DataFrame(feat_imp).sort_values(by='importance_gain', ascending=False).reset_index(drop=True)
    
    # Print Diagnostics
    print(f"\n--- [TRAIN SET METRICS (N={len(y_train)})] ---")
    print(f"Accuracy:  {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"Precision: {train_prec:.4f}")
    print(f"Recall:    {train_rec:.4f}")
    print(f"F1-Score:  {train_f1:.4f}")
    print(f"ROC-AUC:   {train_roc:.4f}")
    
    print(f"\n--- [5-FOLD CV ON TRAINING SET ONLY] ---")
    print(f"CV Accuracy: {cv_acc_mean:.4f} ± {cv_acc_std:.4f}")
    print(f"CV F1-Score: {cv_f1_mean:.4f} ± {cv_f1_std:.4f}")
    print(f"CV ROC-AUC:  {cv_roc_mean:.4f} ± {cv_roc_std:.4f}")
    
    print(f"\n--- [HELD-OUT 20% TEST SET METRICS (N={len(y_test)})] ---")
    print(f"Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall:    {test_rec:.4f}")
    print(f"F1-Score:  {test_f1:.4f}")
    print(f"ROC-AUC:   {test_roc:.4f}")
    
    print(f"\nConfusion Matrix (Test Set):")
    print(f" [[TN={test_cm[0,0]}  FP={test_cm[0,1]}]")
    print(f"  [FN={test_cm[1,0]}  TP={test_cm[1,1]}]]")
    
    print(f"\nClassification Report (Test Set):\n{test_report}")
    
    print(f"Top 5 Features by Gain:")
    for idx, r in df_feat_imp.head(5).iterrows():
        print(f"  {idx+1}. {r['feature']:<32} | Gain: {r['importance_gain']:.4f} | Weight: {r['importance_weight']:.0f}")
        
    return {
        'model_name': model_name,
        'model': model,
        'feature_names': feature_names,
        'train_metrics': {
            'accuracy': train_acc,
            'precision': train_prec,
            'recall': train_rec,
            'f1': train_f1,
            'roc_auc': train_roc
        },
        'cv_metrics': {
            'accuracy_mean': cv_acc_mean,
            'accuracy_std': cv_acc_std,
            'f1_mean': cv_f1_mean,
            'f1_std': cv_f1_std,
            'roc_auc_mean': cv_roc_mean,
            'roc_auc_std': cv_roc_std
        },
        'test_metrics': {
            'accuracy': test_acc,
            'precision': test_prec,
            'recall': test_rec,
            'f1': test_f1,
            'roc_auc': test_roc,
            'confusion_matrix': test_cm.tolist(),
            'y_true': y_test.tolist(),
            'y_pred': y_test_pred.tolist(),
            'y_proba': y_test_proba.tolist()
        },
        'feature_importance': df_feat_imp
    }

# ---------------------------------------------------------
# 6. Execute Both Models
# ---------------------------------------------------------
results_12 = run_model_pipeline("XGBoost-12", X12_train, y_train, X12_test, y_test, FEATURES_12)
results_7 = run_model_pipeline("XGBoost-7", X7_train, y_train, X7_test, y_test, FEATURES_7)

# ---------------------------------------------------------
# 7. Comparison Summary Table
# ---------------------------------------------------------
print("\n" + "=" * 80)
print("MODEL COMPARISON TABLE")
print("=" * 80)

metrics_rows = [
    ("Train Accuracy", f"{results_12['train_metrics']['accuracy']:.4f}", f"{results_7['train_metrics']['accuracy']:.4f}"),
    ("Train F1-Score", f"{results_12['train_metrics']['f1']:.4f}", f"{results_7['train_metrics']['f1']:.4f}"),
    ("Train ROC-AUC", f"{results_12['train_metrics']['roc_auc']:.4f}", f"{results_7['train_metrics']['roc_auc']:.4f}"),
    ("CV Accuracy (5-Fold Mean±Std)", f"{results_12['cv_metrics']['accuracy_mean']:.4f} ± {results_12['cv_metrics']['accuracy_std']:.4f}", f"{results_7['cv_metrics']['accuracy_mean']:.4f} ± {results_7['cv_metrics']['accuracy_std']:.4f}"),
    ("CV F1-Score (5-Fold Mean±Std)", f"{results_12['cv_metrics']['f1_mean']:.4f} ± {results_12['cv_metrics']['f1_std']:.4f}", f"{results_7['cv_metrics']['f1_mean']:.4f} ± {results_7['cv_metrics']['f1_std']:.4f}"),
    ("CV ROC-AUC (5-Fold Mean±Std)", f"{results_12['cv_metrics']['roc_auc_mean']:.4f} ± {results_12['cv_metrics']['roc_auc_std']:.4f}", f"{results_7['cv_metrics']['roc_auc_mean']:.4f} ± {results_7['cv_metrics']['roc_auc_std']:.4f}"),
    ("Test Accuracy", f"{results_12['test_metrics']['accuracy']:.4f}", f"{results_7['test_metrics']['accuracy']:.4f}"),
    ("Test Precision", f"{results_12['test_metrics']['precision']:.4f}", f"{results_7['test_metrics']['precision']:.4f}"),
    ("Test Recall", f"{results_12['test_metrics']['recall']:.4f}", f"{results_7['test_metrics']['recall']:.4f}"),
    ("Test F1-Score", f"{results_12['test_metrics']['f1']:.4f}", f"{results_7['test_metrics']['f1']:.4f}"),
    ("Test ROC-AUC", f"{results_12['test_metrics']['roc_auc']:.4f}", f"{results_7['test_metrics']['roc_auc']:.4f}"),
    ("Test Confusion Matrix [TN,FP/FN,TP]", f"[{results_12['test_metrics']['confusion_matrix'][0]},{results_12['test_metrics']['confusion_matrix'][1]}]", f"[{results_7['test_metrics']['confusion_matrix'][0]},{results_7['test_metrics']['confusion_matrix'][1]}]")
]

df_comparison = pd.DataFrame(metrics_rows, columns=["Metric", "XGBoost-12", "XGBoost-7"])
print(df_comparison.to_string(index=False))

# ---------------------------------------------------------
# 8. Test Sample-Level Predictions
# ---------------------------------------------------------
print("\n" + "=" * 80)
print("TEST SET PREDICTED PROBABILITIES & ACTUAL LABELS")
print("=" * 80)
df_test_preds = pd.DataFrame({
    'Sample_Idx': test_idx,
    'Actual_Label': y_test,
    'Class_Name': ['CHATTER' if val == 1 else 'STABLE' for val in y_test],
    'XGB12_Prob_Chatter': results_12['test_metrics']['y_proba'],
    'XGB12_Pred': results_12['test_metrics']['y_pred'],
    'XGB7_Prob_Chatter': results_7['test_metrics']['y_proba'],
    'XGB7_Pred': results_7['test_metrics']['y_pred']
})
print(df_test_preds.to_string(index=False))

# ---------------------------------------------------------
# 9. Save Models, CSVs & JSON Artifacts
# ---------------------------------------------------------
# Save XGBoost model JSONs
model12_path = os.path.join(MODELS_DIR, "xgboost_12_features.json")
model7_path = os.path.join(MODELS_DIR, "xgboost_7_features.json")
results_12['model'].save_model(model12_path)
results_7['model'].save_model(model7_path)
print(f"\n[+] Saved Model 1 to: {model12_path}")
print(f"[+] Saved Model 2 to: {model7_path}")

# Save comparison table CSV
csv_comparison_path = os.path.join(BASE_DIR, "xgboost_model_comparison_results.csv")
df_comparison.to_csv(csv_comparison_path, index=False)
print(f"[+] Saved Comparison CSV to: {csv_comparison_path}")

# Save test predictions CSV
csv_preds_path = os.path.join(BASE_DIR, "xgboost_test_predictions.csv")
df_test_preds.to_csv(csv_preds_path, index=False)
print(f"[+] Saved Predictions CSV to: {csv_preds_path}")

# Save detailed JSON summary
summary_data = {
    'dataset_info': {
        'total_samples': len(df),
        'train_samples': len(train_idx),
        'test_samples': len(test_idx),
        'train_class_counts': {'stable_0': int(train_c0), 'chatter_1': int(train_c1)},
        'test_class_counts': {'stable_0': int(test_c0), 'chatter_1': int(test_c1)},
        'random_state': RANDOM_STATE,
        'test_size': 0.20
    },
    'xgboost_baseline_parameters': xgb_params,
    'xgboost_12': {
        'features': FEATURES_12,
        'train_metrics': results_12['train_metrics'],
        'cv_metrics': results_12['cv_metrics'],
        'test_metrics': {
            k: v for k, v in results_12['test_metrics'].items() if k not in ['y_true', 'y_pred', 'y_proba']
        },
        'feature_importance_top5': results_12['feature_importance'].head(5).to_dict(orient='records')
    },
    'xgboost_7': {
        'features': FEATURES_7,
        'train_metrics': results_7['train_metrics'],
        'cv_metrics': results_7['cv_metrics'],
        'test_metrics': {
            k: v for k, v in results_7['test_metrics'].items() if k not in ['y_true', 'y_pred', 'y_proba']
        },
        'feature_importance_top5': results_7['feature_importance'].head(5).to_dict(orient='records')
    }
}

json_summary_path = os.path.join(BASE_DIR, "xgboost_model_comparison_results.json")
with open(json_summary_path, 'w') as f:
    json.dump(summary_data, f, indent=4)
print(f"[+] Saved JSON Results Summary to: {json_summary_path}")

# ---------------------------------------------------------
# 10. Generate High-Resolution Diagnostic Plots
# ---------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig = plt.figure(figsize=(18, 12))

# Subplot 1: ROC Curves (Test Set)
ax1 = fig.add_subplot(2, 3, 1)
fpr12, tpr12, _ = roc_curve(y_test, results_12['test_metrics']['y_proba'])
fpr7, tpr7, _ = roc_curve(y_test, results_7['test_metrics']['y_proba'])
ax1.plot(fpr12, tpr12, label=f"XGB-12 (AUC = {results_12['test_metrics']['roc_auc']:.3f})", color='#1f77b4', lw=2.5)
ax1.plot(fpr7, tpr7, label=f"XGB-7 (AUC = {results_7['test_metrics']['roc_auc']:.3f})", color='#ff7f0e', lw=2.5, linestyle='--')
ax1.plot([0, 1], [0, 1], color='gray', linestyle=':', lw=1.5, label='Random Chance')
ax1.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
ax1.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11, fontweight='bold')
ax1.set_title('Test Set ROC Curves (N=21)', fontsize=12, fontweight='bold')
ax1.legend(loc='lower right', frameon=True)
ax1.grid(True, linestyle='--', alpha=0.6)

# Subplot 2: Confusion Matrix - XGBoost-12
ax2 = fig.add_subplot(2, 3, 2)
cm12 = np.array(results_12['test_metrics']['confusion_matrix'])
sns.heatmap(cm12, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax2,
            xticklabels=['STABLE (0)', 'CHATTER (1)'],
            yticklabels=['STABLE (0)', 'CHATTER (1)'])
ax2.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
ax2.set_ylabel('True Label', fontsize=11, fontweight='bold')
ax2.set_title(f"XGBoost-12 Confusion Matrix\n(Acc: {results_12['test_metrics']['accuracy']:.3f}, F1: {results_12['test_metrics']['f1']:.3f})", fontsize=12, fontweight='bold')

# Subplot 3: Confusion Matrix - XGBoost-7
ax3 = fig.add_subplot(2, 3, 3)
cm7 = np.array(results_7['test_metrics']['confusion_matrix'])
sns.heatmap(cm7, annot=True, fmt='d', cmap='Oranges', cbar=False, ax=ax3,
            xticklabels=['STABLE (0)', 'CHATTER (1)'],
            yticklabels=['STABLE (0)', 'CHATTER (1)'])
ax3.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
ax3.set_ylabel('True Label', fontsize=11, fontweight='bold')
ax3.set_title(f"XGBoost-7 Confusion Matrix\n(Acc: {results_7['test_metrics']['accuracy']:.3f}, F1: {results_7['test_metrics']['f1']:.3f})", fontsize=12, fontweight='bold')

# Subplot 4: Feature Importance (Gain) - XGBoost-12
ax4 = fig.add_subplot(2, 3, 4)
imp12 = results_12['feature_importance'].sort_values(by='importance_gain', ascending=True)
ax4.barh(imp12['feature'], imp12['importance_gain'], color='#1f77b4', edgecolor='black', alpha=0.85)
ax4.set_xlabel('Average Feature Gain', fontsize=11, fontweight='bold')
ax4.set_title('XGBoost-12 Feature Importance (Gain)', fontsize=12, fontweight='bold')
ax4.grid(True, linestyle='--', alpha=0.5)

# Subplot 5: Feature Importance (Gain) - XGBoost-7
ax5 = fig.add_subplot(2, 3, 5)
imp7 = results_7['feature_importance'].sort_values(by='importance_gain', ascending=True)
ax5.barh(imp7['feature'], imp7['importance_gain'], color='#ff7f0e', edgecolor='black', alpha=0.85)
ax5.set_xlabel('Average Feature Gain', fontsize=11, fontweight='bold')
ax5.set_title('XGBoost-7 Feature Importance (Gain)', fontsize=12, fontweight='bold')
ax5.grid(True, linestyle='--', alpha=0.5)

# Subplot 6: Metric Comparison (Train vs 5-Fold CV vs Test)
ax6 = fig.add_subplot(2, 3, 6)
metrics_names = ['Accuracy', 'F1-Score', 'ROC-AUC']
x_idx = np.arange(len(metrics_names))
bar_width = 0.2

train_vals12 = [results_12['train_metrics']['accuracy'], results_12['train_metrics']['f1'], results_12['train_metrics']['roc_auc']]
cv_vals12 = [results_12['cv_metrics']['accuracy_mean'], results_12['cv_metrics']['f1_mean'], results_12['cv_metrics']['roc_auc_mean']]
test_vals12 = [results_12['test_metrics']['accuracy'], results_12['test_metrics']['f1'], results_12['test_metrics']['roc_auc']]

train_vals7 = [results_7['train_metrics']['accuracy'], results_7['train_metrics']['f1'], results_7['train_metrics']['roc_auc']]
cv_vals7 = [results_7['cv_metrics']['accuracy_mean'], results_7['cv_metrics']['f1_mean'], results_7['cv_metrics']['roc_auc_mean']]
test_vals7 = [results_7['test_metrics']['accuracy'], results_7['test_metrics']['f1'], results_7['test_metrics']['roc_auc']]

ax6.bar(x_idx - 0.25, cv_vals12, width=bar_width, yerr=[results_12['cv_metrics']['accuracy_std'], results_12['cv_metrics']['f1_std'], results_12['cv_metrics']['roc_auc_std']],
        capsize=4, label='XGB-12 (5-Fold CV Train)', color='#1f77b4', alpha=0.7)
ax6.bar(x_idx - 0.05, test_vals12, width=bar_width, label='XGB-12 (Test 20%)', color='#1f77b4', hatch='//')

ax6.bar(x_idx + 0.15, cv_vals7, width=bar_width, yerr=[results_7['cv_metrics']['accuracy_std'], results_7['cv_metrics']['f1_std'], results_7['cv_metrics']['roc_auc_std']],
        capsize=4, label='XGB-7 (5-Fold CV Train)', color='#ff7f0e', alpha=0.7)
ax6.bar(x_idx + 0.35, test_vals7, width=bar_width, label='XGB-7 (Test 20%)', color='#ff7f0e', hatch='//')

ax6.set_xticks(x_idx)
ax6.set_xticklabels(metrics_names, fontsize=11, fontweight='bold')
ax6.set_ylim(0.5, 1.05)
ax6.set_ylabel('Score', fontsize=11, fontweight='bold')
ax6.set_title('Cross-Validation vs Test Generalization', fontsize=12, fontweight='bold')
ax6.legend(loc='lower left', fontsize=8, frameon=True)
ax6.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plots_path = os.path.join(BASE_DIR, "xgboost_evaluation_plots.png")
plt.savefig(plots_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"[+] Saved High-Resolution Evaluation Plots to: {plots_path}")

print("\n" + "=" * 80)
print("EXPERIMENT EXECUTION COMPLETE")
print("=" * 80)
