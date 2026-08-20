"""
Master 9,160-Cut Multi-Model Benchmark & Cross-Tool Generalization Study
Project: Digital Machining Database (Physics-Informed Chatter Detection)

Evaluates:
1. XGBoost-12, XGBoost-7 (Pruned), LightGBM, Random Forest, MLP Neural Net
2. Dual Evaluation Protocols:
   - Protocol A: Stratified 5-Fold Cross-Validation (Standard ML baseline)
   - Protocol B: GroupKFold Cross-Validation across 41 dynamic dataset configurations (True Generalization)
3. SHAP Explainability & Global Feature Importance
4. Publication-Ready Multi-Panel Diagnostic Figures
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Set seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Paths
BASE_DIR = r"D:\tony dataset"
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

PRUNED_FEATURE_COLUMNS_7 = [
    "kurtosis_1st_derivative",
    "ar_coeff_2",
    "d2_energy",
    "multi_axis_energy_asymmetry",
    "coherence_at_dominant_resonant",
    "impulse_factor",
    "phase_space_ellipsicity",
]


def load_and_validate():
    print("=" * 80)
    print("DIGITAL MACHINING AI: MASTER 9,160-CUT BENCHMARK & GENERALIZATION STUDY")
    print("=" * 80)

    if not os.path.exists(MASTER_CSV):
        raise FileNotFoundError(f"Master CSV missing at: {MASTER_CSV}")

    df = pd.read_csv(MASTER_CSV)
    print(f"\n[1] Loaded Master Dataset: {MASTER_CSV}")
    print(f"    Total Rows (Samples): {df.shape[0]}")
    print(f"    Unique Datasets (Configurations): {df['dataset_id'].nunique()} -> {sorted(df['dataset_id'].unique().tolist())}")
    
    # Check classes
    counts = df['label'].value_counts().to_dict()
    print(f"    Class Balance: 0 (Stable) = {counts.get(0, 0)} ({counts.get(0, 0)/len(df)*100:.1f}%), "
          f"1 (Chatter) = {counts.get(1, 0)} ({counts.get(1, 0)/len(df)*100:.1f}%)")
    
    # Handle any potential NaNs or Infs
    for feat in FEATURE_COLUMNS_12:
        df[feat] = df[feat].replace([np.inf, -np.inf], np.nan)
        median_val = df[feat].median()
        df[feat] = df[feat].fillna(median_val)
        
    return df


def get_models():
    return {
        "XGBoost-12": XGBClassifier(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.07,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "XGBoost-7": XGBClassifier(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.07,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "LightGBM-12": LGBMClassifier(
            n_estimators=250,
            max_depth=7,
            num_leaves=40,
            learning_rate=0.07,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
        ),
        "RandomForest-12": RandomForestClassifier(
            n_estimators=250,
            max_depth=14,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "MLP_NeuralNet-12": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            max_iter=350,
            early_stopping=True,
            random_state=RANDOM_STATE
        )
    }


def evaluate_cv(df):
    y = df['label'].values
    groups = df['dataset_id'].values
    
    models = get_models()
    benchmark_results = {
        "stratified_cv": {},
        "group_kfold_cv": {}
    }
    
    print("\n[2] Executing Protocol A: Stratified 5-Fold Cross-Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    for name, _ in models.items():
        feat_cols = PRUNED_FEATURE_COLUMNS_7 if "7" in name else FEATURE_COLUMNS_12
        X = df[feat_cols].values
        
        accs, precs, recs, f1s, rocs, prs = [], [], [], [], [], []
        oof_probs = np.zeros(len(y))
        
        for train_idx, val_idx in skf.split(X, y):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va, y_val = X[val_idx], y[val_idx]
            
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_va_sc = scaler.transform(X_va)
            
            clf = get_models()[name]
            if "MLP" in name:
                clf.fit(X_tr_sc, y_tr)
                probs = clf.predict_proba(X_va_sc)[:, 1]
            else:
                clf.fit(X_tr, y_tr)
                probs = clf.predict_proba(X_va)[:, 1]
                
            preds = (probs >= 0.5).astype(int)
            oof_probs[val_idx] = probs
            
            accs.append(accuracy_score(y_val, preds))
            precs.append(precision_score(y_val, preds, zero_division=0))
            recs.append(recall_score(y_val, preds, zero_division=0))
            f1s.append(f1_score(y_val, preds, zero_division=0))
            rocs.append(roc_auc_score(y_val, probs))
            prs.append(average_precision_score(y_val, probs))
            
        benchmark_results["stratified_cv"][name] = {
            "accuracy": {"mean": float(np.mean(accs)), "std": float(np.std(accs))},
            "precision": {"mean": float(np.mean(precs)), "std": float(np.std(precs))},
            "recall": {"mean": float(np.mean(recs)), "std": float(np.std(recs))},
            "f1": {"mean": float(np.mean(f1s)), "std": float(np.std(f1s))},
            "roc_auc": {"mean": float(np.mean(rocs)), "std": float(np.std(rocs))},
            "pr_auc": {"mean": float(np.mean(prs)), "std": float(np.std(prs))},
            "oof_probs": oof_probs.tolist()
        }
        print(f"    {name:<18} | Acc: {np.mean(accs):.4f} +/- {np.std(accs):.4f} | F1: {np.mean(f1s):.4f} | ROC-AUC: {np.mean(rocs):.4f}")

    print("\n[3] Executing Protocol B: GroupKFold Cross-Validation (Cross-Tool Generalization across 41 datasets)...")
    gkf = GroupKFold(n_splits=5)
    
    for name, _ in models.items():
        feat_cols = PRUNED_FEATURE_COLUMNS_7 if "7" in name else FEATURE_COLUMNS_12
        X = df[feat_cols].values
        
        accs, precs, recs, f1s, rocs, prs = [], [], [], [], [], []
        oof_probs = np.zeros(len(y))
        
        for train_idx, val_idx in gkf.split(X, y, groups=groups):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va, y_val = X[val_idx], y[val_idx]
            
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_va_sc = scaler.transform(X_va)
            
            clf = get_models()[name]
            if "MLP" in name:
                clf.fit(X_tr_sc, y_tr)
                probs = clf.predict_proba(X_va_sc)[:, 1]
            else:
                clf.fit(X_tr, y_tr)
                probs = clf.predict_proba(X_va)[:, 1]
                
            preds = (probs >= 0.5).astype(int)
            oof_probs[val_idx] = probs
            
            accs.append(accuracy_score(y_val, preds))
            precs.append(precision_score(y_val, preds, zero_division=0))
            recs.append(recall_score(y_val, preds, zero_division=0))
            f1s.append(f1_score(y_val, preds, zero_division=0))
            rocs.append(roc_auc_score(y_val, probs))
            prs.append(average_precision_score(y_val, probs))
            
        benchmark_results["group_kfold_cv"][name] = {
            "accuracy": {"mean": float(np.mean(accs)), "std": float(np.std(accs))},
            "precision": {"mean": float(np.mean(precs)), "std": float(np.std(precs))},
            "recall": {"mean": float(np.mean(recs)), "std": float(np.std(recs))},
            "f1": {"mean": float(np.mean(f1s)), "std": float(np.std(f1s))},
            "roc_auc": {"mean": float(np.mean(rocs)), "std": float(np.std(rocs))},
            "pr_auc": {"mean": float(np.mean(prs)), "std": float(np.std(prs))},
            "oof_probs": oof_probs.tolist()
        }
        print(f"    {name:<18} | Acc: {np.mean(accs):.4f} +/- {np.std(accs):.4f} | F1: {np.mean(f1s):.4f} | ROC-AUC: {np.mean(rocs):.4f}")

    return benchmark_results, y


def train_and_export_master_models(df):
    print("\n[4] Fitting and Serializing Final Master Models on All 9,160 Cuts...")
    y = df['label'].values
    X_12 = df[FEATURE_COLUMNS_12].values
    X_7 = df[PRUNED_FEATURE_COLUMNS_7].values
    
    # Save Scaler
    scaler_12 = StandardScaler()
    X_12_sc = scaler_12.fit_transform(X_12)
    joblib.dump(scaler_12, os.path.join(MODELS_DIR, "scaler_12.joblib"))
    
    # Fit XGBoost-12
    xgb12 = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    xgb12.fit(X_12, y)
    joblib.dump(xgb12, os.path.join(MODELS_DIR, "xgboost_12_master.joblib"))
    
    # Fit LightGBM-12
    lgb12 = LGBMClassifier(
        n_estimators=300,
        max_depth=7,
        num_leaves=40,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1
    )
    lgb12.fit(X_12, y)
    joblib.dump(lgb12, os.path.join(MODELS_DIR, "lightgbm_12_master.joblib"))
    
    # Fit RandomForest-12
    rf12 = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf12.fit(X_12, y)
    joblib.dump(rf12, os.path.join(MODELS_DIR, "randomforest_12_master.joblib"))
    
    # Fit MLP-12
    mlp12 = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        max_iter=400,
        random_state=RANDOM_STATE
    )
    mlp12.fit(X_12_sc, y)
    joblib.dump(mlp12, os.path.join(MODELS_DIR, "mlp_12_master.joblib"))
    
    print("    [+] All master model artifacts serialized to D:\\tony dataset\\models\\")
    return xgb12


def generate_publication_plots(benchmark_results, y, df, final_xgb):
    print("\n[5] Generating Publication-Ready Multi-Panel Diagnostic Figures...")
    sns.set_theme(style="whitegrid", palette="muted")
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)
    
    models = list(benchmark_results["stratified_cv"].keys())
    
    # Panel 1: Accuracy & F1 Comparison (Stratified CV vs GroupKFold)
    ax1 = fig.add_subplot(gs[0, 0])
    strat_acc = [benchmark_results["stratified_cv"][m]["accuracy"]["mean"] * 100 for m in models]
    group_acc = [benchmark_results["group_kfold_cv"][m]["accuracy"]["mean"] * 100 for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    ax1.bar(x - width/2, strat_acc, width, label="Standard Stratified 5-Fold", color="#3b82f6", alpha=0.9)
    ax1.bar(x + width/2, group_acc, width, label="GroupKFold (Cross-Tool)", color="#f59e0b", alpha=0.9)
    ax1.set_ylabel("Accuracy (%)", fontsize=12, fontweight='bold')
    ax1.set_title("A. Cross-Tool Generalization vs Random CV", fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace("-12", "").replace("_NeuralNet", "") for m in models], rotation=15, ha='right')
    ax1.set_ylim(75, 100)
    ax1.legend(loc="lower right")
    
    # Panel 2: ROC Curves (Stratified CV OOF)
    ax2 = fig.add_subplot(gs[0, 1])
    for m in models:
        probs = np.array(benchmark_results["stratified_cv"][m]["oof_probs"])
        fpr, tpr, _ = roc_curve(y, probs)
        auc_val = benchmark_results["stratified_cv"][m]["roc_auc"]["mean"]
        ax2.plot(fpr, tpr, label=f"{m} (AUC = {auc_val:.3f})", lw=2)
    ax2.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6)
    ax2.set_xlabel("False Positive Rate", fontsize=11)
    ax2.set_ylabel("True Positive Rate", fontsize=11)
    ax2.set_title("B. Out-Of-Fold ROC Curves (9,160 Cuts)", fontsize=13, fontweight='bold')
    ax2.legend(loc="lower right", fontsize=9)
    
    # Panel 3: Precision-Recall Curves
    ax3 = fig.add_subplot(gs[0, 2])
    for m in models:
        probs = np.array(benchmark_results["stratified_cv"][m]["oof_probs"])
        prec, rec, _ = precision_recall_curve(y, probs)
        pr_auc = benchmark_results["stratified_cv"][m]["pr_auc"]["mean"]
        ax3.plot(rec, prec, label=f"{m} (PR-AUC = {pr_auc:.3f})", lw=2)
    ax3.set_xlabel("Recall", fontsize=11)
    ax3.set_ylabel("Precision", fontsize=11)
    ax3.set_title("C. Precision-Recall Dynamics", fontsize=13, fontweight='bold')
    ax3.legend(loc="lower left", fontsize=9)
    
    # Panel 4: Confusion Matrix for Best Model (XGBoost-12 Stratified)
    ax4 = fig.add_subplot(gs[1, 0])
    best_probs = np.array(benchmark_results["stratified_cv"]["XGBoost-12"]["oof_probs"])
    best_preds = (best_probs >= 0.5).astype(int)
    cm = confusion_matrix(y, best_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax4,
                xticklabels=['Stable (0)', 'Chatter (1)'],
                yticklabels=['Stable (0)', 'Chatter (1)'])
    ax4.set_xlabel("Predicted Label", fontsize=11, fontweight='bold')
    ax4.set_ylabel("Ground Truth Label", fontsize=11, fontweight='bold')
    ax4.set_title("D. Confusion Matrix: XGBoost-12 (9,160 Cuts)", fontsize=13, fontweight='bold')
    
    # Panel 5: SHAP Global Feature Importance
    ax5 = fig.add_subplot(gs[1, 1:])
    X_sample = df[FEATURE_COLUMNS_12].sample(min(1500, len(df)), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(final_xgb)
    shap_values = explainer.shap_values(X_sample)
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    sorted_idx = np.argsort(mean_abs_shap)
    
    clean_names = [FEATURE_COLUMNS_12[i].replace("_", " ").title() for i in sorted_idx]
    y_pos = np.arange(len(sorted_idx))
    ax5.barh(y_pos, mean_abs_shap[sorted_idx], color="#0ea5e9", edgecolor="#0369a1", alpha=0.85)
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(clean_names, fontsize=10)
    ax5.set_xlabel("Mean Absolute SHAP Value (Impact on Chatter Output)", fontsize=11, fontweight='bold')
    ax5.set_title("E. Physics-Informed SHAP Feature Importance Ranking", fontsize=13, fontweight='bold')
    
    plot_path = os.path.join(BASE_DIR, "master_model_comparison_plots.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    [+] Saved master diagnostic plot to: {plot_path}")


def save_summary_table(benchmark_results):
    rows = []
    for model_name in benchmark_results["stratified_cv"].keys():
        s = benchmark_results["stratified_cv"][model_name]
        g = benchmark_results["group_kfold_cv"][model_name]
        rows.append({
            "Model": model_name,
            "Stratified_CV_Accuracy": f"{s['accuracy']['mean']*100:.2f}% ± {s['accuracy']['std']*100:.2f}%",
            "Stratified_CV_F1": f"{s['f1']['mean']:.4f}",
            "Stratified_CV_ROC_AUC": f"{s['roc_auc']['mean']:.4f}",
            "GroupKFold_Accuracy": f"{g['accuracy']['mean']*100:.2f}% ± {g['accuracy']['std']*100:.2f}%",
            "GroupKFold_F1": f"{g['f1']['mean']:.4f}",
            "GroupKFold_ROC_AUC": f"{g['roc_auc']['mean']:.4f}",
        })
    df_res = pd.DataFrame(rows)
    csv_path = os.path.join(BASE_DIR, "master_benchmark_results.csv")
    json_path = os.path.join(BASE_DIR, "master_benchmark_results.json")
    
    df_res.to_csv(csv_path, index=False)
    
    # Clean json export (strip large arrays)
    clean_json = {
        "stratified_cv": {
            m: {k: v for k, v in data.items() if k not in ["oof_probs"]}
            for m, data in benchmark_results["stratified_cv"].items()
        },
        "group_kfold_cv": {
            m: {k: v for k, v in data.items() if k not in ["oof_probs"]}
            for m, data in benchmark_results["group_kfold_cv"].items()
        }
    }
    with open(json_path, "w") as f:
        json.dump(clean_json, f, indent=2)
        
    print("\n" + "=" * 80)
    print("FINAL BENCHMARK COMPARISON TABLE (9,160 CUTS):")
    print("=" * 80)
    print(df_res.to_string(index=False))
    print("=" * 80)


def main():
    df = load_and_validate()
    benchmark_results, y = evaluate_cv(df)
    final_xgb = train_and_export_master_models(df)
    generate_publication_plots(benchmark_results, y, df, final_xgb)
    save_summary_table(benchmark_results)
    print("\n[*] MASTER BENCHMARK COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
