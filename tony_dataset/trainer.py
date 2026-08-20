"""
Multi-Model Training Engine for Digital Machining Chatter Classification.
Supports XGBoost, LightGBM, Random Forest, and Multi-Layer Perceptron (MLP)
under both StratifiedKFold and GroupKFold (Leave-One-Dataset-Out) protocols.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import StratifiedKFold, GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


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


class MachiningModelTrainer:
    """Trainer orchestrator for Digital Machining chatter detection."""

    def __init__(
        self,
        random_state: int = 42,
        models_dir: str = "models"
    ):
        self.random_state = random_state
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        self.scaler = StandardScaler()
        self.trained_models = {}

    def get_model_instances(self) -> Dict[str, Any]:
        """Instantiate standardized classifier configurations."""
        return {
            "XGBoost": XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                eval_metric="logloss",
                random_state=self.random_state,
                n_jobs=-1
            ),
            "LightGBM": LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                num_leaves=31,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            "MLP_NeuralNet": MLPClassifier(
                hidden_layer_sizes=(64, 32, 16),
                activation="relu",
                max_iter=300,
                early_stopping=True,
                random_state=self.random_state
            )
        }

    def evaluate_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict[str, float]:
        """Compute standard classification metrics."""
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5,
            "pr_auc": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5,
        }

    def run_stratified_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5
    ) -> Dict[str, Dict[str, Any]]:
        """Run Stratified K-Fold cross validation across all models."""
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        results = {}
        model_instances = self.get_model_instances()

        for name, _ in model_instances.items():
            fold_metrics = []
            oof_preds = np.zeros(len(y))
            oof_probs = np.zeros(len(y))

            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_tr, y_tr = X[train_idx], y[train_idx]
                X_va, y_val = X[val_idx], y[val_idx]

                scaler = StandardScaler()
                X_tr_sc = scaler.fit_transform(X_tr)
                X_va_sc = scaler.transform(X_va)

                clf = self.get_model_instances()[name]
                if name in ["MLP_NeuralNet"]:
                    clf.fit(X_tr_sc, y_tr)
                    probs = clf.predict_proba(X_va_sc)[:, 1]
                else:
                    clf.fit(X_tr, y_tr)
                    probs = clf.predict_proba(X_va)[:, 1]

                preds = (probs >= 0.5).astype(int)
                oof_preds[val_idx] = preds
                oof_probs[val_idx] = probs

                metrics = self.evaluate_predictions(y_val, preds, probs)
                fold_metrics.append(metrics)

            agg = {
                metric: {
                    "mean": float(np.mean([m[metric] for m in fold_metrics])),
                    "std": float(np.std([m[metric] for m in fold_metrics]))
                }
                for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
            }
            cm = confusion_matrix(y, oof_preds).tolist()
            results[name] = {
                "fold_metrics": fold_metrics,
                "summary": agg,
                "confusion_matrix": cm,
                "oof_probs": oof_probs.tolist(),
                "oof_preds": oof_preds.tolist(),
            }

        return results

    def run_group_kfold_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        n_splits: int = 5
    ) -> Dict[str, Dict[str, Any]]:
        """Run GroupKFold cross validation grouped by dataset_id (Cross-Tool Generalization)."""
        unique_groups = np.unique(groups)
        actual_splits = min(n_splits, len(unique_groups))
        gkf = GroupKFold(n_splits=actual_splits)
        results = {}
        model_instances = self.get_model_instances()

        for name, _ in model_instances.items():
            fold_metrics = []
            oof_preds = np.zeros(len(y))
            oof_probs = np.zeros(len(y))

            for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
                X_tr, y_tr = X[train_idx], y[train_idx]
                X_va, y_val = X[val_idx], y[val_idx]

                scaler = StandardScaler()
                X_tr_sc = scaler.fit_transform(X_tr)
                X_va_sc = scaler.transform(X_va)

                clf = self.get_model_instances()[name]
                if name in ["MLP_NeuralNet"]:
                    clf.fit(X_tr_sc, y_tr)
                    probs = clf.predict_proba(X_va_sc)[:, 1]
                else:
                    clf.fit(X_tr, y_tr)
                    probs = clf.predict_proba(X_va)[:, 1]

                preds = (probs >= 0.5).astype(int)
                oof_preds[val_idx] = preds
                oof_probs[val_idx] = probs

                metrics = self.evaluate_predictions(y_val, preds, probs)
                fold_metrics.append(metrics)

            agg = {
                metric: {
                    "mean": float(np.mean([m[metric] for m in fold_metrics])),
                    "std": float(np.std([m[metric] for m in fold_metrics]))
                }
                for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
            }
            cm = confusion_matrix(y, oof_preds).tolist()
            results[name] = {
                "fold_metrics": fold_metrics,
                "summary": agg,
                "confusion_matrix": cm,
                "oof_probs": oof_probs.tolist(),
                "oof_preds": oof_preds.tolist(),
            }

        return results

    def fit_and_save_final_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, str]:
        """Fit final models on the entire dataset and save serialized artifacts."""
        saved_paths = {}
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        scaler_path = os.path.join(self.models_dir, "feature_scaler.joblib")
        joblib.dump(scaler, scaler_path)
        saved_paths["scaler"] = scaler_path

        for name, clf in self.get_model_instances().items():
            if name in ["MLP_NeuralNet"]:
                clf.fit(X_scaled, y)
            else:
                clf.fit(X, y)
            
            model_path = os.path.join(self.models_dir, f"{name.lower()}_master.joblib")
            joblib.dump(clf, model_path)
            saved_paths[name] = model_path

        meta_path = os.path.join(self.models_dir, "feature_metadata.json")
        with open(meta_path, "w") as f:
            json.dump({"features": feature_names, "n_features": len(feature_names)}, f, indent=2)
        saved_paths["metadata"] = meta_path

        return saved_paths
