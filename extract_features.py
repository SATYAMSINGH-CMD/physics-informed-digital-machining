"""Extract 49 Candidate Feature Matrix from Real Tony Schmitz Dataset 1.

This script scans the Dataset 1 directory, loads each time-series HDF5 experiment,
extracts the 49 candidate physics-informed features, builds `data/feature_matrix.csv`,
and generates a data quality report `data/feature_quality_report.csv`.
"""

import glob
import logging
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from tony_dataset.features import FEATURE_NAMES_49, extract_experiment_features
from tony_dataset.loader import load_h5

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default Dataset 1 Path
DATASET_DIR = r"G:\My Drive\Videos to clear my space\Dataset 1 h5"
OUTPUT_MATRIX_PATH = Path("data/feature_matrix.csv")
OUTPUT_QUALITY_PATH = Path("data/feature_quality_report.csv")


def main() -> None:
    """Main execution routine for feature matrix extraction."""
    logger.info(f"Scanning HDF5 dataset directory: {DATASET_DIR}")
    if not os.path.exists(DATASET_DIR):
        logger.error(f"Dataset directory does not exist: {DATASET_DIR}")
        sys.exit(1)

    # 1. Identify all HDF5 files
    all_h5_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.h5")))
    time_series_files = [f for f in all_h5_files if "time" in os.path.basename(f).lower()]
    stability_files = [f for f in all_h5_files if "stability" in os.path.basename(f).lower()]

    total_files_found = len(all_h5_files)
    time_series_count = len(time_series_files)

    logger.info(
        f"Found {total_files_found} HDF5 files total ({time_series_count} time-series, {len(stability_files)} stability boundary)"
    )

    rows = []
    loaded_count = 0
    extracted_count = 0
    failures = []

    # 2. Process each time-series file
    for filepath in time_series_files:
        fname = os.path.basename(filepath)
        try:
            # Load experiment using project loader
            exp = load_h5(filepath)
            loaded_count += 1

            # Validate required components
            df = exp.signals
            if df is None or df.empty:
                raise ValueError("Signals DataFrame is empty")

            required_cols = ["Time", "Force_X", "Force_Y", "Tool_X_Acceleration"]
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required signal columns: {missing_cols}")

            fs = exp.metadata.get("sampling_rate_hz", 0.0)
            if fs <= 0.0:
                raise ValueError("Failed to calculate valid sampling rate")

            # Extract 49 features
            feats = extract_experiment_features(exp, num_teeth=4)
            extracted_count += 1

            # Build matrix row
            row_dict = {
                "dataset_number": exp.dataset_number if exp.dataset_number is not None else np.nan,
                "grid_point": exp.grid_point if exp.grid_point is not None else np.nan,
                "file_name": exp.file_name,
                "RPM": exp.rpm,
                "Axial_Depth": exp.axial_depth,
            }
            # Append all 49 feature values in order
            for feat_name in FEATURE_NAMES_49:
                row_dict[feat_name] = feats.get(feat_name, np.nan)

            rows.append(row_dict)

        except Exception as exc:
            logger.error(f"Failed to process '{fname}': {exc}")
            failures.append({"file_name": fname, "reason": str(exc)})

    # 3. Create Feature Matrix DataFrame
    meta_cols = ["dataset_number", "grid_point", "file_name", "RPM", "Axial_Depth"]
    all_cols = meta_cols + FEATURE_NAMES_49
    matrix_df = pd.DataFrame(rows, columns=all_cols)

    OUTPUT_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    matrix_df.to_csv(OUTPUT_MATRIX_PATH, index=False)
    logger.info(f"Saved feature matrix to: {OUTPUT_MATRIX_PATH.resolve()}")

    # 4. Generate Feature Quality Report
    feature_stats = []
    feature_df = matrix_df[FEATURE_NAMES_49]

    total_nan_count = 0
    total_inf_count = 0

    for col in FEATURE_NAMES_49:
        series = feature_df[col]
        vals = series.to_numpy()

        nan_cnt = int(series.isna().sum())
        inf_cnt = int(np.isinf(vals).sum())
        valid_series = series.dropna()
        valid_cnt = len(valid_series)

        total_nan_count += nan_cnt
        total_inf_count += inf_cnt

        if valid_cnt > 0:
            min_val = float(valid_series.min())
            max_val = float(valid_series.max())
            mean_val = float(valid_series.mean())
            std_val = float(valid_series.std())
        else:
            min_val = np.nan
            max_val = np.nan
            mean_val = np.nan
            std_val = np.nan

        feature_stats.append(
            {
                "feature_name": col,
                "valid_values_count": valid_cnt,
                "nan_count": nan_cnt,
                "inf_count": inf_cnt,
                "min": min_val,
                "max": max_val,
                "mean": mean_val,
                "std": std_val,
            }
        )

    quality_df = pd.DataFrame(feature_stats)
    quality_df.to_csv(OUTPUT_QUALITY_PATH, index=False)
    logger.info(f"Saved feature quality report to: {OUTPUT_QUALITY_PATH.resolve()}")

    # 5. Print Concise Extraction Summary Report
    print("\n" + "=" * 60)
    print(" EXTRACTION SUMMARY REPORT: TONY DATASET 1")
    print("=" * 60)
    print(f" Total H5 files found in directory : {total_files_found}")
    print(f" Time-Series H5 files identified   : {time_series_count}")
    print(f" Stability Boundary H5 files       : {len(stability_files)}")
    print(f" Successfully loaded               : {loaded_count}")
    print(f" Successfully feature-extracted    : {extracted_count}")
    print(f" Failed experiments                : {len(failures)}")
    if failures:
        for f in failures:
            print(f"   - {f['file_name']}: {f['reason']}")
    print("-" * 60)
    print(f" Feature Matrix Rows               : {len(matrix_df)}")
    print(f" Feature Matrix Columns            : {len(matrix_df.columns)} ({len(meta_cols)} metadata + {len(FEATURE_NAMES_49)} features)")
    print(f" Total NaNs in Feature Matrix      : {total_nan_count}")
    print(f" Total Infs in Feature Matrix      : {total_inf_count}")
    print(f" Feature Matrix CSV Output         : {OUTPUT_MATRIX_PATH.resolve()}")
    print(f" Feature Quality CSV Output        : {OUTPUT_QUALITY_PATH.resolve()}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
