"""Standalone 12-Feature Extraction Module for Tony Schmitz Digital Machining Database.

This module provides a direct, reusable API to extract the EXACT 12 high-performance,
leakage-free selected candidate features from a single Digital Machining Database H5 dataset file.

Selected 12 Features:
  1. kurtosis_1st_derivative
  2. ar_coeff_2
  3. d2_energy
  4. skewness_1st_derivative
  5. multi_axis_energy_asymmetry
  6. d3_d4_subband_energy_ratio
  7. coherence_at_dominant_resonant
  8. impulse_factor
  9. cross_axis_peak_delay
  10. cross_spectral_centroid
  11. phase_space_ellipsicity
  12. bivariate_orbit_radius_ratio
"""

from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd

from tony_dataset.experiment import Experiment
from tony_dataset.features import (
    _extract_autoregressive,
    _extract_cross_channel,
    _extract_time_domain,
    _extract_wavelet,
    _extract_nonlinear,
    _get_signal,
)
from tony_dataset.loader import load_h5

SELECTED_12_FEATURES = [
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


def extract_12_features(
    target: Union[str, Path, Experiment], num_teeth: int = 4
) -> Dict[str, float]:
    """Extracts the exact 12 selected candidate features from an H5 dataset file or Experiment instance.

    Args:
        target: File path (.h5) or loaded Experiment instance.
        num_teeth: Number of tool cutting teeth for frequency calculations (default 4).

    Returns:
        Dictionary mapping feature names to their computed float values.
    """
    if isinstance(target, (str, Path)):
        exp = load_h5(target)
    elif isinstance(target, Experiment):
        exp = target
    else:
        raise TypeError(f"Expected file path or Experiment object, got {type(target).__name__}")

    df = exp.signals
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {k: np.nan for k in SELECTED_12_FEATURES}

    fs = float(exp.metadata.get("sampling_rate_hz", 0.0))
    if fs <= 0.0 and "Time" in df and len(df) > 1:
        t_arr = df["Time"].to_numpy()
        dt_med = float(np.median(np.diff(t_arr)))
        fs = 1.0 / dt_med if dt_med > 0.0 else 10000.0

    rpm = float(exp.rpm)

    force_x = _get_signal(df, "Force_X")
    force_y = _get_signal(df, "Force_Y")
    accel_x = _get_signal(df, "Tool_X_Acceleration")
    displ_x = _get_signal(df, "Tool_X_Displacement")
    displ_y = _get_signal(df, "Tool_Y_Displacement")
    vel_x = _get_signal(df, "Tool_X_Velocity")

    raw_features: Dict[str, float] = {}

    # Extract required feature groups using original exact implementations
    raw_features.update(_extract_time_domain(force_x, accel_x, fs))
    raw_features.update(
        _extract_cross_channel(
            force_x, force_y, accel_x, displ_x, displ_y, fs, rpm, num_teeth
        )
    )
    raw_features.update(_extract_wavelet(force_x, fs))
    raw_features.update(_extract_nonlinear(force_x, accel_x, displ_x, vel_x, fs))
    raw_features.update(_extract_autoregressive(force_x, fs))

    # Filter strictly to the 12 selected features
    out_dict: Dict[str, float] = {}
    for k in SELECTED_12_FEATURES:
        val = float(raw_features.get(k, np.nan))
        out_dict[k] = val if np.isfinite(val) else np.nan

    return out_dict


def extract_12_features_dataframe(
    filepaths: list[Union[str, Path]], num_teeth: int = 4
) -> pd.DataFrame:
    """Batch extracts the 12 selected candidate features across multiple H5 files.

    Args:
        filepaths: List of H5 dataset file paths.
        num_teeth: Number of tool teeth.

    Returns:
        pandas DataFrame with 12 feature columns and metadata for each file.
    """
    records = []
    for fp in filepaths:
        p = Path(fp)
        exp = load_h5(p)
        row = {
            "dataset_number": exp.dataset_number,
            "grid_point": exp.grid_point,
            "file_name": exp.file_name,
            "RPM": exp.rpm,
            "Axial_Depth": exp.axial_depth,
        }
        feats = extract_12_features(exp, num_teeth=num_teeth)
        row.update(feats)
        records.append(row)

    return pd.DataFrame(records)
