"""Unit tests for standalone 12-feature extraction module."""

import numpy as np
import pandas as pd
import pytest

from tony_dataset.constants import TIME_COLUMNS
from tony_dataset.experiment import Experiment
from tony_dataset.extract_12_features import (
    SELECTED_12_FEATURES,
    extract_12_features,
)


def create_synthetic_time_series_df(n_samples: int = 1000, fs: float = 12500.0) -> pd.DataFrame:
    """Helper to generate a synthetic signal dataframe with all 15 signal columns."""
    t = np.arange(n_samples) / fs
    data = {
        "RPM": np.full(n_samples, 3000.0),
        "Axial_Depth": np.full(n_samples, 0.005),
        "Time": t,
        "Tool_X_Displacement": 1e-5 * np.sin(2 * np.pi * 100 * t),
        "Tool_Y_Displacement": 1e-5 * np.cos(2 * np.pi * 100 * t),
        "Workpiece_X_Displacement": np.zeros(n_samples),
        "Workpiece_Y_Displacement": np.zeros(n_samples),
        "Tool_X_Velocity": 1e-3 * np.cos(2 * np.pi * 100 * t),
        "Tool_Y_Velocity": -1e-3 * np.sin(2 * np.pi * 100 * t),
        "Workpiece_X_Velocity": np.zeros(n_samples),
        "Workpiece_Y_Velocity": np.zeros(n_samples),
        "Tool_X_Acceleration": 1.0 * np.sin(2 * np.pi * 100 * t),
        "Tool_Y_Acceleration": 1.0 * np.cos(2 * np.pi * 100 * t),
        "Workpiece_X_Acceleration": np.zeros(n_samples),
        "Workpiece_Y_Acceleration": np.zeros(n_samples),
        "Force_X": 100.0 + 20.0 * np.sin(2 * np.pi * 200 * t),
        "Force_Y": 80.0 + 15.0 * np.cos(2 * np.pi * 200 * t),
    }
    return pd.DataFrame(data)[TIME_COLUMNS[2:]]


def test_extract_12_features_keys_and_values():
    df = create_synthetic_time_series_df(1000, 12500.0)
    exp = Experiment(
        signals=df,
        metadata={"sampling_rate_hz": 12500.0},
        rpm=3000.0,
        axial_depth=0.005,
        type="time_series",
        file_name="test_synth.h5",
    )

    res = extract_12_features(exp)

    assert len(res) == 12
    assert list(res.keys()) == SELECTED_12_FEATURES

    for k, v in res.items():
        assert isinstance(v, float)
        assert np.isfinite(v), f"Feature {k} is not finite: {v}"
