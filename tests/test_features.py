"""Unit tests for tony_dataset.features module."""

import unittest

import numpy as np
import pandas as pd

from tony_dataset.constants import DatasetType
from tony_dataset.experiment import Experiment
from tony_dataset.features import (
    FEATURE_NAMES_49,
    extract_experiment_features,
    extract_signals_features,
)


class TestFeaturesModule(unittest.TestCase):
    """Automated unit test suite for tony_dataset/features.py (49 candidate features)."""

    def setUp(self) -> None:
        """Create synthetic signal DataFrame for testing."""
        self.fs = 10000.0
        self.dt = 1.0 / self.fs
        self.t = np.arange(0.0, 1.0, self.dt)  # 10,000 samples (1.0 s)
        self.N = len(self.t)

        # Synthetic signals: sinusoidal force with harmonics + noise
        f_cut = 200.0  # 200 Hz
        force_x = 50.0 + 20.0 * np.sin(2 * np.pi * f_cut * self.t) + 5.0 * np.cos(4 * np.pi * f_cut * self.t)
        force_y = 30.0 + 15.0 * np.cos(2 * np.pi * f_cut * self.t)
        accel_x = 2.0 * np.sin(2 * np.pi * f_cut * self.t + 0.5)
        displ_x = 0.0001 * np.sin(2 * np.pi * f_cut * self.t)
        displ_y = 0.0001 * np.cos(2 * np.pi * f_cut * self.t)
        vel_x = 0.02 * np.cos(2 * np.pi * f_cut * self.t)

        self.df = pd.DataFrame(
            {
                "Time": self.t,
                "Force_X": force_x,
                "Force_Y": force_y,
                "Tool_X_Acceleration": accel_x,
                "Tool_X_Displacement": displ_x,
                "Tool_Y_Displacement": displ_y,
                "Tool_X_Velocity": vel_x,
            }
        )

        self.experiment = Experiment(
            signals=self.df,
            metadata={"sampling_rate_hz": self.fs},
            rpm=3000.0,  # f_tpf = 3000 * 4 / 60 = 200 Hz
            axial_depth=0.002,
            type=DatasetType.TIME_SERIES,
            file_name="synthetic_test.h5",
        )

    def test_feature_count_and_keys(self) -> None:
        """Verify that extract_experiment_features returns exactly 49 feature keys."""
        res = extract_experiment_features(self.experiment, num_teeth=4)
        self.assertIsInstance(res, dict)
        self.assertEqual(len(res), 49)
        self.assertEqual(len(FEATURE_NAMES_49), 49)
        self.assertEqual(set(res.keys()), set(FEATURE_NAMES_49))
        self.assertIs(self.experiment.features, res)

    def test_time_domain_formulas(self) -> None:
        """Test Group 1: 10 Time-Domain features on synthetic signal."""
        res = extract_experiment_features(self.experiment)
        self.assertGreater(res["kurtosis"], 0.0)
        self.assertGreater(res["crest_factor"], 1.0)
        self.assertGreater(res["margin_factor"], 1.0)
        self.assertGreater(res["shape_factor"], 1.0)
        self.assertGreater(res["impulse_factor"], 1.0)
        self.assertGreaterEqual(res["zero_crossing_rate"], 0.0)
        self.assertGreaterEqual(res["mean_crossing_rate"], 0.0)

    def test_frequency_domain_formulas(self) -> None:
        """Test Group 2: 8 Frequency-Domain spectral features."""
        res = extract_experiment_features(self.experiment)
        self.assertGreater(res["spectral_centroid"], 0.0)
        self.assertGreater(res["spectral_entropy"], 0.0)
        self.assertGreater(res["spectral_flatness"], 0.0)
        self.assertGreater(res["spectral_rolloff_85"], 0.0)
        self.assertGreaterEqual(res["off_harmonic_energy_ratio"], 0.0)

    def test_cross_channel_features(self) -> None:
        """Test Group 3: 7 Cross-Channel & multi-axis metrics."""
        res = extract_experiment_features(self.experiment)
        self.assertGreaterEqual(res["cross_correlation_coeff"], -1.0)
        self.assertLessEqual(res["cross_correlation_coeff"], 1.0)
        self.assertGreaterEqual(res["bivariate_orbit_radius_ratio"], 0.0)
        self.assertGreaterEqual(res["coherence_at_tpf"], 0.0)
        self.assertLessEqual(res["coherence_at_tpf"], 1.0)
        self.assertGreaterEqual(res["cross_spectral_centroid"], 0.0)

    def test_wavelet_features(self) -> None:
        """Test Group 4: 10 Discrete Wavelet Transform (db4, level 4) metrics."""
        res = extract_experiment_features(self.experiment)
        self.assertGreaterEqual(res["d1_energy"], 0.0)
        self.assertGreaterEqual(res["a4_energy"], 0.0)
        self.assertGreaterEqual(res["wavelet_energy_entropy"], 0.0)
        self.assertGreaterEqual(res["d1_relative_energy"], 0.0)
        self.assertLessEqual(res["d1_relative_energy"], 1.0)

    def test_nonlinear_features(self) -> None:
        """Test Group 5: 5 Nonlinear dynamics & phase space metrics."""
        res = extract_experiment_features(self.experiment)
        self.assertGreaterEqual(res["permutation_entropy"], 0.0)
        self.assertLessEqual(res["permutation_entropy"], 1.0)
        self.assertGreaterEqual(res["higuchi_fractal_dimension"], 1.0)
        self.assertGreaterEqual(res["katz_fractal_dimension"], 1.0)
        self.assertGreaterEqual(res["phase_space_ellipsicity"], 0.0)
        self.assertGreaterEqual(res["phase_space_area"], 0.0)

    def test_autoregressive_features(self) -> None:
        """Test Group 6: 5 Autoregressive & correlation lag metrics."""
        res = extract_experiment_features(self.experiment)
        self.assertIn("ar_coeff_1", res)
        self.assertIn("ar_coeff_2", res)
        self.assertIn("ar_coeff_3", res)
        self.assertGreaterEqual(res["ar_residual_variance"], 0.0)
        self.assertGreaterEqual(res["autocorr_first_zero_lag"], 0.0)

    def test_physics_features(self) -> None:
        """Test Group 7: 4 Physics-informed metrics."""
        res = extract_experiment_features(self.experiment, num_teeth=4)
        self.assertGreaterEqual(res["harmonic_peak_ratio"], 0.0)
        self.assertGreaterEqual(res["dominant_peak_tpf_ratio"], 0.0)
        self.assertGreaterEqual(res["transimpedance"], 0.0)
        self.assertGreaterEqual(res["acceleration_jerk_rms"], 0.0)

    def test_signals_helper(self) -> None:
        """Verify helper function extract_signals_features directly on DataFrame."""
        res = extract_signals_features(self.df, fs=self.fs, rpm=3000.0, num_teeth=4)
        self.assertEqual(len(res), 49)

    def test_zero_and_edge_cases(self) -> None:
        """Test robust handling of zero arrays, constant signals, NaNs, short signals."""
        zero_df = pd.DataFrame(
            {
                "Time": np.linspace(0.0, 0.1, 50),
                "Force_X": np.zeros(50),
                "Force_Y": np.zeros(50),
                "Tool_X_Acceleration": np.zeros(50),
            }
        )
        zero_exp = Experiment(
            signals=zero_df,
            metadata={"sampling_rate_hz": 1000.0},
            rpm=0.0,
            axial_depth=0.0,
            type=DatasetType.TIME_SERIES,
            file_name="zero_test.h5",
        )

        res = extract_experiment_features(zero_exp)
        self.assertEqual(len(res), 49)
        # All values should be valid floats or NaN (no unhandled exceptions)
        for k, v in res.items():
            self.assertIsInstance(v, float, f"Feature '{k}' is not float: {type(v)}")

    def test_empty_experiment(self) -> None:
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        empty_exp = Experiment(
            signals=empty_df,
            metadata={},
            rpm=0.0,
            axial_depth=0.0,
            type=DatasetType.TIME_SERIES,
            file_name="empty_test.h5",
        )
        res = extract_experiment_features(empty_exp)
        self.assertEqual(len(res), 49)
        for k, v in res.items():
            self.assertEqual(v, 0.0)


if __name__ == "__main__":
    unittest.main()
