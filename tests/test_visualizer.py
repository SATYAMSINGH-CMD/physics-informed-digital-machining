"""Unit tests for tony_dataset.visualizer module."""

import unittest
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from tony_dataset.constants import DatasetType, TIME_COLUMNS
from tony_dataset.experiment import Experiment
from tony_dataset.visualizer import (
    get_statistics,
    plot_acceleration,
    plot_all,
    plot_correlation,
    plot_displacement,
    plot_fft,
    plot_force,
    plot_histogram,
    plot_signal,
    plot_velocity,
)


class TestVisualizerModule(unittest.TestCase):
    """Automated unit test suite for tony_dataset/visualizer.py."""

    def setUp(self) -> None:
        """Create synthetic Experiment object for plotting tests."""
        num_samples = 100
        time = np.linspace(0.0, 0.1, num_samples)
        data_dict = {col: np.random.randn(num_samples) for col in TIME_COLUMNS}
        data_dict["Time"] = time
        data_dict["RPM"] = np.full(num_samples, 1200.0)
        data_dict["Axial_Depth"] = np.full(num_samples, 0.0025)

        self.df = pd.DataFrame(data_dict)
        self.exp = Experiment(
            signals=self.df,
            metadata={"file_name": "test_time.h5"},
            rpm=1200.0,
            axial_depth=0.0025,
            type=DatasetType.TIME_SERIES,
            file_name="test_time.h5",
        )

    def test_plot_signal(self) -> None:
        """Verify plot_signal generates a valid Plotly Figure."""
        fig = plot_signal(self.exp, x_col="Time", y_col="Force_X")
        self.assertIsInstance(fig, go.Figure)
        self.assertGreater(len(fig.data), 0)

    def test_plot_force(self) -> None:
        """Verify plot_force generates a valid Plotly Figure."""
        fig = plot_force(self.exp)
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 2)  # Force_X and Force_Y

    def test_plot_displacement(self) -> None:
        """Verify plot_displacement generates a valid Plotly Figure."""
        fig = plot_displacement(self.exp)
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 4)  # Tool X/Y & Workpiece X/Y

    def test_plot_velocity(self) -> None:
        """Verify plot_velocity generates a valid Plotly Figure."""
        fig = plot_velocity(self.exp)
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 4)

    def test_plot_acceleration(self) -> None:
        """Verify plot_acceleration generates a valid Plotly Figure."""
        fig = plot_acceleration(self.exp)
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 4)

    def test_plot_histogram(self) -> None:
        """Verify plot_histogram generates a valid Plotly Figure."""
        fig = plot_histogram(self.exp, signal_col="Force_X", nbins=20)
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 1)

    def test_plot_correlation(self) -> None:
        """Verify plot_correlation excludes constant channels and generates heatmap."""
        fig = plot_correlation(self.exp)
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 1)
        heatmap = fig.data[0]
        self.assertNotIn("RPM", heatmap.x)
        self.assertNotIn("Axial_Depth", heatmap.x)

    def test_plot_fft(self) -> None:
        """Verify plot_fft computes FFT spectrum and returns a Plotly Figure."""
        fig = plot_fft(self.exp, signal_col="Force_X")
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 1)

    def test_plot_all(self) -> None:
        """Verify plot_all returns dictionary of all figure types."""
        figs = plot_all(self.exp)
        self.assertIn("Force", figs)
        self.assertIn("Displacement", figs)
        self.assertIn("Velocity", figs)
        self.assertIn("Acceleration", figs)
        self.assertIn("Correlation", figs)
        self.assertIn("FFT", figs)
        self.assertIsInstance(figs["Force"], go.Figure)

    def test_get_statistics(self) -> None:
        """Verify get_statistics computes summary statistics DataFrame."""
        stats_df = get_statistics(self.exp)
        self.assertIsInstance(stats_df, pd.DataFrame)
        self.assertIn("Force_X", stats_df.index)
        self.assertIn("Mean", stats_df.columns)
        self.assertIn("RMS", stats_df.columns)
        self.assertIn("Peak", stats_df.columns)

    def test_invalid_experiment_raises_type_error(self) -> None:
        """Verify passing non-Experiment raises TypeError."""
        with self.assertRaises(TypeError):
            plot_force("invalid_experiment_type")  # type: ignore


if __name__ == "__main__":
    unittest.main()
