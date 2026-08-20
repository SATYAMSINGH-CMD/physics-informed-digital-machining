"""Unit tests for tony_dataset.loader module."""

from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import pandas as pd

from tony_dataset.constants import STABILITY_COLUMNS, TIME_COLUMNS, DatasetType
from tony_dataset.experiment import Experiment
from tony_dataset.loader import (
    FileNotFoundDatasetError,
    InvalidExtensionError,
    InvalidMatrixShapeError,
    UnknownDatasetTypeError,
    _parse_filename,
    load_h5,
)


class TestTonyDatasetLoader(unittest.TestCase):
    """Automated unit test suite for tony_dataset/loader.py."""

    def setUp(self) -> None:
        """Create temporary test directory and synthetic HDF5 datasets."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # 1. Create synthetic time-series file (100 samples x 17 channels)
        self.time_file = self.temp_path / "time17_43.h5"
        time_data = np.zeros((100, 17), dtype=np.float64)
        time_data[0, 0] = 1200.0  # RPM
        time_data[0, 1] = 0.0025  # Axial Depth (2.5 mm = 0.0025 m)
        time_data[:, 2] = np.linspace(0.0, 1.0, 100)  # Time
        with h5py.File(self.time_file, "w") as h5:
            h5.create_dataset("time_data", data=time_data)

        # 2. Create synthetic stability boundary file (2 channels x 50 points)
        self.stability_file = self.temp_path / "stability_boundary25.h5"
        stab_data = np.array([[1000, 2000, 3000], [0.001, 0.002, 0.0015]], dtype=np.float64)
        with h5py.File(self.stability_file, "w") as h5:
            h5.create_dataset("stability_curve", data=stab_data)

        # 3. Create invalid shape HDF5 file
        self.invalid_shape_file = self.temp_path / "time99_01_invalid.h5"
        invalid_data = np.zeros((5, 100), dtype=np.float64)
        with h5py.File(self.invalid_shape_file, "w") as h5:
            h5.create_dataset("time_data", data=invalid_data)

        # 4. Create invalid extension file
        self.invalid_ext_file = self.temp_path / "sample.txt"
        self.invalid_ext_file.write_text("dummy text content")

    def tearDown(self) -> None:
        """Clean up temporary test directory."""
        self.temp_dir.cleanup()

    def test_load_time_series_experiment(self) -> None:
        """Verify loading time series returns an Experiment object."""
        exp = load_h5(self.time_file)
        self.assertIsInstance(exp, Experiment)
        self.assertEqual(len(exp.signals), 100)
        self.assertEqual(list(exp.signals.columns), TIME_COLUMNS[2:])
        self.assertNotIn("RPM", exp.signals.columns)
        self.assertNotIn("Axial_Depth", exp.signals.columns)
        self.assertEqual(exp.rpm, 1200.0)
        self.assertEqual(exp.axial_depth, 0.0025)
        self.assertIn("sampling_rate_hz", exp.metadata)
        self.assertGreater(exp.metadata["sampling_rate_hz"], 0)
        self.assertEqual(exp.type, DatasetType.TIME_SERIES)
        self.assertEqual(exp.dataset_number, 17)
        self.assertEqual(exp.grid_point, 43)

    def test_load_stability_boundary_experiment(self) -> None:
        """Verify loading stability boundary returns an Experiment object."""
        exp = load_h5(self.stability_file)
        self.assertIsInstance(exp, Experiment)
        self.assertEqual(len(exp.signals), 3)
        self.assertEqual(list(exp.signals.columns), STABILITY_COLUMNS)
        self.assertEqual(exp.type, DatasetType.STABILITY_BOUNDARY)
        self.assertEqual(exp.dataset_number, 25)
        self.assertIsNone(exp.grid_point)

    def test_parse_filename_helper(self) -> None:
        """Test extraction of dataset number and grid point from file names."""
        ds_num, grid_pt = _parse_filename("time17_43.h5")
        self.assertEqual(ds_num, 17)
        self.assertEqual(grid_pt, 43)

        ds_num2, grid_pt2 = _parse_filename("stability_boundary25.h5")
        self.assertEqual(ds_num2, 25)
        self.assertIsNone(grid_pt2)

    def test_export_csv(self) -> None:
        """Verify CSV export functionality from Experiment object."""
        exp = load_h5(self.time_file)
        csv_out = self.temp_path / "exported_sample.csv"
        returned_path = exp.export_csv(csv_out)

        self.assertTrue(returned_path.is_file())
        df_csv = pd.read_csv(returned_path)
        self.assertEqual(len(df_csv), 100)
        self.assertIn("Time", df_csv.columns)
        self.assertNotIn("RPM", df_csv.columns)

    def test_invalid_shape_raises_exception(self) -> None:
        """Verify that shape mismatch raises InvalidMatrixShapeError."""
        with self.assertRaises(InvalidMatrixShapeError):
            load_h5(self.invalid_shape_file)

    def test_invalid_extension_raises_exception(self) -> None:
        """Verify that invalid file extension raises InvalidExtensionError."""
        with self.assertRaises(InvalidExtensionError):
            load_h5(self.invalid_ext_file)

    def test_file_not_found_raises_exception(self) -> None:
        """Verify that missing file path raises FileNotFoundDatasetError."""
        with self.assertRaises(FileNotFoundDatasetError):
            load_h5(self.temp_path / "non_existent.h5")


if __name__ == "__main__":
    unittest.main()
