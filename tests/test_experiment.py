"""Unit tests for tony_dataset.experiment module."""

import tempfile
from pathlib import Path
import unittest
import pandas as pd
import numpy as np

from tony_dataset.constants import DatasetType
from tony_dataset.experiment import Experiment, export_csv


class TestExperimentDataclass(unittest.TestCase):
    """Automated unit test suite for tony_dataset/experiment.py."""

    def setUp(self) -> None:
        """Create sample DataFrame and metadata."""
        self.df = pd.DataFrame(
            {
                "RPM": [1200.0, 1200.0],
                "Axial_Depth": [2.5, 2.5],
                "Time": [0.0, 0.5],
                "Force_X": [10.0, 12.0],
            }
        )
        self.metadata = {"file_name": "time17_43.h5", "raw": True}
        self.exp = Experiment(
            signals=self.df,
            metadata=self.metadata,
            rpm=1200.0,
            axial_depth=2.5,
            type=DatasetType.TIME_SERIES,
            dataset_number=17,
            grid_point=43,
            duration=0.5,
            samples=2,
            label="Stable",
            file_name="time17_43.h5",
        )

    def test_post_init_validation(self) -> None:
        """Test __post_init__ attributes and validation rules."""
        self.assertEqual(self.exp.samples, 2)
        self.assertEqual(self.exp.feature_count, 0)
        self.assertTrue(self.exp.is_time_series)
        self.assertFalse(self.exp.is_stability_boundary)
        self.assertEqual(self.exp.dataframe.shape, (2, 4))
        self.assertEqual(self.exp.data.shape, (2, 4))
        self.assertEqual(self.exp.depth, 2.5)
        self.assertEqual(self.exp.axial_depth, 2.5)
        self.assertEqual(self.exp.label, "Stable")

        with self.assertRaises(TypeError):
            Experiment(
                signals="invalid_dataframe_type",
                metadata={},
                rpm=1000.0,
                axial_depth=1.0,
                type=DatasetType.TIME_SERIES,
            )

        with self.assertRaises(ValueError):
            Experiment(
                signals=self.df,
                metadata={},
                rpm=-100.0,
                axial_depth=1.0,
                type=DatasetType.TIME_SERIES,
            )

        with self.assertRaises(ValueError):
            Experiment(
                signals=self.df,
                metadata={},
                rpm=1000.0,
                axial_depth=-0.5,
                type=DatasetType.TIME_SERIES,
            )

        with self.assertRaises(ValueError):
            Experiment(
                signals=self.df,
                metadata={},
                rpm=1000.0,
                axial_depth=1.0,
                type=DatasetType.TIME_SERIES,
                confidence=1.5,
            )

    def test_to_dict_and_summary(self) -> None:
        """Test to_dict and summary string outputs."""
        d = self.exp.to_dict()
        self.assertEqual(d["file_name"], "time17_43.h5")
        self.assertEqual(d["dataset_number"], 17)
        self.assertEqual(d["grid_point"], 43)
        self.assertEqual(d["axial_depth"], 2.5)
        self.assertEqual(d["label"], "Stable")

        summary_str = self.exp.summary()
        self.assertIn("time17_43.h5", summary_str)
        self.assertIn("1200.0", summary_str)
        self.assertIn("Stable", summary_str)

        repr_str = repr(self.exp)
        self.assertIn("time17_43.h5", repr_str)
        self.assertIn("axial_depth=2.5mm", repr_str)

    def test_export_csv_method(self) -> None:
        """Test exporting experiment data to CSV."""
        with tempfile.TemporaryDirectory() as temp_dir:
            out_file = Path(temp_dir) / "output.csv"
            res_path = self.exp.export_csv(out_file)
            self.assertTrue(res_path.is_file())
            loaded_df = pd.read_csv(res_path)
            self.assertEqual(len(loaded_df), 2)
            self.assertIn("Force_X", loaded_df.columns)


if __name__ == "__main__":
    unittest.main()
