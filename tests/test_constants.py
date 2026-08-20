"""Unit tests for tony_dataset.constants module."""

import unittest

from tony_dataset.constants import (
    ACCELERATION_SIGNALS,
    APP_LAYOUT,
    APP_TITLE,
    DECISION_FEATURES,
    DEFAULT_COLOR_SEQUENCE,
    DEFAULT_MODEL_NAME,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SIGNALS,
    DEFAULT_THEME,
    DEFAULT_X_AXIS,
    DEFAULT_Y_AXIS,
    DISPLACEMENT_SIGNALS,
    FEATURE_REGISTRY,
    FORCE_SIGNALS,
    FREQUENCY_DOMAIN_FEATURES,
    PATTERN_FILENAME_PARSER,
    PHYSICS_FEATURES,
    PLOTLY_DEFAULT_COLORS,
    SIGNAL_CATEGORIES,
    SIGNAL_UNITS,
    STABILITY_COLUMNS,
    SUPPORTED_DATASET_PREFIXES,
    SUPPORTED_EXTENSIONS,
    TIME_COLUMNS,
    TIME_DOMAIN_FEATURES,
    TIME_SERIES_COLUMNS,
    TOOL_SIGNALS,
    VELOCITY_SIGNALS,
    WORKPIECE_SIGNALS,
    DatasetType,
)


class TestConstantsModule(unittest.TestCase):
    """Automated unit test suite for tony_dataset/constants.py."""

    def test_dataset_type_enum(self) -> None:
        """Verify DatasetType enum members."""
        self.assertEqual(DatasetType.TIME_SERIES.value, "time_series")
        self.assertEqual(DatasetType.STABILITY_BOUNDARY.value, "stability_boundary")
        self.assertEqual(DatasetType.AUDIO.value, "audio")
        self.assertEqual(DatasetType.UNKNOWN.value, "unknown")

    def test_signal_names_and_defaults(self) -> None:
        """Verify TIME_COLUMNS, STABILITY_COLUMNS, and DEFAULT_SIGNALS lists."""
        self.assertEqual(len(TIME_COLUMNS), 17)
        self.assertIn("RPM", TIME_COLUMNS)
        self.assertIn("Axial_Depth", TIME_COLUMNS)
        self.assertIn("Force_Y", TIME_COLUMNS)
        self.assertEqual(TIME_SERIES_COLUMNS, TIME_COLUMNS)

        self.assertEqual(len(STABILITY_COLUMNS), 2)
        self.assertEqual(STABILITY_COLUMNS, ["RPM", "Limiting_Depth"])

        self.assertEqual(len(DEFAULT_SIGNALS), 4)
        self.assertIn("Force_X", DEFAULT_SIGNALS)
        self.assertIn("Tool_X_Displacement", DEFAULT_SIGNALS)

    def test_signal_units_si(self) -> None:
        """Verify SI engineering units mapping for all signal channels."""
        for col in TIME_COLUMNS:
            self.assertIn(col, SIGNAL_UNITS)

        self.assertEqual(SIGNAL_UNITS["RPM"], "rpm")
        self.assertEqual(SIGNAL_UNITS["Axial_Depth"], "m")
        self.assertEqual(SIGNAL_UNITS["Limiting_Depth"], "m")
        self.assertEqual(SIGNAL_UNITS["Tool_X_Displacement"], "m")
        self.assertEqual(SIGNAL_UNITS["Force_X"], "N")

    def test_signal_categories(self) -> None:
        """Verify signal channel groupings."""
        self.assertEqual(len(DISPLACEMENT_SIGNALS), 4)
        self.assertEqual(len(VELOCITY_SIGNALS), 4)
        self.assertEqual(len(ACCELERATION_SIGNALS), 4)
        self.assertEqual(len(FORCE_SIGNALS), 2)
        self.assertIn("Displacement", SIGNAL_CATEGORIES)
        self.assertIn("Force", SIGNAL_CATEGORIES)

    def test_plot_defaults(self) -> None:
        """Verify Plotly and plot rendering defaults."""
        self.assertEqual(DEFAULT_X_AXIS, "Time")
        self.assertEqual(DEFAULT_Y_AXIS, "Force_X")
        self.assertEqual(DEFAULT_THEME, "plotly_white")
        self.assertEqual(DEFAULT_COLOR_SEQUENCE, PLOTLY_DEFAULT_COLORS)

    def test_supported_extensions(self) -> None:
        """Verify supported file extensions."""
        self.assertIn(".h5", SUPPORTED_EXTENSIONS)
        self.assertIn(".mat", SUPPORTED_EXTENSIONS)
        self.assertIn(".csv", SUPPORTED_EXTENSIONS)
        self.assertIn(".wav", SUPPORTED_EXTENSIONS)

    def test_feature_registries(self) -> None:
        """Verify feature name registries and FEATURE_REGISTRY structure."""
        self.assertEqual(TIME_DOMAIN_FEATURES[0], "Mean")
        self.assertEqual(TIME_DOMAIN_FEATURES[1], "Std")
        self.assertEqual(TIME_DOMAIN_FEATURES[3], "RMS")
        self.assertIn("Spectral_Entropy", FREQUENCY_DOMAIN_FEATURES)
        self.assertIn("Tooth_Passing_Frequency", PHYSICS_FEATURES)
        self.assertIn("Chatter_Risk", DECISION_FEATURES)

        self.assertIn("time_domain", FEATURE_REGISTRY)
        self.assertIn("frequency_domain", FEATURE_REGISTRY)
        self.assertIn("physics", FEATURE_REGISTRY)
        self.assertIn("decision", FEATURE_REGISTRY)

    def test_ml_defaults(self) -> None:
        """Verify ML hyperparameters and defaults."""
        self.assertEqual(DEFAULT_RANDOM_STATE, 42)
        self.assertEqual(DEFAULT_MODEL_NAME, "random_forest")

    def test_streamlit_config(self) -> None:
        """Verify Streamlit UI configurations."""
        self.assertEqual(APP_LAYOUT, "wide")
        self.assertEqual(APP_TITLE, "Tony Dataset Explorer")

    def test_regex_patterns(self) -> None:
        """Verify regex patterns for filename parsing."""
        self.assertIn("time", SUPPORTED_DATASET_PREFIXES)

        match = PATTERN_FILENAME_PARSER.match("time17_43.h5")
        self.assertIsNotNone(match)
        if match:
            self.assertEqual(match.group("prefix"), "time")
            self.assertEqual(match.group("dataset_num"), "17")
            self.assertEqual(match.group("grid_point"), "43")


if __name__ == "__main__":
    unittest.main()
