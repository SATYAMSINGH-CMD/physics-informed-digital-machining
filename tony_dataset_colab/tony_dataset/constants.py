"""Single Source of Truth Constants for the Tony Dataset Package.

This module defines all project-wide constants, enums, signal channel definitions,
engineering units, category mappings, feature registries, plot styling defaults,
ML hyperparameters, Streamlit UI settings, and regular expression patterns.
"""

from enum import Enum
import re
from typing import Dict, List

# =============================================================================
# SECTION 1: Dataset Types
# =============================================================================


class DatasetType(str, Enum):
    """Supported dataset types in the Tony Schmitz Machining Database."""

    TIME_SERIES = "time_series"
    STABILITY_BOUNDARY = "stability_boundary"
    AUDIO = "audio"
    UNKNOWN = "unknown"


# =============================================================================
# SECTION 2: Signal Names
# =============================================================================

TIME_COLUMNS: List[str] = [
    "RPM",
    "Axial_Depth",
    "Time",
    "Tool_X_Displacement",
    "Tool_Y_Displacement",
    "Workpiece_X_Displacement",
    "Workpiece_Y_Displacement",
    "Tool_X_Velocity",
    "Tool_Y_Velocity",
    "Workpiece_X_Velocity",
    "Workpiece_Y_Velocity",
    "Tool_X_Acceleration",
    "Tool_Y_Acceleration",
    "Workpiece_X_Acceleration",
    "Workpiece_Y_Acceleration",
    "Force_X",
    "Force_Y",
]

# Alias for backward compatibility
TIME_SERIES_COLUMNS: List[str] = TIME_COLUMNS
SIGNAL_COLUMNS: List[str] = TIME_COLUMNS[2:]


STABILITY_COLUMNS: List[str] = [
    "RPM",
    "Limiting_Depth",
]

DEFAULT_SIGNALS: List[str] = [
    "Force_X",
    "Force_Y",
    "Tool_X_Displacement",
    "Tool_Y_Displacement",
]

# =============================================================================
# SECTION 3: Engineering Units (SI Units)
# =============================================================================

SIGNAL_UNITS: Dict[str, str] = {
    "RPM": "rpm",
    "Axial_Depth": "m",
    "Limiting_Depth": "m",
    "Time": "s",
    "Tool_X_Displacement": "m",
    "Tool_Y_Displacement": "m",
    "Workpiece_X_Displacement": "m",
    "Workpiece_Y_Displacement": "m",
    "Tool_X_Velocity": "m/s",
    "Tool_Y_Velocity": "m/s",
    "Workpiece_X_Velocity": "m/s",
    "Workpiece_Y_Velocity": "m/s",
    "Tool_X_Acceleration": "m/s²",
    "Tool_Y_Acceleration": "m/s²",
    "Workpiece_X_Acceleration": "m/s²",
    "Workpiece_Y_Acceleration": "m/s²",
    "Force_X": "N",
    "Force_Y": "N",
}

# =============================================================================
# SECTION 4: Signal Categories
# =============================================================================

DISPLACEMENT_SIGNALS: List[str] = [
    "Tool_X_Displacement",
    "Tool_Y_Displacement",
    "Workpiece_X_Displacement",
    "Workpiece_Y_Displacement",
]

VELOCITY_SIGNALS: List[str] = [
    "Tool_X_Velocity",
    "Tool_Y_Velocity",
    "Workpiece_X_Velocity",
    "Workpiece_Y_Velocity",
]

ACCELERATION_SIGNALS: List[str] = [
    "Tool_X_Acceleration",
    "Tool_Y_Acceleration",
    "Workpiece_X_Acceleration",
    "Workpiece_Y_Acceleration",
]

FORCE_SIGNALS: List[str] = [
    "Force_X",
    "Force_Y",
]

WORKPIECE_SIGNALS: List[str] = [
    "Workpiece_X_Displacement",
    "Workpiece_Y_Displacement",
    "Workpiece_X_Velocity",
    "Workpiece_Y_Velocity",
    "Workpiece_X_Acceleration",
    "Workpiece_Y_Acceleration",
]

TOOL_SIGNALS: List[str] = [
    "Tool_X_Displacement",
    "Tool_Y_Displacement",
    "Tool_X_Velocity",
    "Tool_Y_Velocity",
    "Tool_X_Acceleration",
    "Tool_Y_Acceleration",
]

TIME_SIGNALS: List[str] = [
    "Time",
]

SIGNAL_CATEGORIES: Dict[str, List[str]] = {
    "Displacement": DISPLACEMENT_SIGNALS,
    "Velocity": VELOCITY_SIGNALS,
    "Acceleration": ACCELERATION_SIGNALS,
    "Force": FORCE_SIGNALS,
    "Workpiece": WORKPIECE_SIGNALS,
    "Tool": TOOL_SIGNALS,
    "Time": TIME_SIGNALS,
}

# =============================================================================
# SECTION 5: Default Plot Configuration
# =============================================================================

DEFAULT_X_AXIS: str = "Time"
DEFAULT_Y_AXIS: str = "Force_X"
DEFAULT_THEME: str = "plotly_white"

PLOTLY_DEFAULT_COLORS: List[str] = [
    "#1f77b4",  # Muted Blue
    "#ff7f0e",  # Safety Orange
    "#2ca02c",  # Cooked Asparagus Green
    "#d62728",  # Brick Red
    "#9467bd",  # Muted Purple
    "#8c564b",  # Chestnut Brown
]

DEFAULT_COLOR_SEQUENCE: List[str] = PLOTLY_DEFAULT_COLORS

DEFAULT_FIGURE_HEIGHT: int = 600
DEFAULT_FIGURE_WIDTH: int = 1000

# =============================================================================
# SECTION 6: Supported File Types
# =============================================================================

SUPPORTED_EXTENSIONS: List[str] = [
    ".h5",
    ".mat",
    ".csv",
    ".wav",
]

# =============================================================================
# SECTION 7: Feature Registry
# =============================================================================

TIME_DOMAIN_FEATURES: List[str] = [
    "Mean",
    "Std",
    "Variance",
    "RMS",
    "Peak",
    "Skewness",
    "Kurtosis",
    "Crest_Factor",
    "Margin_Factor",
    "Shape_Factor",
    "Impulse_Factor",
]

FREQUENCY_DOMAIN_FEATURES: List[str] = [
    "FFT_Peak_Freq",
    "FFT_Peak_Mag",
    "Spectral_Centroid",
    "Spectral_Spread",
    "Spectral_Entropy",
    "Spectral_Energy",
    "Dominant_Frequency",
]

PHYSICS_FEATURES: List[str] = [
    "Tooth_Passing_Frequency",
    "Tooth_Passing_Harmonic_Ratio",
    "Stability_Margin",
    "Chatter_Energy_Ratio",
    "Phase_Coherence",
]

DECISION_FEATURES: List[str] = [
    "Chatter_Risk",
    "Stability_Status",
    "Recommended_RPM_Shift",
    "Recommended_Depth_Adjustment",
]

FEATURE_REGISTRY: Dict[str, List[str]] = {
    "time_domain": TIME_DOMAIN_FEATURES,
    "frequency_domain": FREQUENCY_DOMAIN_FEATURES,
    "physics": PHYSICS_FEATURES,
    "decision": DECISION_FEATURES,
}

# =============================================================================
# SECTION 8: Machine Learning
# =============================================================================

DEFAULT_RANDOM_STATE: int = 42
DEFAULT_TEST_SIZE: float = 0.2
DEFAULT_CV_FOLDS: int = 5
DEFAULT_SCORING: str = "f1_weighted"
DEFAULT_MODEL_NAME: str = "random_forest"

# =============================================================================
# SECTION 9: Streamlit Configuration
# =============================================================================

APP_TITLE: str = "Tony Dataset Explorer"
APP_ICON: str = "⚙️"
APP_LAYOUT: str = "wide"
SIDEBAR_WIDTH: int = 300
DEFAULT_UPLOAD_MESSAGE: str = (
    "Drag and drop HDF5 (.h5), MAT (.mat), WAV (.wav), or CSV (.csv) machining datasets"
)

# =============================================================================
# SECTION 10: Miscellaneous
# =============================================================================

SUPPORTED_DATASET_PREFIXES: List[str] = [
    "time",
    "stability_boundary",
    "audio",
]

REGEX_DATASET_GRID_POINT: str = r"[a-zA-Z]*(\d+)[_\-](\d+)"
REGEX_DATASET_NUMBER: str = r"(\d+)"
REGEX_FILENAME_PARSER: str = (
    r"^(?P<prefix>[a-zA-Z_]+)(?P<dataset_num>\d+)(?:[_\-](?P<grid_point>\d+))?\.(?P<ext>h5|mat|csv|wav)$"
)

PATTERN_DATASET_GRID_POINT: re.Pattern[str] = re.compile(REGEX_DATASET_GRID_POINT)
PATTERN_DATASET_NUMBER: re.Pattern[str] = re.compile(REGEX_DATASET_NUMBER)
PATTERN_FILENAME_PARSER: re.Pattern[str] = re.compile(REGEX_FILENAME_PARSER, re.IGNORECASE)
