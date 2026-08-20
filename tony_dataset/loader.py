"""HDF5 Data Loader for Tony Schmitz Digital Machining Database.

This module provides high-performance loading and validation utilities for HDF5 machining
experiments, returning unified Experiment objects.
"""

import logging
from pathlib import Path
import re
from typing import Any, Dict, Optional, Tuple, Union

try:
    import h5py
except ImportError:
    h5py = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import numpy as np
except ImportError:
    np = None

from tony_dataset.constants import (
    SIGNAL_COLUMNS,
    STABILITY_COLUMNS,
    SUPPORTED_EXTENSIONS,
    TIME_COLUMNS,
    DatasetType,
)
from tony_dataset.experiment import Experiment

logger = logging.getLogger(__name__)


# =============================================================================
# Custom Exception Hierarchy
# =============================================================================


class TonyDatasetError(Exception):
    """Base exception for all Tony Dataset loader errors."""


class FileNotFoundDatasetError(TonyDatasetError, FileNotFoundError):
    """Raised when the specified HDF5 dataset file does not exist."""


class InvalidExtensionError(TonyDatasetError, ValueError):
    """Raised when the input file extension is not supported by the loader."""


class InvalidMatrixShapeError(TonyDatasetError, ValueError):
    """Raised when raw dataset dimensions fail matrix channel validation."""


class UnknownDatasetTypeError(TonyDatasetError, TypeError):
    """Raised when dataset type cannot be mapped to a supported DatasetType."""


# =============================================================================
# Public API
# =============================================================================


def load_h5(filepath: Union[str, Path]) -> Experiment:
    """Loads a Tony Schmitz HDF5 machining dataset into an Experiment object.

    Args:
        filepath: Path to the target HDF5 file (.h5).

    Returns:
        An Experiment object containing signal DataFrame, metadata, and operational parameters.

    Raises:
        FileNotFoundDatasetError: If the specified file path does not exist.
        InvalidExtensionError: If the file extension is not .h5.
        InvalidMatrixShapeError: If the raw dataset shape does not match channel expectations.
        UnknownDatasetTypeError: If the dataset type cannot be recognized.
    """
    path = Path(filepath).resolve()
    _verify_file(path)

    logger.info(f"Opening HDF5 dataset: {path.name}")
    try:
        with h5py.File(path, "r") as h5_file:
            keys = list(h5_file.keys())
            if not keys:
                raise UnknownDatasetTypeError(f"HDF5 file '{path.name}' contains no datasets.")

            root_key = keys[0]
            dataset_obj = h5_file[root_key]
            if not isinstance(dataset_obj, h5py.Dataset):
                raise UnknownDatasetTypeError(
                    f"Root entry '{root_key}' in '{path.name}' is not a valid dataset."
                )

            raw_shape = dataset_obj.shape
            dataset_type = _detect_dataset_type(path.name, root_key, raw_shape)
            ds_number, grid_point = _parse_filename(path.name)

            if dataset_type == DatasetType.TIME_SERIES:
                return _load_time_dataset(
                    h5_file, root_key, path, ds_number, grid_point
                )
            if dataset_type == DatasetType.STABILITY_BOUNDARY:
                return _load_stability_dataset(
                    h5_file, root_key, path, ds_number, grid_point
                )

            raise UnknownDatasetTypeError(
                f"Unsupported dataset type '{dataset_type.value}' for root key '{root_key}' in '{path.name}'."
            )

    except TonyDatasetError:
        raise
    except Exception as exc:
        logger.error(f"Unexpected error loading HDF5 file '{path.name}': {exc}")
        raise UnknownDatasetTypeError(f"Failed to read HDF5 dataset '{path.name}': {exc}") from exc


# =============================================================================
# Private Helper Functions
# =============================================================================


def _verify_file(path: Path) -> None:
    """Verifies that the target file exists and has a valid .h5 extension."""
    if not path.is_file():
        logger.error(f"File not found: {path}")
        raise FileNotFoundDatasetError(f"HDF5 file does not exist: {path}")

    if path.suffix.lower() != ".h5":
        logger.error(f"Invalid extension '{path.suffix}' for file: {path.name}")
        raise InvalidExtensionError(
            f"Expected .h5 extension, got '{path.suffix}' for file '{path.name}'."
        )


def _parse_filename(file_name: str) -> Tuple[Optional[int], Optional[int]]:
    """Extracts dataset number and grid point from standard Tony Schmitz filenames.

    Examples:
        - 'time17_43.h5' -> (17, 43)
        - 'stability_boundary25.h5' -> (25, None)
    """
    stem = Path(file_name).stem
    match_two = re.search(r"[a-zA-Z]*(\d+)[_\-](\d+)", stem)
    if match_two:
        return int(match_two.group(1)), int(match_two.group(2))

    match_one = re.search(r"(\d+)", stem)
    if match_one:
        return int(match_one.group(1)), None

    return None, None


def _detect_dataset_type(
    file_name: str, root_key: str, shape: Tuple[int, ...]
) -> DatasetType:
    """Determines dataset category using filename patterns, root keys, and dimensions."""
    name_lower = file_name.lower()
    key_lower = root_key.lower()

    if "stability" in name_lower or "stability" in key_lower:
        return DatasetType.STABILITY_BOUNDARY

    if "time" in name_lower or "time" in key_lower:
        return DatasetType.TIME_SERIES

    if len(shape) == 2:
        channels = min(shape)
        if channels == len(TIME_COLUMNS):
            return DatasetType.TIME_SERIES
        if channels == len(STABILITY_COLUMNS):
            return DatasetType.STABILITY_BOUNDARY

    return DatasetType.UNKNOWN


def _validate_matrix(
    raw_data: Any, expected_channels: int, dataset_type: DatasetType
) -> Any:
    """Validates 2D matrix dimensions and transposes raw data exactly once."""
    if raw_data.ndim != 2:
        raise InvalidMatrixShapeError(
            f"Expected 2D matrix for {dataset_type.value}, got {raw_data.ndim}D array."
        )

    rows, cols = raw_data.shape
    if cols == expected_channels:
        return raw_data

    if rows == expected_channels:
        return raw_data.T

    raise InvalidMatrixShapeError(
        f"Shape mismatch for {dataset_type.value}. "
        f"Expected {expected_channels} channels, but got shape ({rows}, {cols})."
    )


def _build_metadata(
    root_key: str,
    raw_shape: Tuple[int, ...],
    raw_dtype: str,
    dataset_type: DatasetType,
) -> Dict[str, Any]:
    """Constructs raw HDF5 metadata dictionary for Experiment objects."""
    return {
        "root_dataset_key": root_key,
        "file_type": dataset_type.value,
        "raw_shape": raw_shape,
        "raw_dtype": raw_dtype,
    }


def _load_time_dataset(
    h5_file: Any,
    root_key: str,
    file_path: Path,
    dataset_number: Optional[int],
    grid_point: Optional[int],
) -> Experiment:
    """Loads and validates a time-series machining dataset."""
    dataset_obj = h5_file[root_key]
    raw_matrix = dataset_obj[:]
    raw_shape = raw_matrix.shape
    raw_dtype = str(dataset_obj.dtype)

    matrix = _validate_matrix(
        raw_matrix, len(TIME_COLUMNS), DatasetType.TIME_SERIES
    )

    rpm_val = float(matrix[0, 0])
    axial_depth_val = float(matrix[0, 1])

    signals_data = matrix[:, 2:]
    df = pd.DataFrame(signals_data, columns=SIGNAL_COLUMNS)

    if "Time" in df and len(df) > 1:
        time_vals = df["Time"].to_numpy()
        duration_val = float(time_vals[-1] - time_vals[0])
        diffs = np.diff(time_vals)
        dt_median = float(np.median(diffs)) if len(diffs) > 0 else 0.0
        sampling_rate_val = float(1.0 / dt_median) if dt_median > 0.0 else 0.0
    else:
        duration_val = 0.0
        sampling_rate_val = 0.0

    samples_count = len(df)

    metadata = _build_metadata(
        root_key,
        raw_shape,
        raw_dtype,
        DatasetType.TIME_SERIES,
    )
    metadata["sampling_rate_hz"] = sampling_rate_val

    logger.info(
        f"Loaded Time Series dataset '{file_path.name}' ({samples_count} samples, RPM={rpm_val}, Depth={axial_depth_val}m, fs={sampling_rate_val:.2f}Hz)"
    )

    return Experiment(
        signals=df,
        metadata=metadata,
        rpm=rpm_val,
        axial_depth=axial_depth_val,
        type=DatasetType.TIME_SERIES,
        dataset_number=dataset_number,
        grid_point=grid_point,
        duration=duration_val,
        samples=samples_count,
        file_name=file_path.name,
    )


def _load_stability_dataset(
    h5_file: Any,
    root_key: str,
    file_path: Path,
    dataset_number: Optional[int],
    grid_point: Optional[int],
) -> Experiment:
    """Loads and validates a stability boundary diagram dataset."""
    dataset_obj = h5_file[root_key]
    raw_matrix = dataset_obj[:]
    raw_shape = raw_matrix.shape
    raw_dtype = str(dataset_obj.dtype)

    transposed = _validate_matrix(
        raw_matrix, len(STABILITY_COLUMNS), DatasetType.STABILITY_BOUNDARY
    )
    df = pd.DataFrame(transposed, columns=STABILITY_COLUMNS)
    samples_count = len(df)

    metadata = _build_metadata(
        root_key,
        raw_shape,
        raw_dtype,
        DatasetType.STABILITY_BOUNDARY,
    )

    logger.info(
        f"Loaded Stability Boundary dataset '{file_path.name}' ({samples_count} limit curve points)"
    )

    return Experiment(
        signals=df,
        metadata=metadata,
        rpm=0.0,
        axial_depth=0.0,
        type=DatasetType.STABILITY_BOUNDARY,
        dataset_number=dataset_number,
        grid_point=grid_point,
        duration=0.0,
        samples=samples_count,
        file_name=file_path.name,
    )
