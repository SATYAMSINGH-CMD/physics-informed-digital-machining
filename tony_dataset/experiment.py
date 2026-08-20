"""Experiment domain model for Tony Schmitz Digital Machining Database."""

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import pandas as pd
except ImportError:
    pd = None

from tony_dataset.constants import DatasetType

logger = logging.getLogger(__name__)


@dataclass
class Experiment:
    """Represents a single machining experiment and its metadata.

    Attributes:
        signals: Signal measurement DataFrame or stability boundary coordinates.
        metadata: Raw dictionary of metadata attributes loaded from HDF5.
        rpm: Spindle rotational speed in Revolutions Per Minute (RPM).
        axial_depth: Axial depth of cut in millimeters (mm).
        type: Dataset category (TIME_SERIES or STABILITY_BOUNDARY).
        dataset_number: Optional integer dataset ID parsed from filename.
        grid_point: Optional integer grid coordinate parsed from filename.
        duration: Total acquisition signal duration in seconds.
        samples: Total number of time-domain samples or boundary points.
        label: Ground truth chatter state label ("Stable", "Unstable", etc.).
        features: Calculated signal features (FFT, time-domain metrics, etc.).
        prediction: ML/DL model chatter state prediction label.
        confidence: Prediction confidence score between 0.0 and 1.0.
        recommendation: Recommended operating parameters from optimization engine.
        file_name: Source HDF5 filename.
    """

    signals: Any
    metadata: Dict[str, Any]
    rpm: float
    axial_depth: float
    type: DatasetType
    dataset_number: Optional[int] = None
    grid_point: Optional[int] = None
    duration: float = 0.0
    samples: int = 0
    label: Optional[str] = None
    features: Dict[str, Any] = field(default_factory=dict)
    prediction: Optional[Any] = None
    confidence: Optional[float] = None
    recommendation: Optional[Any] = None
    file_name: str = ""

    def __post_init__(self) -> None:
        """Validates experiment attributes and initializes derived fields."""
        if pd is not None and not isinstance(self.signals, pd.DataFrame):
            raise TypeError(
                f"Experiment signals must be a pandas DataFrame, got {type(self.signals).__name__}."
            )

        if pd is not None and isinstance(self.signals, pd.DataFrame):
            actual_len = len(self.signals)
            if self.samples == 0:
                self.samples = actual_len
            elif self.samples != actual_len:
                logger.warning(
                    f"Sample count mismatch for '{self.file_name}': "
                    f"explicit samples={self.samples}, DataFrame rows={actual_len}."
                )

        if self.rpm < 0.0:
            raise ValueError(f"Spindle speed (RPM) cannot be negative: {self.rpm}")

        if self.axial_depth < 0.0:
            raise ValueError(f"Axial depth cannot be negative: {self.axial_depth}")

        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Confidence score must be between 0.0 and 1.0, got {self.confidence}"
            )

    @property
    def data(self) -> Any:
        """Alias property for backward compatibility returning signals DataFrame."""
        return self.signals

    @property
    def dataframe(self) -> Any:
        """Alias property for backward compatibility returning signals DataFrame."""
        return self.signals

    @property
    def depth(self) -> float:
        """Alias property for axial depth of cut (mm)."""
        return self.axial_depth

    @property
    def is_time_series(self) -> bool:
        """Returns True if the experiment is a time-series signal dataset."""
        return self.type == DatasetType.TIME_SERIES

    @property
    def is_stability_boundary(self) -> bool:
        """Returns True if the experiment is a stability boundary curve."""
        return self.type == DatasetType.STABILITY_BOUNDARY

    @property
    def feature_count(self) -> int:
        """Returns the number of calculated features currently stored in experiment.features."""
        return len(self.features)

    def export_csv(
        self, output_path: Union[str, Path], index: bool = False
    ) -> Path:
        """Export experiment signals DataFrame to a CSV file.

        Args:
            output_path: Target path for the output CSV file.
            index: Whether to include DataFrame index. Defaults to False.

        Returns:
            Resolved Path object of the exported CSV file.
        """
        target_path = Path(output_path).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        self.signals.to_csv(target_path, index=index)
        logger.info(f"Exported experiment signals to CSV: {target_path}")
        return target_path

    def to_dict(self) -> Dict[str, Any]:
        """Convert experiment metadata and properties to a dictionary.

        Returns:
            Dictionary summary representation of the Experiment instance.
        """
        return {
            "file_name": self.file_name,
            "dataset_number": self.dataset_number,
            "grid_point": self.grid_point,
            "type": self.type.value if hasattr(self.type, "value") else str(self.type),
            "rpm": self.rpm,
            "axial_depth": self.axial_depth,
            "duration": self.duration,
            "samples": self.samples,
            "label": self.label,
            "features_count": self.feature_count,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "recommendation": self.recommendation,
            "raw_metadata": self.metadata,
        }

    def summary(self) -> str:
        """Generates a human-readable multi-line text summary of the experiment.

        Returns:
            Formatted string detailing experiment attributes.
        """
        lines = [
            "==================================================",
            f" EXPERIMENT SUMMARY: {self.file_name or 'Unnamed'}",
            "==================================================",
            f" Type            : {self.type.value if hasattr(self.type, 'value') else self.type}",
            f" Dataset Number  : {self.dataset_number if self.dataset_number is not None else 'N/A'}",
            f" Grid Point      : {self.grid_point if self.grid_point is not None else 'N/A'}",
            f" Spindle RPM     : {self.rpm:.1f}",
            f" Axial Depth     : {self.axial_depth:.2f} mm",
            f" Duration        : {self.duration:.4f} s",
            f" Samples Count   : {self.samples}",
            f" Label           : {self.label if self.label is not None else 'Unlabeled'}",
            f" Features Count  : {self.feature_count}",
            f" Prediction      : {self.prediction if self.prediction is not None else 'Uncomputed'}",
            f" Confidence      : {f'{self.confidence:.2%}' if self.confidence is not None else 'N/A'}",
            "==================================================",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Detailed string representation of the Experiment."""
        return (
            f"Experiment("
            f"file_name='{self.file_name}', "
            f"type={self.type.value if hasattr(self.type, 'value') else self.type}, "
            f"rpm={self.rpm}, "
            f"axial_depth={self.axial_depth}mm, "
            f"samples={self.samples}, "
            f"label={self.label}"
            f")"
        )


def export_csv(
    experiment: Experiment, output_path: Union[str, Path], index: bool = False
) -> Path:
    """Standalone helper function to export an Experiment to CSV."""
    return experiment.export_csv(output_path, index=index)
