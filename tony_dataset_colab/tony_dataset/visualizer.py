"""Plotly Visualization Module for Tony Dataset Machining Experiments.

This module provides production-ready plotting functions for analyzing machining
signals, spectral FFT distributions, signal correlations, and kinematics using Plotly.
All plotting functions take an Experiment object and return a plotly.graph_objects.Figure.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.fft import rfft, rfftfreq

from tony_dataset.constants import (
    ACCELERATION_SIGNALS,
    DEFAULT_COLOR_SEQUENCE,
    DEFAULT_THEME,
    DISPLACEMENT_SIGNALS,
    FORCE_SIGNALS,
    SIGNAL_UNITS,
    VELOCITY_SIGNALS,
)
from tony_dataset.experiment import Experiment

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================


def _validate_experiment(experiment: Experiment) -> None:
    """Validates that input is a valid Experiment object with signals DataFrame."""
    if not isinstance(experiment, Experiment):
        raise TypeError(
            f"Expected Experiment instance, got {type(experiment).__name__}."
        )

    if not isinstance(experiment.signals, pd.DataFrame) or experiment.signals.empty:
        raise ValueError(f"Experiment '{experiment.file_name}' contains no signal data.")


def _validate_column(df: pd.DataFrame, col_name: str) -> None:
    """Ensures requested column exists in DataFrame."""
    if col_name not in df.columns:
        raise KeyError(
            f"Column '{col_name}' not found in dataset. Available columns: {list(df.columns)}"
        )


def _apply_layout(
    fig: go.Figure, title: str, x_label: str, y_label: str
) -> go.Figure:
    """Applies standardized publication-ready Plotly layout styling."""
    fig.update_layout(
        title={"text": f"<b>{title}</b>", "x": 0.5, "xanchor": "center"},
        xaxis_title=x_label,
        yaxis_title=y_label,
        template=DEFAULT_THEME,
        colorway=DEFAULT_COLOR_SEQUENCE,
        hovermode="x unified",
        margin=dict(l=60, r=40, t=60, b=60),
    )
    return fig


def _plot_signal_group(
    experiment: Experiment,
    signals: list[str],
    title: str,
    y_label: str,
) -> go.Figure:
    """Private helper to plot a group of related signal channels against Time."""
    _validate_experiment(experiment)
    df = experiment.signals
    _validate_column(df, "Time")

    fig = go.Figure()
    for col in signals:
        if col in df.columns:
            unit = SIGNAL_UNITS.get(col, "")
            name_str = f"{col} ({unit})" if unit else col
            fig.add_trace(
                go.Scatter(
                    x=df["Time"],
                    y=df[col],
                    mode="lines",
                    name=name_str,
                    line=dict(width=1.5),
                )
            )

    full_title = f"{title} ({experiment.file_name or 'Experiment'})"
    return _apply_layout(fig, full_title, "Time (s)", y_label)


# =============================================================================
# Public Plotting API
# =============================================================================


def plot_signal(
    experiment: Experiment,
    x_col: str = "Time",
    y_col: str = "Force_X",
) -> go.Figure:
    """Plots any arbitrary signal channel against any x-axis column.

    Args:
        experiment: Input Experiment object.
        x_col: Column name for X-axis. Defaults to "Time".
        y_col: Column name for Y-axis. Defaults to "Force_X".

    Returns:
        Plotly Figure object.
    """
    _validate_experiment(experiment)
    df = experiment.signals
    _validate_column(df, x_col)
    _validate_column(df, y_col)

    x_unit = SIGNAL_UNITS.get(x_col, "")
    y_unit = SIGNAL_UNITS.get(y_col, "")

    x_label = f"{x_col} ({x_unit})" if x_unit else x_col
    y_label = f"{y_col} ({y_unit})" if y_unit else y_col

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode="lines",
            name=y_col,
            line=dict(width=1.5),
        )
    )

    title = f"{y_col} vs {x_col} ({experiment.file_name or 'Experiment'})"
    return _apply_layout(fig, title, x_label, y_label)


def plot_force(experiment: Experiment) -> go.Figure:
    """Plots Force X and Force Y signals against Time.

    Args:
        experiment: Input Experiment object.

    Returns:
        Plotly Figure object displaying cutting force signals.
    """
    return _plot_signal_group(
        experiment, FORCE_SIGNALS, "Cutting Forces vs Time", "Force (N)"
    )


def plot_displacement(experiment: Experiment) -> go.Figure:
    """Plots Tool and Workpiece X/Y displacement signals against Time.

    Args:
        experiment: Input Experiment object.

    Returns:
        Plotly Figure object displaying displacement signals.
    """
    return _plot_signal_group(
        experiment,
        DISPLACEMENT_SIGNALS,
        "Tool & Workpiece Displacements vs Time",
        "Displacement (m)",
    )


def plot_velocity(experiment: Experiment) -> go.Figure:
    """Plots Tool and Workpiece X/Y velocity signals against Time.

    Args:
        experiment: Input Experiment object.

    Returns:
        Plotly Figure object displaying velocity signals.
    """
    return _plot_signal_group(
        experiment, VELOCITY_SIGNALS, "Velocity Signals vs Time", "Velocity (m/s)"
    )


def plot_acceleration(experiment: Experiment) -> go.Figure:
    """Plots Tool and Workpiece X/Y acceleration signals against Time.

    Args:
        experiment: Input Experiment object.

    Returns:
        Plotly Figure object displaying acceleration signals.
    """
    return _plot_signal_group(
        experiment,
        ACCELERATION_SIGNALS,
        "Acceleration Signals vs Time",
        "Acceleration (m/s²)",
    )


def plot_histogram(
    experiment: Experiment,
    signal_col: str = "Force_X",
    nbins: int = 50,
) -> go.Figure:
    """Plots histogram distribution for a selected signal channel.

    Args:
        experiment: Input Experiment object.
        signal_col: Column name to plot. Defaults to "Force_X".
        nbins: Number of histogram bins. Defaults to 50.

    Returns:
        Plotly Figure object displaying signal histogram.
    """
    _validate_experiment(experiment)
    df = experiment.signals
    _validate_column(df, signal_col)

    unit = SIGNAL_UNITS.get(signal_col, "")
    unit_str = f" ({unit})" if unit else ""

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=df[signal_col],
            nbinsx=nbins,
            name=signal_col,
            marker_color=DEFAULT_COLOR_SEQUENCE[0],
            opacity=0.85,
        )
    )

    title = f"Signal Histogram Distribution: {signal_col} ({experiment.file_name or 'Experiment'})"
    return _apply_layout(fig, title, f"{signal_col}{unit_str}", "Sample Count")


def plot_correlation(experiment: Experiment) -> go.Figure:
    """Generates a correlation matrix heatmap across non-constant numerical signal channels.

    Args:
        experiment: Input Experiment object.

    Returns:
        Plotly Figure object displaying correlation heatmap.
    """
    _validate_experiment(experiment)
    df = experiment.signals
    numeric_df = df.select_dtypes(include=[np.number])

    # Drop constant operational parameters from correlation matrix
    numeric_df = numeric_df.drop(columns=["RPM", "Axial_Depth"], errors="ignore")

    if numeric_df.empty:
        raise ValueError(f"No numeric channels found for correlation in '{experiment.file_name}'.")

    corr_matrix = numeric_df.corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=list(corr_matrix.columns),
            y=list(corr_matrix.index),
            colorscale="Viridis",
            zmin=-1.0,
            zmax=1.0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            hoverongaps=False,
        )
    )

    title = f"Signal Correlation Matrix ({experiment.file_name or 'Experiment'})"
    return _apply_layout(fig, title, "Channels", "Channels")


def plot_fft(
    experiment: Experiment,
    signal_col: str = "Force_X",
    sampling_rate: Optional[float] = None,
) -> go.Figure:
    """Computes and plots Fast Fourier Transform (FFT) frequency spectrum using SciPy.

    Args:
        experiment: Input Experiment object.
        signal_col: Column name to analyze. Defaults to "Force_X".
        sampling_rate: Optional explicit sampling rate in Hz. If None, derived from Time column.

    Returns:
        Plotly Figure object displaying frequency magnitude spectrum.
    """
    _validate_experiment(experiment)
    df = experiment.signals
    _validate_column(df, signal_col)

    signal = df[signal_col].to_numpy()
    n = len(signal)
    if n < 2:
        raise ValueError(f"Insufficient samples ({n}) to compute FFT.")

    if sampling_rate is None or sampling_rate <= 0.0:
        if "Time" in df and len(df) > 1:
            time_vals = df["Time"].to_numpy()
            diffs = np.diff(time_vals)
            positive_diffs = diffs[diffs > 0]
            dt = float(np.mean(positive_diffs)) if len(positive_diffs) > 0 else 1e-4
            sampling_rate = 1.0 / dt
        else:
            sampling_rate = 10000.0  # Fallback default

    freqs = rfftfreq(n, d=1.0 / sampling_rate)
    fft_vals = np.abs(rfft(signal)) / n
    if len(fft_vals) > 1:
        fft_vals[1:] *= 2.0  # Single-sided spectrum amplitude correction

    unit = SIGNAL_UNITS.get(signal_col, "")
    unit_str = f" ({unit})" if unit else ""

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=freqs,
            y=fft_vals,
            mode="lines",
            name=f"FFT {signal_col}",
            line=dict(width=1.5, color=DEFAULT_COLOR_SEQUENCE[1]),
        )
    )

    title = f"FFT Frequency Spectrum: {signal_col} ({experiment.file_name or 'Experiment'})"
    return _apply_layout(fig, title, "Frequency (Hz)", f"Magnitude{unit_str}")


def plot_all(experiment: Experiment) -> Dict[str, go.Figure]:
    """Generates all standard plot figures for an Experiment in a single dictionary.

    Args:
        experiment: Input Experiment object.

    Returns:
        Dictionary mapping view names ("Force", "Displacement", etc.) to Plotly Figures.
    """
    _validate_experiment(experiment)
    figures: Dict[str, go.Figure] = {}

    try:
        figures["Force"] = plot_force(experiment)
    except Exception as exc:
        logger.warning(f"Could not generate Force plot: {exc}")

    try:
        figures["Displacement"] = plot_displacement(experiment)
    except Exception as exc:
        logger.warning(f"Could not generate Displacement plot: {exc}")

    try:
        figures["Velocity"] = plot_velocity(experiment)
    except Exception as exc:
        logger.warning(f"Could not generate Velocity plot: {exc}")

    try:
        figures["Acceleration"] = plot_acceleration(experiment)
    except Exception as exc:
        logger.warning(f"Could not generate Acceleration plot: {exc}")

    try:
        figures["Correlation"] = plot_correlation(experiment)
    except Exception as exc:
        logger.warning(f"Could not generate Correlation plot: {exc}")

    try:
        figures["FFT"] = plot_fft(experiment)
    except Exception as exc:
        logger.warning(f"Could not generate FFT plot: {exc}")

    return figures


def get_statistics(experiment: Experiment) -> pd.DataFrame:
    """Computes summary statistics (Mean, Std, Min, Max, Peak, RMS) for signal channels.

    Args:
        experiment: Input Experiment object.

    Returns:
        DataFrame containing statistical summary metrics per signal channel.
    """
    _validate_experiment(experiment)
    df = experiment.signals.select_dtypes(include=[np.number])
    signal_cols = [col for col in df.columns if col != "Time"]

    stats_data = []
    for col in signal_cols:
        series = df[col].dropna()
        if series.empty:
            continue

        vals = series.to_numpy()
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals))
        min_val = float(np.min(vals))
        max_val = float(np.max(vals))
        peak_val = float(np.max(np.abs(vals)))
        rms_val = float(np.sqrt(np.mean(vals**2)))
        unit = SIGNAL_UNITS.get(col, "")

        stats_data.append(
            {
                "Signal": col,
                "Unit": unit,
                "Mean": mean_val,
                "Std": std_val,
                "Min": min_val,
                "Max": max_val,
                "Peak": peak_val,
                "RMS": rms_val,
            }
        )

    stats_df = pd.DataFrame(stats_data)
    if not stats_df.empty:
        stats_df = stats_df.set_index("Signal")
    return stats_df
