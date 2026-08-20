"""Physics-Informed Feature Engineering Module for Tony Schmitz Digital Machining Database.

This module provides signal processing, spectral analysis, cross-channel correlation,
discrete wavelet transform, nonlinear dynamics, autoregressive modeling, and physics-based
feature extraction for machining experiment signals.

It extracts a unified 49-candidate feature vector per Experiment instance.
"""

from dataclasses import field
import math
import logging

import numpy as np
import pandas as pd
import pywt
import scipy.linalg
import scipy.signal
import scipy.spatial
import scipy.stats

from tony_dataset.experiment import Experiment

logger = logging.getLogger(__name__)

# =============================================================================
# Standard 49 Feature Name Registry
# =============================================================================

FEATURE_NAMES_49 = [
    # Group 1 — Time Domain (10)
    "kurtosis",
    "skewness",
    "crest_factor",
    "margin_factor",
    "shape_factor",
    "impulse_factor",
    "zero_crossing_rate",
    "mean_crossing_rate",
    "kurtosis_1st_derivative",
    "skewness_1st_derivative",
    # Group 2 — Frequency Domain (8)
    "off_harmonic_energy_ratio",
    "spectral_centroid",
    "spectral_entropy",
    "spectral_flatness",
    "spectral_rolloff_85",
    "spectral_spread",
    "spectral_skewness",
    "spectral_kurtosis",
    # Group 3 — Cross-Channel (7)
    "cross_correlation_coeff",
    "cross_axis_peak_delay",
    "bivariate_orbit_radius_ratio",
    "coherence_at_tpf",
    "coherence_at_dominant_resonant",
    "cross_spectral_centroid",
    "multi_axis_energy_asymmetry",
    # Group 4 — Wavelet (10)
    "d1_energy",
    "d2_energy",
    "d3_energy",
    "d4_energy",
    "a4_energy",
    "wavelet_energy_entropy",
    "d1_wavelet_kurtosis",
    "d3_wavelet_std",
    "d1_relative_energy",
    "d3_d4_subband_energy_ratio",
    # Group 5 — Nonlinear (5)
    "permutation_entropy",
    "higuchi_fractal_dimension",
    "katz_fractal_dimension",
    "phase_space_ellipsicity",
    "phase_space_area",
    # Group 6 — Autoregressive (5)
    "ar_coeff_1",
    "ar_coeff_2",
    "ar_coeff_3",
    "ar_residual_variance",
    "autocorr_first_zero_lag",
    # Group 7 — Physics (4)
    "harmonic_peak_ratio",
    "dominant_peak_tpf_ratio",
    "transimpedance",
    "acceleration_jerk_rms",
]

EPS = 1e-12


# =============================================================================
# Helper Utilities
# =============================================================================


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    """Safe division preventing division by zero or NaN results."""
    if abs(b) < EPS or np.isnan(b) or np.isnan(a):
        return default
    val = a / b
    return float(val) if np.isfinite(val) else default


def _rms(x: np.ndarray) -> float:
    """Root Mean Square calculation."""
    if len(x) == 0:
        return 0.0
    return float(np.sqrt(np.mean(x**2)))


def _get_signal(df: pd.DataFrame, col: str) -> np.ndarray:
    """Extract signal array from DataFrame or return zero array if missing."""
    if col in df.columns:
        arr = df[col].to_numpy(dtype=np.float64)
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr
    return np.zeros(len(df), dtype=np.float64)


# =============================================================================
# Domain Feature Extractors
# =============================================================================


def _extract_time_domain(
    force_x: np.ndarray, accel_x: np.ndarray, fs: float
) -> dict[str, float]:
    """Group 1: 10 Time-Domain Statistics & Shape Metrics."""
    features: dict[str, float] = {}
    N = len(force_x)

    if N < 2:
        for k in [
            "kurtosis",
            "skewness",
            "crest_factor",
            "margin_factor",
            "shape_factor",
            "impulse_factor",
            "zero_crossing_rate",
            "mean_crossing_rate",
            "kurtosis_1st_derivative",
            "skewness_1st_derivative",
        ]:
            features[k] = 0.0
        return features

    # 1. Kurtosis
    features["kurtosis"] = float(scipy.stats.kurtosis(force_x, fisher=False))

    # 2. Skewness
    features["skewness"] = float(scipy.stats.skew(force_x))

    # 3. Crest Factor
    peak_fx = float(np.max(np.abs(force_x)))
    rms_fx = _rms(force_x)
    features["crest_factor"] = _safe_div(peak_fx, rms_fx)

    # 4. Margin Factor
    mean_sqrt_fx = float(np.mean(np.sqrt(np.abs(force_x))))
    features["margin_factor"] = _safe_div(peak_fx, mean_sqrt_fx**2)

    # 5. Shape Factor
    mean_abs_fx = float(np.mean(np.abs(force_x)))
    features["shape_factor"] = _safe_div(rms_fx, mean_abs_fx)

    # 6. Impulse Factor
    features["impulse_factor"] = _safe_div(peak_fx, mean_abs_fx)

    # 7. Zero Crossing Rate (Tool_X_Acceleration, mean-centered)
    ax_centered = accel_x - np.mean(accel_x)
    zcr = (
        np.sum(np.diff(np.signbit(ax_centered)) != 0) / float(N - 1)
        if N > 1
        else 0.0
    )
    features["zero_crossing_rate"] = float(zcr)

    # 8. Mean Crossing Rate (Tool_X_Acceleration)
    mcr = (
        np.sum(np.diff(np.signbit(ax_centered)) != 0) / float(N - 1)
        if N > 1
        else 0.0
    )
    features["mean_crossing_rate"] = float(mcr)

    # 9. Kurtosis of 1st Derivative (Tool_X_Acceleration)
    dt = 1.0 / fs if fs > 0 else 1.0
    d_accel = np.diff(accel_x) / dt
    features["kurtosis_1st_derivative"] = (
        float(scipy.stats.kurtosis(d_accel, fisher=False))
        if len(d_accel) > 2 and np.std(d_accel) > EPS
        else 0.0
    )

    # 10. Skewness of 1st Derivative (Force_X)
    d_force = np.diff(force_x) / dt
    features["skewness_1st_derivative"] = (
        float(scipy.stats.skew(d_force))
        if len(d_force) > 2 and np.std(d_force) > EPS
        else 0.0
    )

    return features


def _extract_frequency_domain(
    force_x: np.ndarray,
    fs: float,
    rpm: float,
    num_teeth: int,
    harmonic_bw_hz: float = 10.0,
) -> dict[str, float]:
    """Group 2: 8 Frequency-Domain Spectral Distribution Metrics."""
    features: dict[str, float] = {}
    N = len(force_x)

    if N < 4 or fs <= 0.0:
        for k in [
            "off_harmonic_energy_ratio",
            "spectral_centroid",
            "spectral_entropy",
            "spectral_flatness",
            "spectral_rolloff_85",
            "spectral_spread",
            "spectral_skewness",
            "spectral_kurtosis",
        ]:
            features[k] = 0.0
        return features

    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    fft_vals = np.abs(np.fft.rfft(force_x)) / N
    if len(fft_vals) > 1:
        fft_vals[1:] *= 2.0  # single-sided magnitude
    power = fft_vals**2

    total_power = float(np.sum(power))

    # 11. Off-Harmonic Energy Ratio
    # Engineering definition: 1 - (Harmonic Energy / Total Energy)
    # Bandwidth: +/- harmonic_bw_hz (default 10 Hz) around tooth passing frequency harmonics
    f_tpf = (rpm * num_teeth) / 60.0
    if f_tpf > 0.0 and total_power > EPS:
        max_freq = freqs[-1]
        harmonic_mask = np.zeros(len(freqs), dtype=bool)
        k = 1
        while k * f_tpf <= max_freq:
            f_center = k * f_tpf
            harmonic_mask |= np.abs(freqs - f_center) <= harmonic_bw_hz
            k += 1
        harmonic_power = float(np.sum(power[harmonic_mask]))
        features["off_harmonic_energy_ratio"] = float(
            1.0 - _safe_div(harmonic_power, total_power)
        )
    else:
        features["off_harmonic_energy_ratio"] = 0.0

    # 12. Spectral Centroid
    centroid = (
        _safe_div(float(np.sum(freqs * fft_vals)), float(np.sum(fft_vals)))
        if np.sum(fft_vals) > EPS
        else 0.0
    )
    features["spectral_centroid"] = centroid

    # 13. Spectral Entropy
    sum_pow = np.sum(power)
    if sum_pow > EPS:
        p = power / sum_pow
        p = p[p > 0]
        spectral_ent = -np.sum(p * np.log2(p)) / np.log2(len(power))
        features["spectral_entropy"] = float(spectral_ent)
    else:
        features["spectral_entropy"] = 0.0

    # 14. Spectral Flatness (Geometric Mean / Arithmetic Mean)
    if total_power > EPS:
        log_pow = np.log(power + EPS)
        geom_mean = np.exp(np.mean(log_pow))
        arith_mean = np.mean(power)
        features["spectral_flatness"] = _safe_div(float(geom_mean), float(arith_mean))
    else:
        features["spectral_flatness"] = 0.0

    # 15. Spectral Roll-off (85%)
    cum_mag = np.cumsum(fft_vals)
    total_mag = cum_mag[-1]
    if total_mag > EPS:
        idx_85 = np.searchsorted(cum_mag, 0.85 * total_mag)
        idx_85 = min(idx_85, len(freqs) - 1)
        features["spectral_rolloff_85"] = float(freqs[idx_85])
    else:
        features["spectral_rolloff_85"] = 0.0

    # 16. Spectral Spread (Standard deviation around Spectral Centroid)
    sum_mag = float(np.sum(fft_vals))
    if sum_mag > EPS:
        spread = np.sqrt(
            _safe_div(float(np.sum(((freqs - centroid) ** 2) * fft_vals)), sum_mag)
        )
        features["spectral_spread"] = float(spread)
    else:
        spread = 0.0
        features["spectral_spread"] = 0.0

    # 17. Spectral Skewness
    if sum_mag > EPS and spread > EPS:
        m3 = _safe_div(
            float(np.sum(((freqs - centroid) ** 3) * fft_vals)), sum_mag
        )
        features["spectral_skewness"] = _safe_div(m3, spread**3)
    else:
        features["spectral_skewness"] = 0.0

    # 18. Spectral Kurtosis
    if sum_mag > EPS and spread > EPS:
        m4 = _safe_div(
            float(np.sum(((freqs - centroid) ** 4) * fft_vals)), sum_mag
        )
        features["spectral_kurtosis"] = _safe_div(m4, spread**4)
    else:
        features["spectral_kurtosis"] = 0.0

    return features


def _extract_cross_channel(
    force_x: np.ndarray,
    force_y: np.ndarray,
    accel_x: np.ndarray,
    displ_x: np.ndarray,
    displ_y: np.ndarray,
    fs: float,
    rpm: float,
    num_teeth: int,
) -> dict[str, float]:
    """Group 3: 7 Cross-Channel & Multi-Axis Metrics."""
    features: dict[str, float] = {}
    N = len(force_x)

    if N < 4 or fs <= 0.0:
        for k in [
            "cross_correlation_coeff",
            "cross_axis_peak_delay",
            "bivariate_orbit_radius_ratio",
            "coherence_at_tpf",
            "coherence_at_dominant_resonant",
            "cross_spectral_centroid",
            "multi_axis_energy_asymmetry",
        ]:
            features[k] = 0.0
        return features

    # 19. Cross-Correlation Coefficient (Force_X, Force_Y)
    fx_zero = force_x - np.mean(force_x)
    fy_zero = force_y - np.mean(force_y)
    std_x = np.std(fx_zero)
    std_y = np.std(fy_zero)

    if std_x > EPS and std_y > EPS:
        corr = scipy.signal.correlate(fx_zero, fy_zero, mode="full") / (
            std_x * std_y * N
        )
        lags = scipy.signal.correlation_lags(N, N, mode="full")
        peak_idx = int(np.argmax(np.abs(corr)))
        max_corr = float(corr[peak_idx])
        peak_lag = int(lags[peak_idx])
        features["cross_correlation_coeff"] = max_corr
        # 20. Cross-Axis Peak Delay (in seconds)
        features["cross_axis_peak_delay"] = float(abs(peak_lag) / fs)
    else:
        features["cross_correlation_coeff"] = 0.0
        features["cross_axis_peak_delay"] = 0.0

    # 21. Bivariate Orbit Radius Ratio (Tool_X_Displacement, Tool_Y_Displacement)
    dx_zero = displ_x - np.mean(displ_x)
    dy_zero = displ_y - np.mean(displ_y)
    if np.std(dx_zero) > EPS or np.std(dy_zero) > EPS:
        orbit_matrix = np.column_stack([dx_zero, dy_zero])
        u, s, vt = np.linalg.svd(orbit_matrix, full_matrices=False)
        features["bivariate_orbit_radius_ratio"] = _safe_div(
            float(s[1]), float(s[0])
        )
    else:
        features["bivariate_orbit_radius_ratio"] = 0.0

    # 22. Coherence at TPF (Force_X, Force_Y)
    f_tpf = (rpm * num_teeth) / 60.0
    nperseg = min(N, 1024)
    if nperseg >= 16:
        f_coh, cxy = scipy.signal.coherence(force_x, force_y, fs=fs, nperseg=nperseg)
        if f_tpf > 0.0:
            idx_tpf = int(np.argmin(np.abs(f_coh - f_tpf)))
            features["coherence_at_tpf"] = float(cxy[idx_tpf])
        else:
            features["coherence_at_tpf"] = 0.0
    else:
        features["coherence_at_tpf"] = 0.0

    # 23. Coherence at Dominant Resonant Frequency (Tool_X_Acceleration, Force_X)
    if nperseg >= 16:
        f_acc, p_acc = scipy.signal.welch(accel_x, fs=fs, nperseg=nperseg)
        f_dom = float(f_acc[np.argmax(p_acc)])
        f_coh2, cxy2 = scipy.signal.coherence(accel_x, force_x, fs=fs, nperseg=nperseg)
        idx_dom = int(np.argmin(np.abs(f_coh2 - f_dom)))
        features["coherence_at_dominant_resonant"] = float(cxy2[idx_dom])
    else:
        features["coherence_at_dominant_resonant"] = 0.0

    # 24. Cross-Spectral Centroid
    # Formula: sum(f * abs(CPSD)) / sum(abs(CPSD))
    if nperseg >= 16:
        f_csd, pxy = scipy.signal.csd(force_x, force_y, fs=fs, nperseg=nperseg)
        abs_pxy = np.abs(pxy)
        sum_abs_pxy = float(np.sum(abs_pxy))
        features["cross_spectral_centroid"] = (
            _safe_div(float(np.sum(f_csd * abs_pxy)), sum_abs_pxy)
            if sum_abs_pxy > EPS
            else 0.0
        )
    else:
        features["cross_spectral_centroid"] = 0.0

    # 25. Multi-Axis Energy Asymmetry: (Ex - Ey) / (Ex + Ey)
    ex = float(np.sum(force_x**2))
    ey = float(np.sum(force_y**2))
    features["multi_axis_energy_asymmetry"] = _safe_div(ex - ey, ex + ey)

    return features


def _extract_wavelet(force_x: np.ndarray, fs: float) -> dict[str, float]:
    """Group 4: 10 Discrete Wavelet Transform (db4, level=4) Sub-band Metrics."""
    features: dict[str, float] = {}
    N = len(force_x)

    if N < 32 or fs <= 0.0:
        for k in [
            "d1_energy",
            "d2_energy",
            "d3_energy",
            "d4_energy",
            "a4_energy",
            "wavelet_energy_entropy",
            "d1_wavelet_kurtosis",
            "d3_wavelet_std",
            "d1_relative_energy",
            "d3_d4_subband_energy_ratio",
        ]:
            features[k] = 0.0
        return features

    try:
        coeffs = pywt.wavedec(force_x, "db4", level=4)
        # coeffs layout: [A4, D4, D3, D2, D1]
        a4, d4, d3, d2, d1 = coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4]

        e_a4 = float(np.sum(a4**2))
        e_d4 = float(np.sum(d4**2))
        e_d3 = float(np.sum(d3**2))
        e_d2 = float(np.sum(d2**2))
        e_d1 = float(np.sum(d1**2))
        e_total = e_a4 + e_d4 + e_d3 + e_d2 + e_d1

        # 26-30. Sub-band Energies
        features["d1_energy"] = e_d1
        features["d2_energy"] = e_d2
        features["d3_energy"] = e_d3
        features["d4_energy"] = e_d4
        features["a4_energy"] = e_a4

        # 31. Wavelet Energy Entropy
        if e_total > EPS:
            energies = np.array([e_a4, e_d4, e_d3, e_d2, e_d1])
            p = energies / e_total
            p = p[p > 0]
            features["wavelet_energy_entropy"] = float(-np.sum(p * np.log2(p)))
        else:
            features["wavelet_energy_entropy"] = 0.0

        # 32. D1 Wavelet Kurtosis
        features["d1_wavelet_kurtosis"] = (
            float(scipy.stats.kurtosis(d1, fisher=False))
            if len(d1) > 2 and np.std(d1) > EPS
            else 0.0
        )

        # 33. D3 Wavelet Standard Deviation
        features["d3_wavelet_std"] = float(np.std(d3))

        # 34. D1 Relative Energy
        features["d1_relative_energy"] = _safe_div(e_d1, e_total)

        # 35. D3/D4 Sub-Band Energy Ratio
        features["d3_d4_subband_energy_ratio"] = _safe_div(e_d3, e_d4)

    except Exception as exc:
        logger.warning(f"Wavelet decomposition error: {exc}")
        for k in [
            "d1_energy",
            "d2_energy",
            "d3_energy",
            "d4_energy",
            "a4_energy",
            "wavelet_energy_entropy",
            "d1_wavelet_kurtosis",
            "d3_wavelet_std",
            "d1_relative_energy",
            "d3_d4_subband_energy_ratio",
        ]:
            features[k] = 0.0

    return features


def _extract_nonlinear(
    force_x: np.ndarray,
    accel_x: np.ndarray,
    displ_x: np.ndarray,
    vel_x: np.ndarray,
    fs: float,
) -> dict[str, float]:
    """Group 5: 5 Nonlinear Dynamics & Phase Space Metrics."""
    features: dict[str, float] = {}
    N = len(force_x)

    if N < 16:
        for k in [
            "permutation_entropy",
            "higuchi_fractal_dimension",
            "katz_fractal_dimension",
            "phase_space_ellipsicity",
            "phase_space_area",
        ]:
            features[k] = 0.0
        return features

    # 36. Permutation Entropy (m=4, delay=1)
    m = 4
    if N >= m:
        sub_vectors = np.column_stack([force_x[i : N - m + 1 + i] for i in range(m)])
        ranks = np.argsort(sub_vectors, axis=1)
        # Convert rank rows to tuple pattern counts
        patterns, counts = np.unique(ranks, axis=0, return_counts=True)
        probs = counts / np.sum(counts)
        pe = -np.sum(probs * np.log2(probs)) / np.log2(float(math.factorial(m)))
        features["permutation_entropy"] = float(pe)
    else:
        features["permutation_entropy"] = 0.0

    # 37. Higuchi Fractal Dimension (k_max=10)
    k_max = min(10, N // 4)
    if k_max >= 2:
        lk = []
        k_values = np.arange(1, k_max + 1)
        for k in k_values:
            lm = []
            for m_idx in range(k):
                n_max = (N - 1 - m_idx) // k
                if n_max > 0:
                    sub_seq = force_x[m_idx :: k]
                    diff_sum = np.sum(np.abs(np.diff(sub_seq[: n_max + 1])))
                    norm_fact = (N - 1) / float(n_max * k)
                    lm.append(diff_sum * norm_fact / float(k))
            if lm:
                lk.append(np.mean(lm))

        if len(lk) == len(k_values) and all(val > 0 for val in lk):
            poly = np.polyfit(np.log(1.0 / k_values), np.log(lk), 1)
            features["higuchi_fractal_dimension"] = float(poly[0])
        else:
            features["higuchi_fractal_dimension"] = 1.0
    else:
        features["higuchi_fractal_dimension"] = 1.0

    # 38. Katz Fractal Dimension: D = log10(N-1) / (log10(N-1) + log10(d / L))
    if N >= 3:
        dx_step = 1.0
        dy = np.diff(force_x)
        step_lengths = np.sqrt(dx_step**2 + dy**2)
        L = float(np.sum(step_lengths))
        x0 = force_x[0]
        distances = np.sqrt((np.arange(N) * dx_step) ** 2 + (force_x - x0) ** 2)
        d = float(np.max(distances))

        if d > EPS and L > d:
            denom = np.log10(N - 1) + np.log10(d / L)
            if abs(denom) > EPS:
                kfd = float(np.log10(N - 1) / denom)
                features["katz_fractal_dimension"] = float(max(1.0, kfd))
            else:
                features["katz_fractal_dimension"] = 1.0
        else:
            features["katz_fractal_dimension"] = 1.0
    else:
        features["katz_fractal_dimension"] = 1.0

    # 39. Phase Space Ellipsicity (Tool_X_Acceleration, 2D SVD singular value ratio)
    tau = max(1, min(N // 4, 5))
    if N > tau + 4:
        ps_accel = np.column_stack([accel_x[:-tau], accel_x[tau:]])
        ps_accel = ps_accel - np.mean(ps_accel, axis=0)
        u, s, vt = np.linalg.svd(ps_accel, full_matrices=False)
        features["phase_space_ellipsicity"] = _safe_div(float(s[1]), float(s[0]))
    else:
        features["phase_space_ellipsicity"] = 0.0

    # 40. Phase Space Area (Convex Hull of Tool_X_Displacement vs Tool_X_Velocity)
    if N >= 6 and np.std(displ_x) > EPS and np.std(vel_x) > EPS:
        try:
            pts = np.column_stack([displ_x, vel_x])
            hull = scipy.spatial.ConvexHull(pts)
            features["phase_space_area"] = float(hull.volume)
        except Exception:
            features["phase_space_area"] = 0.0
    else:
        features["phase_space_area"] = 0.0

    return features


def _extract_autoregressive(force_x: np.ndarray, fs: float) -> dict[str, float]:
    """Group 6: 5 Autoregressive Model & Correlation Metrics."""
    features: dict[str, float] = {}
    N = len(force_x)

    if N < 8:
        for k in [
            "ar_coeff_1",
            "ar_coeff_2",
            "ar_coeff_3",
            "ar_residual_variance",
            "autocorr_first_zero_lag",
        ]:
            features[k] = 0.0
        return features

    # Autocorrelation of Force_X
    fx_zero = force_x - np.mean(force_x)
    var_fx = np.var(fx_zero)

    if var_fx > EPS:
        autocorr = scipy.signal.correlate(fx_zero, fx_zero, mode="full")
        mid = len(autocorr) // 2
        r = autocorr[mid : mid + 4] / (var_fx * N)

        # 41-43. AR(3) Coefficients via Yule-Walker equations: R * phi = r[1:4]
        try:
            r_mat = scipy.linalg.toeplitz(r[:3])
            phi = scipy.linalg.solve(r_mat, r[1:4])
            features["ar_coeff_1"] = float(phi[0])
            features["ar_coeff_2"] = float(phi[1])
            features["ar_coeff_3"] = float(phi[2])

            # 44. AR Residual Variance
            pred = (
                phi[0] * fx_zero[2:-1]
                + phi[1] * fx_zero[1:-2]
                + phi[2] * fx_zero[:-3]
            )
            res = fx_zero[3:] - pred
            features["ar_residual_variance"] = float(np.var(res))
        except Exception:
            features["ar_coeff_1"] = 0.0
            features["ar_coeff_2"] = 0.0
            features["ar_coeff_3"] = 0.0
            features["ar_residual_variance"] = 0.0

        # 45. Autocorrelation First Zero-Crossing Lag (in seconds)
        pos_r = autocorr[mid:]
        zero_crossings = np.where(pos_r <= 0)[0]
        if len(zero_crossings) > 0:
            first_zero_idx = int(zero_crossings[0])
            dt = 1.0 / fs if fs > 0 else 1.0
            features["autocorr_first_zero_lag"] = float(first_zero_idx * dt)
        else:
            features["autocorr_first_zero_lag"] = 0.0
    else:
        features["ar_coeff_1"] = 0.0
        features["ar_coeff_2"] = 0.0
        features["ar_coeff_3"] = 0.0
        features["ar_residual_variance"] = 0.0
        features["autocorr_first_zero_lag"] = 0.0

    return features


def _extract_physics(
    force_x: np.ndarray,
    accel_x: np.ndarray,
    fs: float,
    rpm: float,
    num_teeth: int,
) -> dict[str, float]:
    """Group 7: 4 Physics-Informed Metrics."""
    features: dict[str, float] = {}
    N = len(force_x)

    if N < 4 or fs <= 0.0:
        for k in [
            "harmonic_peak_ratio",
            "dominant_peak_tpf_ratio",
            "transimpedance",
            "acceleration_jerk_rms",
        ]:
            features[k] = 0.0
        return features

    f_tpf = (rpm * num_teeth) / 60.0
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    fft_vals = np.abs(np.fft.rfft(force_x)) / N
    if len(fft_vals) > 1:
        fft_vals[1:] *= 2.0
    power = fft_vals**2

    # 46. Harmonic Peak Ratio: Power(2 * TPF) / Power(TPF)
    if f_tpf > 0.0:
        idx_tpf = int(np.argmin(np.abs(freqs - f_tpf)))
        idx_2tpf = int(np.argmin(np.abs(freqs - 2.0 * f_tpf)))
        pow_tpf = float(power[idx_tpf])
        pow_2tpf = float(power[idx_2tpf])
        features["harmonic_peak_ratio"] = _safe_div(pow_2tpf, pow_tpf)
    else:
        features["harmonic_peak_ratio"] = 0.0

    # 47. Dominant Peak / TPF Ratio: f_max / f_tpf
    idx_dom = int(np.argmax(power))
    f_dom = float(freqs[idx_dom])
    if f_tpf > 0.0:
        features["dominant_peak_tpf_ratio"] = _safe_div(f_dom, f_tpf)
    else:
        features["dominant_peak_tpf_ratio"] = 0.0

    # 48. Force-to-Acceleration Transimpedance: RMS(Force_X) / RMS(Tool_X_Acceleration)
    rms_fx = _rms(force_x)
    rms_ax = _rms(accel_x)
    features["transimpedance"] = _safe_div(rms_fx, rms_ax)

    # 49. Acceleration Jerk RMS: RMS(d ax / dt)
    dt = 1.0 / fs if fs > 0 else 1.0
    jerk = np.diff(accel_x) / dt
    features["acceleration_jerk_rms"] = _rms(jerk)

    return features


# =============================================================================
# Main Public Extraction API
# =============================================================================


def extract_experiment_features(
    experiment: Experiment, num_teeth: int = 4
) -> dict[str, float]:
    """Extracts the 49 candidate feature vector from an Experiment object.

    Args:
        experiment: Valid Experiment instance loaded via Tony Dataset loader.
        num_teeth: Number of cutting tool teeth for TPF calculations (default 4).

    Returns:
        Dictionary mapping feature names to computed float values.
        Also attaches the resulting dictionary directly to `experiment.features`.
    """
    if not isinstance(experiment, Experiment):
        raise TypeError(
            f"Expected Experiment instance, got {type(experiment).__name__}."
        )

    df = experiment.signals
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        logger.warning(f"Experiment '{experiment.file_name}' contains no signal data.")
        empty_features = {k: 0.0 for k in FEATURE_NAMES_49}
        experiment.features = empty_features
        return empty_features

    # Retrieve sampling rate from metadata or compute from Time column
    fs = float(experiment.metadata.get("sampling_rate_hz", 0.0))
    if fs <= 0.0 and "Time" in df and len(df) > 1:
        t_arr = df["Time"].to_numpy()
        dt_med = float(np.median(np.diff(t_arr)))
        fs = 1.0 / dt_med if dt_med > 0.0 else 10000.0

    rpm = float(experiment.rpm)

    # Extract required signal arrays
    force_x = _get_signal(df, "Force_X")
    force_y = _get_signal(df, "Force_Y")
    accel_x = _get_signal(df, "Tool_X_Acceleration")
    displ_x = _get_signal(df, "Tool_X_Displacement")
    displ_y = _get_signal(df, "Tool_Y_Displacement")
    vel_x = _get_signal(df, "Tool_X_Velocity")

    all_features: dict[str, float] = {}

    # Extract across all 7 domain groups
    all_features.update(_extract_time_domain(force_x, accel_x, fs))
    all_features.update(
        _extract_frequency_domain(force_x, fs, rpm, num_teeth)
    )
    all_features.update(
        _extract_cross_channel(
            force_x, force_y, accel_x, displ_x, displ_y, fs, rpm, num_teeth
        )
    )
    all_features.update(_extract_wavelet(force_x, fs))
    all_features.update(_extract_nonlinear(force_x, accel_x, displ_x, vel_x, fs))
    all_features.update(_extract_autoregressive(force_x, fs))
    all_features.update(_extract_physics(force_x, accel_x, fs, rpm, num_teeth))

    # Sanitize dictionary values (preserve np.nan for non-finite values)
    sanitized: dict[str, float] = {}
    for k in FEATURE_NAMES_49:
        val = float(all_features.get(k, np.nan))
        sanitized[k] = val if np.isfinite(val) else np.nan

    experiment.features = sanitized
    return sanitized


def extract_signals_features(
    df: pd.DataFrame,
    fs: float = 10000.0,
    rpm: float = 0.0,
    num_teeth: int = 4,
) -> dict[str, float]:
    """Helper function to extract 49 candidate features directly from a DataFrame.

    Args:
        df: DataFrame containing signal columns.
        fs: Sampling frequency in Hz.
        rpm: Spindle rotational speed in RPM.
        num_teeth: Number of tool teeth.

    Returns:
        Dictionary mapping 49 feature names to float values.
    """
    dummy_exp = Experiment(
        signals=df,
        metadata={"sampling_rate_hz": fs},
        rpm=rpm,
        axial_depth=0.0,
        type="time_series",
        file_name="dataframe_dummy",
    )
    return extract_experiment_features(dummy_exp, num_teeth=num_teeth)
