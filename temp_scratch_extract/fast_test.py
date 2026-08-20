import h5py, time
import numpy as np
import scipy.stats, scipy.signal, scipy.linalg, pywt

EPS = 1e-12

def _safe_div(a, b, default=0.0):
    if abs(b) < EPS or np.isnan(b) or np.isnan(a): return default
    val = a / b
    return float(val) if np.isfinite(val) else default

def _validate_matrix(raw_data, expected_channels):
    if raw_data.ndim != 2: return None
    if raw_data.shape[1] == expected_channels: return raw_data
    if raw_data.shape[0] == expected_channels: return raw_data.T
    return None

def fast_extract_12_features(raw_matrix):
    matrix = _validate_matrix(raw_matrix, 17)
    if matrix is None:
        raise ValueError(f"Invalid shape: {raw_matrix.shape}")
    
    rpm = float(matrix[0, 0])
    axial_depth = float(matrix[0, 1])
    time_vec = matrix[:, 2]
    displ_x = matrix[:, 3]
    displ_y = matrix[:, 4]
    accel_x = matrix[:, 11]
    force_x = matrix[:, 15]
    force_y = matrix[:, 16]
    
    N = len(time_vec)
    diffs = np.diff(time_vec)
    dt = float(np.median(diffs)) if len(diffs) > 0 else 1e-4
    fs = float(1.0 / dt) if dt > 0 else 10000.0
    
    # 1. kurtosis_1st_derivative
    d_accel = np.diff(accel_x) / dt
    kurt_1st = float(scipy.stats.kurtosis(d_accel, fisher=False)) if (len(d_accel) > 2 and np.std(d_accel) > EPS) else 0.0

    # 2. ar_coeff_2 (Fast O(N) lag calculation)
    fx_zero = force_x - np.mean(force_x)
    var_fx = np.var(fx_zero)
    ar_2 = 0.0
    if var_fx > EPS and N >= 8:
        r0 = np.dot(fx_zero, fx_zero) / (var_fx * N)
        r = [r0] + [np.dot(fx_zero[:-k], fx_zero[k:]) / (var_fx * (N - k)) for k in range(1, 4)]
        r = np.array(r)
        try:
            r_mat = scipy.linalg.toeplitz(r[:3])
            phi = scipy.linalg.solve(r_mat, r[1:4])
            ar_2 = float(phi[1])
        except Exception: pass

    # 3. d2_energy & 6. d3_d4_subband_energy_ratio
    d2_energy, d3_d4_ratio = 0.0, 0.0
    if N >= 32:
        try:
            # For wavelet, if N is huge, use last 32768 samples for steady-state
            fx_sub = force_x[-32768:] if N > 32768 else force_x
            coeffs = pywt.wavedec(fx_sub, "db4", level=4)
            d4, d3, d2 = coeffs[1], coeffs[2], coeffs[3]
            e_d4, e_d3, e_d2 = float(np.sum(d4**2)), float(np.sum(d3**2)), float(np.sum(d2**2))
            d2_energy = e_d2
            d3_d4_ratio = _safe_div(e_d3, e_d4)
        except Exception: pass

    # 4. skewness_1st_derivative
    d_force = np.diff(force_x) / dt
    skew_1st = float(scipy.stats.skew(d_force)) if (len(d_force) > 2 and np.std(d_force) > EPS) else 0.0

    # 5. multi_axis_energy_asymmetry
    ex, ey = float(np.sum(force_x**2)), float(np.sum(force_y**2))
    multi_asym = _safe_div(ex - ey, ex + ey)

    # 7. coherence_at_dominant_resonant
    nperseg = min(N, 1024)
    coh_dom = 0.0
    if nperseg >= 16:
        try:
            f_acc, p_acc = scipy.signal.welch(accel_x, fs=fs, nperseg=nperseg)
            f_dom = float(f_acc[np.argmax(p_acc)])
            f_coh, cxy = scipy.signal.coherence(accel_x, force_x, fs=fs, nperseg=nperseg)
            idx_dom = int(np.argmin(np.abs(f_coh - f_dom)))
            coh_dom = float(cxy[idx_dom])
        except Exception: pass

    # 8. impulse_factor
    peak_fx, mean_abs_fx = float(np.max(np.abs(force_x))), float(np.mean(np.abs(force_x)))
    impulse_factor = _safe_div(peak_fx, mean_abs_fx)

    # 9. cross_axis_peak_delay (FFT accelerated)
    peak_delay = 0.0
    fy_zero = force_y - np.mean(force_y)
    std_x, std_y = np.std(fx_zero), np.std(fy_zero)
    if std_x > EPS and std_y > EPS and N >= 4:
        try:
            corr = scipy.signal.correlate(fx_zero, fy_zero, mode="full", method="fft") / (std_x * std_y * N)
            lags = scipy.signal.correlation_lags(N, N, mode="full")
            peak_idx = int(np.argmax(np.abs(corr)))
            peak_delay = float(abs(lags[peak_idx]) / fs)
        except Exception: pass

    # 10. cross_spectral_centroid
    csd_centroid = 0.0
    if nperseg >= 16:
        try:
            f_csd, pxy = scipy.signal.csd(force_x, force_y, fs=fs, nperseg=nperseg)
            abs_pxy = np.abs(pxy)
            sum_abs_pxy = float(np.sum(abs_pxy))
            csd_centroid = _safe_div(float(np.sum(f_csd * abs_pxy)), sum_abs_pxy)
        except Exception: pass

    # 11. phase_space_ellipsicity
    tau = max(1, min(N // 4, 5))
    ellipsicity = 0.0
    # Subsample for SVD if N is large
    step = max(1, N // 5000)
    accel_sub = accel_x[::step]
    N_sub = len(accel_sub)
    if N_sub > tau + 4:
        try:
            ps_accel = np.column_stack([accel_sub[:-tau], accel_sub[tau:]])
            ps_accel = ps_accel - np.mean(ps_accel, axis=0)
            _, s, _ = np.linalg.svd(ps_accel, full_matrices=False)
            ellipsicity = _safe_div(float(s[1]), float(s[0]))
        except Exception: pass

    # 12. bivariate_orbit_radius_ratio
    orbit_ratio = 0.0
    dx_zero, dy_zero = displ_x[::step] - np.mean(displ_x[::step]), displ_y[::step] - np.mean(displ_y[::step])
    if np.std(dx_zero) > EPS or np.std(dy_zero) > EPS:
        try:
            orbit_matrix = np.column_stack([dx_zero, dy_zero])
            _, s, _ = np.linalg.svd(orbit_matrix, full_matrices=False)
            orbit_ratio = _safe_div(float(s[1]), float(s[0]))
        except Exception: pass

    return {
        "omega_rpm": rpm,
        "axial_depth_m": axial_depth,
        "kurtosis_1st_derivative": kurt_1st,
        "ar_coeff_2": ar_2,
        "d2_energy": d2_energy,
        "skewness_1st_derivative": skew_1st,
        "multi_axis_energy_asymmetry": multi_asym,
        "d3_d4_subband_energy_ratio": d3_d4_ratio,
        "coherence_at_dominant_resonant": coh_dom,
        "impulse_factor": impulse_factor,
        "cross_axis_peak_delay": peak_delay,
        "cross_spectral_centroid": csd_centroid,
        "phase_space_ellipsicity": ellipsicity,
        "bivariate_orbit_radius_ratio": orbit_ratio,
    }

if __name__ == "__main__":
    with h5py.File("temp_scratch_extract/time32_4.h5", "r") as f:
        k = list(f.keys())[0]
        data = np.array(f[k])
    t0 = time.time()
    feats = fast_extract_12_features(data)
    t1 = time.time()
    print(f"FAST Extraction took: {t1-t0:.4f} seconds!")
    print(feats)
