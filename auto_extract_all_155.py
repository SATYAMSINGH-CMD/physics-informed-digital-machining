import os
import sys
import re
import time
import shutil
import h5py
import subprocess
import numpy as np
import pandas as pd
import scipy.stats
import scipy.signal
import scipy.linalg
import scipy.io
import pywt

# Fix Windows console encoding
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# 1. CONFIGURE KAGGLE API
os.environ["KAGGLE_API_TOKEN"] = "KGAT_549b727bd68374e3fad850a85e5a9edb"
kaggle_dir = os.path.expanduser("~/.kaggle")
os.makedirs(kaggle_dir, exist_ok=True)
with open(os.path.join(kaggle_dir, "access_token"), "w") as f:
    f.write("KGAT_549b727bd68374e3fad850a85e5a9edb\n")

# 2. OUTPUT & SCRATCH DIRS
BASE_DIR = r"D:\tony dataset"
MASTER_CSV_PATH = os.path.join(BASE_DIR, "all_datasets_features_12_master.csv")
TEMP_DIR = os.path.join(BASE_DIR, "temp_scratch_extract")
os.makedirs(TEMP_DIR, exist_ok=True)

# 3. LOAD EXISTING PROGRESS
processed_files = set()
completed_datasets = set()

if os.path.exists(MASTER_CSV_PATH):
    try:
        df_ex = pd.read_csv(MASTER_CSV_PATH)
        processed_files.update(df_ex["file"].tolist())
        completed_datasets.update(df_ex["dataset_id"].unique().tolist())
    except Exception:
        pass

print("=======================================================")
print("[*] HIGH-SPEED EXTRACTION PIPELINE (DIRECT JUMP TO MISSING)")
print(f"[*] Total cuts in Master CSV: {len(processed_files)} cuts.")
print(f"[*] Completed datasets: {sorted(list(completed_datasets))}")
print("=======================================================\n", flush=True)

EPS = 1e-12
dataset_slug = "tonylschmitz/digital-machining-database"

def clean_partial_files(dest_dir):
    for f in os.listdir(dest_dir):
        if f.endswith('.kaggle-partial') or f.endswith('.tmp'):
            try:
                os.remove(os.path.join(dest_dir, f))
            except Exception:
                pass

def safe_download(file_path, dest_dir, max_retries=3):
    clean_partial_files(dest_dir)
    cmd = f'kaggle datasets download -d {dataset_slug} -f "{file_path}" -p "{dest_dir}"'
    for attempt in range(max_retries):
        try:
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=25)
            if res.returncode == 0:
                time.sleep(0.2)
                return True
            if "429" in res.stderr:
                print(f"  [!] Rate limit (429). Pausing 30s (Attempt {attempt+1}/{max_retries})...", flush=True)
                time.sleep(30)
            elif "404" in res.stderr or "not found" in res.stderr.lower():
                return False
            else:
                time.sleep(0.3)
        except subprocess.TimeoutExpired:
            print(f"  [!] Download timed out for {file_path}. Retrying...", flush=True)
            clean_partial_files(dest_dir)
            time.sleep(1)
        except Exception:
            time.sleep(0.5)
    return False

def _safe_div(a, b, default=0.0):
    if abs(b) < EPS or np.isnan(b) or np.isnan(a): return default
    val = a / b
    return float(val) if np.isfinite(val) else default

def _validate_matrix(raw_data, expected_channels):
    if raw_data.ndim != 2: return None
    if raw_data.shape[1] == expected_channels: return raw_data
    if raw_data.shape[0] == expected_channels: return raw_data.T
    return None

def extract_12_features(raw_matrix):
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

    # 2. ar_coeff_2
    fx_zero = force_x - np.mean(force_x)
    var_fx = np.var(fx_zero)
    ar_2 = 0.0
    if var_fx > EPS and N >= 8:
        autocorr = scipy.signal.correlate(fx_zero, fx_zero, mode="full")
        mid = len(autocorr) // 2
        r = autocorr[mid : mid + 4] / (var_fx * N)
        try:
            r_mat = scipy.linalg.toeplitz(r[:3])
            phi = scipy.linalg.solve(r_mat, r[1:4])
            ar_2 = float(phi[1])
        except Exception: pass

    # 3. d2_energy & 6. d3_d4_subband_energy_ratio
    d2_energy, d3_d4_ratio = 0.0, 0.0
    if N >= 32:
        try:
            coeffs = pywt.wavedec(force_x, "db4", level=4)
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

    # 9. cross_axis_peak_delay
    peak_delay = 0.0
    fy_zero = force_y - np.mean(force_y)
    std_x, std_y = np.std(fx_zero), np.std(fy_zero)
    if std_x > EPS and std_y > EPS and N >= 4:
        try:
            corr = scipy.signal.correlate(fx_zero, fy_zero, mode="full") / (std_x * std_y * N)
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
    if N > tau + 4:
        try:
            ps_accel = np.column_stack([accel_x[:-tau], accel_x[tau:]])
            ps_accel = ps_accel - np.mean(ps_accel, axis=0)
            _, s, _ = np.linalg.svd(ps_accel, full_matrices=False)
            ellipsicity = _safe_div(float(s[1]), float(s[0]))
        except Exception: pass

    # 12. bivariate_orbit_radius_ratio
    orbit_ratio = 0.0
    dx_zero, dy_zero = displ_x - np.mean(displ_x), displ_y - np.mean(displ_y)
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

def get_stability_boundary(ds_id):
    omega_curve, blim_curve = None, None
    paths_to_try = [
        f"Dataset {ds_id} h5/stability_boundary{ds_id}.h5",
        f"Dataset {ds_id} mat/stability_boundary{ds_id}.mat",
    ]
    for p in paths_to_try:
        if safe_download(p, TEMP_DIR):
            fname = os.path.basename(p)
            local_zip = os.path.join(TEMP_DIR, f"{fname}.zip")
            local_file = os.path.join(TEMP_DIR, fname)
            if os.path.exists(local_zip):
                shutil.unpack_archive(local_zip, TEMP_DIR)
                try: os.remove(local_zip)
                except Exception: pass
            if os.path.exists(local_file):
                if local_file.endswith('.h5'):
                    try:
                        with h5py.File(local_file, "r") as fb:
                            kb = list(fb.keys())[0]
                            bmat = _validate_matrix(np.array(fb[kb]), 2)
                            if bmat is not None: omega_curve, blim_curve = bmat[:, 0], bmat[:, 1]
                    except Exception: pass
                elif local_file.endswith('.mat'):
                    try:
                        mat_dict = scipy.io.loadmat(local_file)
                        if 'omega_vector' in mat_dict and 'blim_vector' in mat_dict:
                            omega_curve = mat_dict['omega_vector'].flatten()
                            blim_curve = mat_dict['blim_vector'].flatten()
                    except Exception: pass
                try: os.remove(local_file)
                except Exception: pass
                if omega_curve is not None: return omega_curve, blim_curve
    return None, None

# ------------------------------------------------------------------------------
# 7. FAST EXTRACTION: DIRECTLY PROCESS UNEXTRACTED DATASETS
# ------------------------------------------------------------------------------
ALL_DATASET_IDS = list(range(1, 156))
# Filter to only datasets not yet fully completed
DATASET_IDS = [d for d in ALL_DATASET_IDS if d not in completed_datasets or d == 32]

print(f"[*] Total target datasets to extract: {len(DATASET_IDS)} datasets.")
print(f"[*] Queue: {DATASET_IDS}\n", flush=True)

for ds_id in DATASET_IDS:
    print(f"\n>>> [Dataset {ds_id}/155] Checking stability boundary...", flush=True)
    
    omega_curve, blim_curve = get_stability_boundary(ds_id)
    if omega_curve is None:
        print(f"  --> Stability boundary for Dataset {ds_id} not available on Kaggle. Skipping.", flush=True)
        continue

    cut_count = 0
    for cut_num in range(1, 350):
        file_name = f"time{ds_id}_{cut_num}.h5"
        file_tag = f"ds{ds_id}_{file_name}"
        
        if file_tag in processed_files:
            continue
            
        file_kaggle_path = f"Dataset {ds_id} h5/{file_name}"
        if not safe_download(file_kaggle_path, TEMP_DIR):
            break
            
        local_zip = os.path.join(TEMP_DIR, f"{file_name}.zip")
        local_h5 = os.path.join(TEMP_DIR, file_name)
        
        if os.path.exists(local_zip):
            shutil.unpack_archive(local_zip, TEMP_DIR)
            try: os.remove(local_zip)
            except Exception: pass
            
        if not os.path.exists(local_h5):
            break
            
        try:
            with h5py.File(local_h5, "r") as f:
                key = list(f.keys())[0]
                raw_data = np.array(f[key])
                
            feats = extract_12_features(raw_data)
            blim_val = float(np.interp(feats["omega_rpm"], omega_curve, blim_curve))
            label = 1 if feats["axial_depth_m"] > blim_val else 0
            
            row = {
                "dataset_id": ds_id,
                "file": file_tag,
                "boundary_m": blim_val,
                "label": label,
                **feats
            }
            
            df_row = pd.DataFrame([row])
            if not os.path.exists(MASTER_CSV_PATH):
                df_row.to_csv(MASTER_CSV_PATH, index=False)
            else:
                df_row.to_csv(MASTER_CSV_PATH, mode="a", header=False, index=False)
                
            processed_files.add(file_tag)
            cut_count += 1
            
        except Exception as e:
            pass
        finally:
            if os.path.exists(local_h5):
                try:
                    os.remove(local_h5)
                except Exception:
                    time.sleep(0.1)
                    try: os.remove(local_h5)
                    except Exception: pass
                
    print(f"  --> Dataset {ds_id} Done! ({cut_count} new cuts added. Total database: {len(processed_files)} cuts).", flush=True)

print(f"\n[+] EXTRACTION COMPLETE! Master CSV location: {MASTER_CSV_PATH}", flush=True)
