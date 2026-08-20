# PROJECT SNAPSHOT: Tony Dataset Digital Machining AI

**Document Type**: Project Reference & Architectural Map (Non-executable)  
**Workspace Root**: `d:\tony dataset`  
**Dataset Directory**: `G:\My Drive\Videos to clear my space\Dataset 1 h5`  
**Last Updated**: 2026-08-11  

---

## 1. Project Overview

* **What Tony Dataset Is**: A physics-informed machining AI package built around Tony Schmitz's Digital Machining Database (specifically Dataset 1 HDF5 files).
* **Problem Solved**: Automated detection, feature extraction, and physics-informed modeling of milling dynamics, tool-workpiece vibrations, and chatter stability boundaries.
* **Input Data**: 103 HDF5 files containing multi-channel time-series signals (`Force`, `Displacement`, `Velocity`, `Acceleration`) and stability boundary limit curves acquired across diverse spindle speeds (RPM) and axial depths of cut.
* **Eventual ML Objective**: Construct a robust chatter classification and parameter optimization model using supervised machine learning (XGBoost/LightGBM) and explainable AI (SHAP).
* **Final Output**: A unified digital twin toolkit featuring data ingestion, physics-informed 49-candidate feature extraction, automated feature pruning, predictive chatter classification, and an interactive Streamlit diagnostic explorer.

---

## 2. Current Architecture

```
Raw H5 Dataset 1 (103 files)
     │
     ▼
HDF5 Loader (loader.py)  [IMPLEMENTED]
     │
     ▼
Experiment Domain Object (experiment.py)  [IMPLEMENTED]
     │
     ▼
Physics-Informed Feature Engine (features.py - 49 Candidates)  [IMPLEMENTED]
     │
     ▼
Experiment Feature Dictionary (experiment.features)  [IMPLEMENTED]
     │
     ▼
Batch Feature Matrix Building  [PLANNED]
     │
     ▼
Pearson Correlation & Redundancy Pruning  [PLANNED]
     │
     ▼
Chatter Label Mapping (via Stability Boundary)  [PLANNED]
     │
     ▼
Supervised Model Training (XGBoost / LightGBM)  [NOT IMPLEMENTED]
     │
     ▼
SHAP Feature Importance Analysis  [NOT IMPLEMENTED]
     │
     ▼
Streamlit Interactive Dashboard (app.py)  [NOT IMPLEMENTED]
```

### Stage Status Summary
* **IMPLEMENTED**: Data Loader (`loader.py`), Domain Model (`experiment.py`), Constants Registry (`constants.py`), Plotly Visualization Engine (`visualizer.py`), Physics-Informed Feature Extractor (`features.py` - 49 candidate features), Test Suite (`tests/` - 42 passing tests).
* **PLANNED**: Batch dataset feature matrix creation, Pearson correlation analysis, chatter label generation from stability curve, feature redundancy pruning.
* **NOT IMPLEMENTED**: Model Trainer (`trainer.py`), Chatter Predictor (`predictor.py`), Parameter Recommender (`recommender.py`), Streamlit UI App (`app.py`, `streamlit_app/app.py`).

---

## 3. Complete Folder Tree

```
tony_dataset/
├── configs/
│   └── settings.yaml                      # Project YAML configuration
├── data/
│   ├── features/                          # Directory for extracted feature CSVs [EMPTY]
│   ├── processed/                         # Directory for processed datasets [EMPTY]
│   └── raw/                               # Directory for raw data copies [EMPTY]
├── docs/
│   ├── .gitkeep
│   ├── EverythingAboutProject.md          # Project documentation [STUB]
│   ├── Project_Log.md                     # Development log [STUB]
│   └── Research_Registry.md               # Research references [STUB]
├── models/
│   └── .gitkeep                           # Directory for trained ML model artifacts [EMPTY]
├── notebooks/
│   └── .gitkeep                           # Jupyter exploration notebooks [EMPTY]
├── streamlit_app/
│   ├── .gitkeep
│   └── app.py                             # Streamlit entry point [NOT IMPLEMENTED]
├── tests/
│   ├── test_constants.py                  # Unit tests for constants.py [PASSED]
│   ├── test_experiment.py                 # Unit tests for experiment.py [PASSED]
│   ├── test_features.py                   # Unit tests for features.py (49 features) [PASSED]
│   ├── test_feature_engineering.py        # Unit tests for feature_engineering.py re-exports [PASSED]
│   ├── test_loader.py                     # Unit tests for loader.py [PASSED]
│   └── test_visualizer.py                 # Unit tests for visualizer.py [PASSED]
├── tony_dataset/
│   ├── __init__.py                        # Top-level package exports
│   ├── constants.py                       # Project single source of truth constants & registries
│   ├── experiment.py                      # Experiment domain dataclass
│   ├── feature_engineering.py             # Re-export module for features.py [COMPATIBILITY ALIAS]
│   ├── features.py                        # 49-candidate physics-informed feature extraction engine
│   ├── loader.py                          # HDF5 dataset loader & validator
│   ├── predictor.py                       # ML predictor module [NOT IMPLEMENTED]
│   ├── recommender.py                     # Optimization recommender module [NOT IMPLEMENTED]
│   ├── trainer.py                         # ML model training module [NOT IMPLEMENTED]
│   ├── utils.py                           # Helper utility functions [NOT IMPLEMENTED]
│   └── visualizer.py                      # Plotly signal, FFT, heatmap & orbit visualizer
├── .gitignore
├── app.py                                 # Root application launcher [NOT IMPLEMENTED]
├── PROJECT_SNAPSHOT.md                    # Single Source of Truth Reference Document (This file)
├── README.md                              # Package description
└── requirements.txt                       # Python dependencies manifest [NEEDS REVIEW]
```

---

## 4. Purpose of Every File

| File | Status | Input | Output | Dependencies | Main Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `tony_dataset/constants.py` | **Implemented** | None | Enums, Lists, Dicts | `enum`, `re`, `typing` | Single source of truth for channel names, SI units, categories, plot styling, file extension patterns. |
| `tony_dataset/experiment.py` | **Implemented** | Dataclass parameters | `Experiment` instance, CSV export, dict summary | `dataclasses`, `pathlib`, `pandas`, `constants` | Primary domain dataclass holding signals DataFrame, RPM, depth, metadata, calculated features dict. |
| `tony_dataset/loader.py` | **Implemented** | HDF5 file path | `Experiment` object | `h5py`, `pandas`, `numpy`, `re`, `constants`, `experiment` | Validates HDF5 structure, extracts RPM (`[0,0]`) and Axial Depth (`[0,1]`), slices 15 continuous signal columns, computes $f_s$. |
| `tony_dataset/visualizer.py` | **Implemented** | `Experiment` object | `plotly.graph_objects.Figure` | `plotly`, `numpy`, `pandas`, `scipy.fft`, `constants`, `experiment` | Generates publication-ready interactive Plotly charts (Forces, Displacement, Velocity, Acceleration, Correlation Heatmap, FFT). |
| `tony_dataset/features.py` | **Implemented** | `Experiment` object, `num_teeth` | `Dict[str, float]` (49 feature keys) | `numpy`, `pandas`, `scipy`, `pywt`, `math`, `experiment` | Calculates 49 physics-informed candidate features across 7 domain groups and attaches to `experiment.features`. |
| `tony_dataset/feature_engineering.py` | **Implemented** | `Experiment` object | `Dict[str, float]` | `features.py` | Re-export alias module forwarding calls to `features.py` for backward compatibility. |
| `tony_dataset/trainer.py` | **Not Implemented** | Feature matrix, labels | Saved ML Model | `[NOT IMPLEMENTED]` | Model training pipeline for chatter classification. |
| `tony_dataset/predictor.py` | **Not Implemented** | `Experiment` / features | Prediction & confidence | `[NOT IMPLEMENTED]` | Chatter state prediction wrapper. |
| `tony_dataset/recommender.py` | **Not Implemented** | Experiment / stability map | Recommended RPM/depth | `[NOT IMPLEMENTED]` | Parameter optimization recommendation engine. |
| `tony_dataset/utils.py` | **Not Implemented** | Misc inputs | Various | `[NOT IMPLEMENTED]` | General package utilities. |
| `app.py` / `streamlit_app/app.py` | **Not Implemented** | User interactions | Web Dashboard | `streamlit` | Streamlit web application. |
| `tests/test_constants.py` | **Implemented** | None | Test results | `unittest`, `constants` | Verifies enums, signal channel definitions, unit mappings, and regex patterns. |
| `tests/test_experiment.py` | **Implemented** | Synthetic DataFrames | Test results | `unittest`, `experiment`, `constants` | Verifies `Experiment` initialization, summary generation, property aliases, and CSV export. |
| `tests/test_loader.py` | **Implemented** | Synthetic HDF5 files | Test results | `unittest`, `h5py`, `numpy`, `loader` | Verifies HDF5 loading, matrix dimensions, RPM/depth extraction, and exception handling. |
| `tests/test_visualizer.py` | **Implemented** | Synthetic Experiment | Test results | `unittest`, `visualizer` | Verifies Plotly figure generation for all signal groups, FFT, heatmaps, and statistics. |
| `tests/test_features.py` | **Implemented** | Synthetic Experiment | Test results | `unittest`, `numpy`, `pandas`, `features` | Verifies exact 49 feature extraction formulas, domain groups, edge cases (zero signals, NaNs), and DataFrame helper. |
| `tests/test_feature_engineering.py` | **Implemented** | Synthetic Experiment | Test results | `unittest`, `feature_engineering` | Verifies re-export functionality. |

---

## 5. Data Architecture

### Dataset 1 HDF5 Structure
* **File Directory**: `G:\My Drive\Videos to clear my space\Dataset 1 h5`
* **File Breakdown**: 103 files total (1 `stability_boundary1.h5` and 102 `time1_1.h5` .. `time1_102.h5`).
* **HDF5 Root Dataset Key**: Single root dataset matching filename stem (e.g. key `'time1_43'` for `time1_43.h5`).
* **Raw Matrix Orientation**: `(N, 17)` where $N \in [3840, 22560]$ rows (time samples) and 17 columns (channels).
* **Storage Data Type**: `float64` (double precision).

### Channel Layout & Operational Parameters
* **Row 0 Scalar Parameters**:
  * Column 0: `RPM` (spindle rotational speed in RPM, e.g. 2000 to 12000). Rows `1` to `N-1` are `0.0`.
  * Column 1: `Axial_Depth` (axial depth of cut in meters, e.g. 0.001 to 0.015 m). Rows `1` to `N-1` are `0.0`.
* **Continuous Signal Columns (Columns 2 to 16)**:
  * Column 2: `Time` (s) — linearly increasing from $0.0\text{ s}$ to $T_{\text{end}}$.
  * Column 3: `Tool_X_Displacement` (m)
  * Column 4: `Tool_Y_Displacement` (m)
  * Column 5: `Workpiece_X_Displacement` (m)
  * Column 6: `Workpiece_Y_Displacement` (m)
  * Column 7: `Tool_X_Velocity` (m/s)
  * Column 8: `Tool_Y_Velocity` (m/s)
  * Column 9: `Workpiece_X_Velocity` (m/s)
  * Column 10: `Workpiece_Y_Velocity` (m/s)
  * Column 11: `Tool_X_Acceleration` (m/s²)
  * Column 12: `Tool_Y_Acceleration` (m/s²)
  * Column 13: `Workpiece_X_Acceleration` (m/s²)
  * Column 14: `Workpiece_Y_Acceleration` (m/s²)
  * Column 15: `Force_X` (N)
  * Column 16: `Force_Y` (N)

### Sampling Frequency Derivation
Sampling frequency ($f_s$) is **NOT hardcoded** to 10,000 Hz. It is calculated directly per experiment from the `Time` vector:
$$f_s = \frac{1}{\text{median}(\Delta t)} \quad \text{where} \quad \Delta t = \text{diff}(\text{Time})$$
Actual sampling rates across Dataset 1 files range between **$12,266.67\text{ Hz}$ and $12,800.00\text{ Hz}$** ($12.27 - 12.80\text{ kHz}$).

---

## 6. Experiment Object

`Experiment` is defined in `tony_dataset/experiment.py` as a Python `@dataclass`.

### Primary Dataclass Fields
* `signals: Any`: pandas DataFrame containing the 15 continuous signal columns (`Time` through `Force_Y`).
* `metadata: Dict[str, Any]`: HDF5 metadata (`root_dataset_key`, `file_type`, `raw_shape`, `raw_dtype`, `sampling_rate_hz`).
* `rpm: float`: Spindle speed in RPM extracted from raw matrix `[0, 0]`.
* `axial_depth: float`: Axial depth of cut in meters extracted from raw matrix `[0, 1]`.
* `type: DatasetType`: Enum member (`TIME_SERIES` or `STABILITY_BOUNDARY`).
* `dataset_number: Optional[int]`: Dataset ID parsed from filename (e.g. `1` for `time1_43.h5`).
* `grid_point: Optional[int]`: Grid point ID parsed from filename (e.g. `43` for `time1_43.h5`).
* `duration: float`: Signal acquisition duration in seconds ($t_{\text{end}} - t_0$).
* `samples: int`: Total sample count ($N$).
* `label: Optional[str]`: Ground truth chatter state (`[NOT IMPLEMENTED / UNLABELED IN DATASET 1]`).
* `features: Dict[str, Any]`: Calculated feature dictionary (attached by `extract_experiment_features`).
* `prediction: Optional[Any]`: ML model prediction (`[NOT IMPLEMENTED]`).
* `confidence: Optional[float]`: Model confidence score (`[NOT IMPLEMENTED]`).
* `recommendation: Optional[Any]`: Parameter optimization recommendation (`[NOT IMPLEMENTED]`).
* `file_name: str`: Source HDF5 filename.

### Property Aliases & Compatibility Methods
* `data`: Alias property returning `self.signals`.
* `dataframe`: Alias property returning `self.signals`.
* `depth`: Alias property returning `self.axial_depth`.
* `is_time_series`: Returns `True` if `type == DatasetType.TIME_SERIES`.
* `is_stability_boundary`: Returns `True` if `type == DatasetType.STABILITY_BOUNDARY`.
* `feature_count`: Returns `len(self.features)`.
* `export_csv(output_path, index=False)`: Exports `signals` DataFrame to CSV.

---

## 7. Feature Engineering

The feature extraction engine in `tony_dataset/features.py` extracts **exactly 49 candidate features** from one `Experiment` instance in a single call to `extract_experiment_features(experiment, num_teeth=4)`.

### Feature Configuration Parameters
* **Tooth Passing Frequency ($f_{\text{tpf}}$)**: $f_{\text{tpf}} = \frac{\text{RPM} \cdot N_t}{60}$ (default $N_t = 4$).
* **Off-Harmonic Energy Bandwidth**: $\pm 10\text{ Hz}$ around $k \cdot f_{\text{tpf}}$ harmonics.
* **Wavelet Configuration**: Discrete Wavelet Transform using `db4`, level 4 (`pywt.wavedec`). Frequency sub-band boundaries are calculated dynamically from each experiment's actual sampling rate $f_s$.
* **Autoregressive Configuration**: AR(3) model fitted via Yule-Walker equations (`scipy.linalg.solve`).
* **Nonlinear Configuration**: Permutation Entropy ($m=4, \tau=1$), Higuchi Fractal Dimension ($k_{\text{max}}=10$), Katz Fractal Dimension (normalized), Phase Space Ellipsicity (2D SVD singular value ratio of delay embedding $[a_x(t), a_x(t+\tau)]$), Phase Space Area (2D Convex Hull area of $[x(t), \dot{x}(t)]$).

### Complete 49-Feature Matrix Table

| # | Feature Name | Domain Group | Primary Signal(s) | Calculation Method |
| :-: | :--- | :--- | :--- | :--- |
| **1** | `kurtosis` | Time Domain | `Force_X` | Fisher=False 4th central moment / $\sigma^4$ |
| **2** | `skewness` | Time Domain | `Force_X` | 3rd central moment / $\sigma^3$ |
| **3** | `crest_factor` | Time Domain | `Force_X` | $\max(\vert X\vert) / \text{RMS}(X)$ |
| **4** | `margin_factor` | Time Domain | `Force_X` | $\max(\vert X\vert) / (\text{mean}(\sqrt{\vert X\vert}))^2$ |
| **5** | `shape_factor` | Time Domain | `Force_X` | $\text{RMS}(X) / \text{mean}(\vert X\vert)$ |
| **6** | `impulse_factor` | Time Domain | `Force_X` | $\max(\vert X\vert) / \text{mean}(\vert X\vert)$ |
| **7** | `zero_crossing_rate` | Time Domain | `Tool_X_Acceleration` | Mean-centered sign change count / $(N-1)$ |
| **8** | `mean_crossing_rate` | Time Domain | `Tool_X_Acceleration` | Mean-centered sign change count / $(N-1)$ |
| **9** | `kurtosis_1st_derivative` | Time Domain | `Tool_X_Acceleration` | Kurtosis of numerical derivative $\frac{da_x}{dt}$ |
| **10** | `skewness_1st_derivative` | Time Domain | `Force_X` | Skewness of numerical derivative $\frac{dF_x}{dt}$ |
| **11** | `off_harmonic_energy_ratio` | Frequency Domain | `Force_X` | $1 - (E_{\text{harmonics}} / E_{\text{total}})$ using $\pm 10\text{ Hz}$ around $k f_{\text{tpf}}$ |
| **12** | `spectral_centroid` | Frequency Domain | `Force_X` | $\sum f \cdot \vert X(f)\vert / \sum \vert X(f)\vert$ |
| **13** | `spectral_entropy` | Frequency Domain | `Force_X` | Shannon entropy of normalized power spectrum |
| **14** | `spectral_flatness` | Frequency Domain | `Force_X` | Geometric mean / Arithmetic mean of power spectrum |
| **15** | `spectral_rolloff_85` | Frequency Domain | `Force_X` | Frequency below which 85% spectral magnitude lies |
| **16** | `spectral_spread` | Frequency Domain | `Force_X` | Standard deviation around Spectral Centroid |
| **17** | `spectral_skewness` | Frequency Domain | `Force_X` | 3rd moment around Spectral Centroid |
| **18** | `spectral_kurtosis` | Frequency Domain | `Force_X` | 4th moment around Spectral Centroid |
| **19** | `cross_correlation_coeff` | Cross-Channel | `(Force_X, Force_Y)` | Peak normalized cross-correlation magnitude |
| **20** | `cross_axis_peak_delay` | Cross-Channel | `(Force_X, Force_Y)` | Time lag $\vert\tau_{\text{peak}}\vert / f_s$ at correlation peak |
| **21** | `bivariate_orbit_radius_ratio` | Cross-Channel | `(Tool_X_Displ, Tool_Y_Displ)` | 2D SVD singular value ratio $s_2 / s_1$ of orbit trajectory |
| **22** | `coherence_at_tpf` | Cross-Channel | `(Force_X, Force_Y)` | Magnitude-squared coherence at $f_{\text{tpf}}$ |
| **23** | `coherence_at_dominant_resonant` | Cross-Channel | `(Tool_X_Accel, Force_X)` | Coherence at acceleration peak frequency $f_{\text{dom}}$ |
| **24** | `cross_spectral_centroid` | Cross-Channel | `(Force_X, Force_Y)` | Centroid of Cross-Spectral Density magnitude $\vert P_{xy}(f)\vert$ |
| **25** | `multi_axis_energy_asymmetry` | Cross-Channel | `(Force_X, Force_Y)` | Energy difference $(E_x - E_y) / (E_x + E_y)$ |
| **26** | `d1_energy` | Wavelet | `Force_X` | Sub-band energy of D1 detail (`db4`, L4) |
| **27** | `d2_energy` | Wavelet | `Force_X` | Sub-band energy of D2 detail |
| **28** | `d3_energy` | Wavelet | `Force_X` | Sub-band energy of D3 detail |
| **29** | `d4_energy` | Wavelet | `Force_X` | Sub-band energy of D4 detail |
| **30** | `a4_energy` | Wavelet | `Force_X` | Sub-band energy of A4 approximation |
| **31** | `wavelet_energy_entropy` | Wavelet | `Force_X` | Shannon entropy of relative sub-band energies |
| **32** | `d1_wavelet_kurtosis` | Wavelet | `Force_X` | Kurtosis of D1 detail coefficients |
| **33** | `d3_wavelet_std` | Wavelet | `Force_X` | Standard deviation of D3 detail coefficients |
| **34** | `d1_relative_energy` | Wavelet | `Force_X` | $E_{D1} / E_{\text{total\_wavelet}}$ |
| **35** | `d3_d4_subband_energy_ratio` | Wavelet | `Force_X` | $E_{D3} / E_{D4}$ |
| **36** | `permutation_entropy` | Nonlinear | `Force_X` | Ordinal pattern entropy ($m=4, \tau=1$) |
| **37** | `higuchi_fractal_dimension` | Nonlinear | `Force_X` | Higuchi fractal dimension ($k_{\text{max}}=10$) |
| **38** | `katz_fractal_dimension` | Nonlinear | `Force_X` | Normalized Katz fractal dimension |
| **39** | `phase_space_ellipsicity` | Nonlinear | `Tool_X_Acceleration` | 2D SVD singular value ratio of delay embedding |
| **40** | `phase_space_area` | Nonlinear | `(Tool_X_Displ, Tool_X_Vel)` | 2D Convex Hull area of displacement vs velocity phase orbit |
| **41** | `ar_coeff_1` | Autoregressive | `Force_X` | First Yule-Walker AR(3) coefficient $\phi_1$ |
| **42** | `ar_coeff_2` | Autoregressive | `Force_X` | Second Yule-Walker AR(3) coefficient $\phi_2$ |
| **43** | `ar_coeff_3` | Autoregressive | `Force_X` | Third Yule-Walker AR(3) coefficient $\phi_3$ |
| **44** | `ar_residual_variance` | Autoregressive | `Force_X` | Residual error variance $\sigma_e^2$ of AR(3) model fit |
| **45** | `autocorr_first_zero_lag` | Autoregressive | `Force_X` | Lag time (seconds) of first zero-crossing of autocorrelation |
| **46** | `harmonic_peak_ratio` | Physics-Informed | `Force_X` | Power at $2 f_{\text{tpf}}$ / Power at $f_{\text{tpf}}$ |
| **47** | `dominant_peak_tpf_ratio` | Physics-Informed | `Force_X` | Peak frequency $f_{\text{max}} / f_{\text{tpf}}$ |
| **48** | `transimpedance` | Physics-Informed | `(Force_X, Tool_X_Accel)` | Dynamic compliance ratio $\text{RMS}(F_x) / \text{RMS}(a_x)$ |
| **49** | `acceleration_jerk_rms` | Physics-Informed | `Tool_X_Acceleration` | RMS of numerical jerk $\text{RMS}(\frac{da_x}{dt})$ |

---

## 8. Feature Selection Plan `[PLANNED]`

The post-extraction feature selection pipeline is **PLANNED** for the next development phase:

1. **Batch Extraction**: Load all 102 time-series experiments in Dataset 1 and extract the 49-candidate feature vector per experiment $\rightarrow$ $102 \times 49$ Feature Matrix.
2. **Correlation Analysis**: Compute Pearson correlation matrix $R$ across all 49 features.
3. **Redundancy Pruning**: Identify collinear pairs ($|r| > 0.85$ or $|r| > 0.90$) and prune redundant features based on domain priority and signal stability.
4. **Separation Analysis**: Evaluate class separability (ANOVA F-score / Mutual Information) against stability labels.
5. **Model-Based Importance**: Train preliminary XGBoost / LightGBM classifiers and calculate SHAP (SHapley Additive exPlanations) values.
6. **Final Feature Reduction**: Select approximately **6 compact, highly discriminative, non-redundant features** for production deployment.

---

## 9. ML Architecture `[NOT IMPLEMENTED]`

* **Ground Truth Labels**: `[NOT IMPLEMENTED / UNLABELED IN DATASET 1]`. Dataset 1 HDF5 files contain raw signal measurements without explicit "Stable" / "Unstable" labels. Labels must be generated by mapping experiment RPM and Axial Depth against `stability_boundary1.h5`.
* **Train / Test Split Strategy**: `[NOT IMPLEMENTED]`. Stratified $k$-fold cross-validation or grid point splitting.
* **Candidate ML Models**: `[NOT IMPLEMENTED]`. XGBoost, LightGBM, Random Forest, Support Vector Machine.
* **Explainability Pipeline**: `[NOT IMPLEMENTED]`. SHAP tree explainer to quantify feature contributions.
* **Evaluation Metrics**: `[NOT IMPLEMENTED]`. F1-score (weighted), Precision, Recall, ROC-AUC, Confusion Matrix.

---

## 10. Visualization

The visualization engine in `tony_dataset/visualizer.py` provides publication-ready Plotly charts.

### Implemented Plotly Functions
* `plot_signal(experiment, x_col, y_col)`: Plot arbitrary signal channel vs Time.
* `plot_force(experiment)`: Dual-trace plot of `Force_X` and `Force_Y` vs Time.
* `plot_displacement(experiment)`: Multi-trace plot of Tool & Workpiece X/Y displacements.
* `plot_velocity(experiment)`: Multi-trace plot of Tool & Workpiece X/Y velocities.
* `plot_acceleration(experiment)`: Multi-trace plot of Tool & Workpiece X/Y accelerations.
* `plot_histogram(experiment, signal_col)`: Histogram sample distribution of a signal.
* `plot_correlation(experiment)`: Heatmap correlation matrix across numerical signal channels.
* `plot_fft(experiment, signal_col)`: SciPy FFT magnitude frequency spectrum.
* `plot_all(experiment)`: Returns dictionary of all standard figures.
* `get_statistics(experiment)`: Computes statistical summary DataFrame (Mean, Std, Min, Max, Peak, RMS).

### Planned Interfaces
* `app.py` / `streamlit_app/app.py`: Interactive Streamlit web application `[NOT IMPLEMENTED]`.

---

## 11. Dependencies

| Package | Used By | Purpose | Installed? |
| :--- | :--- | :--- | :---: |
| `h5py` | `loader.py` | HDF5 binary file reading and structure inspection | Yes (v3.16.0) |
| `numpy` | `loader.py`, `visualizer.py`, `features.py` | Array math, linear algebra, FFT, SVD, statistics | Yes (v2.4.2) |
| `pandas` | `loader.py`, `visualizer.py`, `features.py`, `experiment.py` | Signal DataFrames, statistics tables, CSV exports | Yes (v2.3.3) |
| `scipy` | `visualizer.py`, `features.py` | FFT, Welch PSD, CSD, Coherence, SVD, ConvexHull, Yule-Walker AR | Yes (v1.17.1) |
| `pywt` (PyWavelets) | `features.py` | Discrete Wavelet Transform (`db4`, level 4 decomposition) | Yes (v1.9.0) |
| `plotly` | `visualizer.py` | Publication-ready interactive charts and heatmaps | Yes (v6.5.2) |
| `streamlit` | `streamlit_app/app.py` | Interactive web dashboard `[NOT IMPLEMENTED]` | Yes (v1.54.0) |
| `xgboost` | `trainer.py` | Machine learning classifier `[NOT IMPLEMENTED]` | Yes (v3.2.0) |
| `lightgbm` | `trainer.py` | Machine learning classifier `[NOT IMPLEMENTED]` | Yes (v4.6.0) |
| `shap` | `trainer.py` | Explainable AI feature importance `[NOT IMPLEMENTED]` | Yes (v0.52.0) |

*Note*: `requirements.txt` is currently an empty 1-byte file and needs to be updated with project requirements (`[NEEDS REVIEW]`).

---

## 12. Tests

The project includes an automated test suite using Python's built-in `unittest` framework located in `tests/`.

### Test Summary
* **Total Test Files**: 6 files
* **Total Test Cases**: 42 tests
* **Execution Command**: `python -m unittest discover tests`
* **Latest Test Result**: **`OK (Ran 42 tests in 0.960s)`**

### Breakdown by Test Module
1. `tests/test_constants.py`: 10 tests verifying enums, column definitions, units, plot defaults, regex patterns.
2. `tests/test_experiment.py`: 7 tests verifying `Experiment` initialization, summary, alias properties (`data`, `depth`), CSV export.
3. `tests/test_loader.py`: 7 tests verifying HDF5 dataset loading, matrix dimension handling, RPM/depth extraction, sampling rate calculation, and custom exceptions.
4. `tests/test_visualizer.py`: 6 tests verifying Plotly figure generation for signal groups, FFT, heatmaps, and statistical summaries.
5. `tests/test_features.py`: 11 tests verifying the 49 candidate feature extraction formulas, domain groups, edge cases (zero signals, NaNs, empty DataFrames), and DataFrame helper.
6. `tests/test_feature_engineering.py`: 1 test verifying backwards-compatibility re-exports.

---

## 13. Current Project Status

### Completed Checklist
* [x] Project structure
* [x] Constants single source of truth (`constants.py`)
* [x] Experiment domain model (`experiment.py`)
* [x] Real HDF5 data loader (`loader.py`)
* [x] Signal & FFT visualizer (`visualizer.py`)
* [x] 49-candidate physics-informed feature extraction engine (`features.py`)
* [x] Complete unit test suite (42 passing tests)

### Next Development Steps
1. **Batch Feature Extraction**: Build a batch processing script to extract the 49-feature vector for all 102 Dataset 1 experiments and save to `data/features/dataset1_features.csv`.
2. **Stability Label Generation**: Map experiment RPM and Axial Depth parameters against `stability_boundary1.h5` limit curve to assign ground truth chatter labels (`Stable` vs `Unstable`).
3. **Correlation & Redundancy Pruning**: Perform Pearson correlation matrix analysis across the 49 extracted features to eliminate redundant collinear features ($|r| > 0.85$).
4. **ML Model Training & SHAP**: Implement `trainer.py` to train XGBoost / LightGBM classifiers and evaluate feature importance using SHAP.
5. **Streamlit UI Application**: Build `streamlit_app/app.py` for interactive dataset exploration, feature distribution plots, and real-time chatter prediction.

---

## 14. Known Problems / Compatibility Issues

1. **`constants.py` Legacy Feature Registry `[COMPATIBILITY ISSUE]` / `[NEEDS REVIEW]`**:
   - `constants.py` Section 7 contains legacy feature lists (`TIME_DOMAIN_FEATURES`, `FREQUENCY_DOMAIN_FEATURES`, `PHYSICS_FEATURES`, `DECISION_FEATURES`) with old string names (e.g. `"Mean"`, `"Std"`, `"Variance"`, `"Peak"`) from initial prototyping.
   - `features.py` defines and extracts the audited **49 candidate feature set** (`FEATURE_NAMES_49`).
   - *Recommendation*: `constants.py` Section 7 should be updated in a future refactoring to reflect `FEATURE_NAMES_49`.

2. **Unlabeled HDF5 Signals `[NOT IMPLEMENTED]`**:
   - Raw Dataset 1 HDF5 files contain signal arrays but no ground truth chatter classification labels (`label` attribute is `None`).
   - Labels must be derived by comparing (RPM, Axial Depth) pairs against `stability_boundary1.h5`.

3. **Empty Requirements Manifest `[NEEDS REVIEW]`**:
   - `requirements.txt` is an empty 1-byte file, although installed packages (`numpy`, `pandas`, `scipy`, `pywt`, `h5py`, `plotly`) are actively used.

4. **Stub Files `[NOT IMPLEMENTED]`**:
   - `app.py`, `streamlit_app/app.py`, `tony_dataset/trainer.py`, `tony_dataset/predictor.py`, `tony_dataset/recommender.py`, `tony_dataset/utils.py` are empty 1-byte stubs.

---

## 15. Important Engineering Decisions

1. **Candidate Feature Bank Size**: Selected **49 candidate features** covering 7 mathematical domains for initial extraction on real Tony data before applying correlation pruning.
2. **Single Vector Per Experiment**: One experiment produces **ONE 49-element feature vector**. Sliding window segmentation is deferred to a later early-warning phase.
3. **No Hardcoded Sampling Frequency**: Sampling rate ($f_s$) is computed dynamically per experiment from $\frac{1}{\text{median}(\Delta t)}$ ($f_s \approx 12.266 - 12.800\text{ kHz}$).
4. **Primary Signal Selection**:
   - Cutting Force: `Force_X` (and `(Force_X, Force_Y)` for cross-channel).
   - Tool Vibration Acceleration: `Tool_X_Acceleration`.
   - Tool Displacement Orbit: `(Tool_X_Displacement, Tool_Y_Displacement)`.
5. **Tooth Passing Frequency Calculation**: $f_{\text{tpf}} = \frac{\text{RPM} \cdot N_t}{60}$ with default $N_t = 4$ teeth.
6. **Wavelet Settings**: `db4` wavelet, level 4 decomposition (`pywt.wavedec`). Dynamic sub-band boundaries derived from $f_s$.
7. **Off-Harmonic Ratio Bandwidth**: $\pm 10\text{ Hz}$ bandwidth around tooth passing frequency harmonics ($k \cdot f_{\text{tpf}}$).
8. **Phase Space Ellipsicity**: SVD singular value ratio $\frac{\sigma_2}{\sigma_1}$ of 2D delay embedding $[a_x(t), a_x(t+\tau)]$.
9. **Phase Space Area**: 2D Convex Hull area (`scipy.spatial.ConvexHull`) of Tool X displacement vs Tool X velocity orbit.
10. **Transimpedance Definition**: Dynamic compliance proxy $\frac{\text{RMS}(\text{Force\_X})}{\text{RMS}(\text{Tool\_X\_Acceleration})}$.

---

## AI DEVELOPMENT RULES

* **Read `PROJECT_SNAPSHOT.md` First**: Always consult this document before proposing architectural or code modifications.
* **Inspect Source Code First**: Verify actual function signatures, class attributes, and file structures before making assumptions.
* **Do Not Invent Data**: Rely strictly on verified HDF5 datasets and metadata. Do not hardcode fictitious values or unverified constants.
* **Do Not Alter Feature Definitions Silently**: Preserve the mathematical definitions of the 49 candidate features as specified in `features.py`.
* **Maintain Architectural Scope**: Do not modify unrelated modules or introduce unnecessary heavy frameworks.
* **Run Test Suite**: Run `python -m unittest discover tests` after any meaningful code modification and verify all tests pass (`OK`).
* **Distinguish Implemented vs Planned**: Clearly label new features, tasks, or pipeline steps as `[IMPLEMENTED]`, `[PLANNED]`, or `[NOT IMPLEMENTED]`.
