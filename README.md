# ⚙️ Physics-Informed Digital Machining AI & Digital Twin

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Dataset](https://img.shields.io/badge/Dataset-Schmitz_Digital_Machining-orange.svg)](https://github.com/tonyschmitz)

An end-to-end, physics-informed digital twin framework for real-time milling chatter classification, cross-tool stability generalization, and sub-millisecond edge latency benchmarking across **9,160 experimental cuts and 42 distinct dynamic setups**.

---

## 🏗️ End-to-End System Architecture

```
                      RAW MACHINING SENSOR STREAM
                                   │
                                   ▼
                         Signal Preprocessing
                                   │
                                   ▼
                Physics-Informed Feature Extraction (12 Features)
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    ▼                             ▼
           Classical Ensembles           Physics-Informed Models
         (XGBoost / LightGBM / RF)           (PINN / MLP)
                    │                             │
                    │                      Data + Physics Loss
                    │                             │
                    └──────────────┬──────────────┘
                                   ▼
                        Classification Engine
                                   │
                                   ▼
             Stratified 5-Fold vs GroupKFold (Cross-Tool)
                                   │
                                   ▼
                       Generalization Analysis
                                   │
                                   ▼
                 Real-Time Sliding Buffer Benchmark (50 ms)
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
                 XGBoost        LightGBM      MLP / ONNX
                    │              │              │
                    └──────────────┼──────────────┘
                                   ▼
                  Sub-ms Latency Profile (p50 / p95 / p99)
                                   │
                                   ▼
             Interactive Digital Twin Dashboard (Streamlit)
```

---

## 🏆 Master Benchmark Results (9,160 Cuts across 42 Configurations)

| Model | Stratified CV Accuracy | Stratified CV F1 | Stratified ROC-AUC | GroupKFold (LODO) Accuracy | GroupKFold F1 | GroupKFold ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM-12** | **92.85% ± 0.62%** | **0.9239** | **0.9856** | **90.68% ± 3.02%** | **0.8913** | **0.9659** |
| **XGBoost-12** | **92.84% ± 0.49%** | **0.9237** | **0.9853** | **90.53% ± 3.12%** | **0.8900** | **0.9669** |
| **RandomForest-12** | 92.48% ± 0.68% | 0.9191 | 0.9822 | 90.41% ± 3.11% | 0.8858 | 0.9635 |
| **XGBoost-7 (Pruned)** | 91.35% ± 1.07% | 0.9072 | 0.9758 | 87.95% ± 4.13% | 0.8593 | 0.9446 |
| **MLP Neural Net** | 89.98% ± 1.37% | 0.8909 | 0.9645 | 85.84% ± 3.53% | 0.8422 | 0.9254 |

---

## ⏱️ Real-Time Edge Latency Profiling (5,000 Continuous Sliding Windows)

| Pipeline Stage | Mean (ms) | Median p50 (ms) | p95 (ms) | p99 (ms) | Hard Real-Time Compliance (< 50 ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Feature Extraction** | 6.673 ms | 6.649 ms | 8.210 ms | 10.354 ms | **PASS** |
| **MLP / Neural Net Inference** | 0.644 ms | 0.629 ms | 0.808 ms | 1.120 ms | **PASS** |
| **XGBoost Inference** | 1.406 ms | 1.064 ms | 2.348 ms | 11.181 ms | **PASS** |
| **LightGBM Inference** | 1.932 ms | 1.705 ms | 2.709 ms | 7.626 ms | **PASS** |
| **Total Round-Trip (XGBoost)** | **8.079 ms** | **7.762 ms** | **10.563 ms** | **19.337 ms** | **PASS ($\ll 50\text{ ms}$)** |

---

## 🚀 Quickstart & Interactive Digital Twin

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run master benchmark
python scripts/benchmark_master_9160.py

# 3. Run real-time latency profiler
python scripts/profile_realtime_latency.py

# 4. Launch Interactive Streamlit Digital Twin Dashboard
streamlit run app.py
```

---

## 📚 Documentation & Research
* [Research Abstract & Paper Draft](docs/Research_Abstract_Paper.md)
* [Data Scarcity & Physics Ablation Results](research_ablation_results.csv)
* [Real-Time Latency Benchmark](realtime_latency_benchmark.csv)
