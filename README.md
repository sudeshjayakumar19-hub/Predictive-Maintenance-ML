# Predictive Maintenance Using Machine Learning

## Live Demo
[Streamlit deployment link will be added after deployment]

## GitHub Repository
[GitHub repository link will be added]

---

## Project Overview
This project provides an end-to-end Machine Learning pipeline and interactive operations dashboard for turbofan aircraft engine predictive maintenance. Using multivariate degradation telemetry from the benchmark NASA C-MAPSS dataset, the system forecasts impending equipment failure risk across operating cycles, enabling engineering teams to prevent unscheduled downtime, reduce catastrophic failure risk, and eliminate unnecessary premature maintenance.

---

## Problem Statement
In industrial fleet operations, turbofan engines undergo progressive wear and thermodynamic degradation. Relying purely on reactive maintenance (repairing after failure) risks in-flight safety incidents and major operational disruption. Conversely, rigid calendar-based maintenance schedules often replace healthy components prematurely. The challenge is to accurately identify impending system failure within a clear operational decision window while controlling false alarms.

---

## Objective
* Develop a robust, anti-leakage Machine Learning pipeline on turbofan engine degradation time-series data.
* Predict binary failure risk within an actionable operational window before functional failure.
* Compare multiple classification algorithms with class-imbalance mitigation using time-aware chronological evaluation.
* Optimize decision thresholds to prioritize safety-critical failure recall while minimizing false alarm rates.
* Deliver an interactive, deployment-ready Streamlit operations dashboard for real-time fleet health monitoring and early warning alerts.

---

## Dataset
* **Source**: NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) Turbofan Engine Degradation Dataset
* **Subset**: **FD001**
  * **Operating Conditions**: ONE (Sea Level)
  * **Fault Mode**: ONE (High-Pressure Compressor [HPC] Degradation)
  * **Fleet Size**: 100 complete run-to-failure engine trajectories (Units 1 to 100)
  * **Total Observations**: 20,631 operational snapshot cycles
  * **Telemetry**: 3 operational flight condition settings and 21 physical engine sensor measurements (temperatures, pressures, fan and core speeds, bypass ratio, coolant bleed rates).

---

## Failure Prediction Window
* **Window Duration**: **30 cycles**
* **Ground Truth Remaining Useful Life (RUL)**:
  $$\text{RUL}_{i, t} = \max(\text{cycle}_i) - \text{cycle}_{i, t}$$
* **Binary Failure-Risk Target**:
  $$\text{target} = \begin{cases} 1 & \text{if } \text{RUL} \le 30 \text{ (Impending Failure Risk)} \\ 0 & \text{if } \text{RUL} > 30 \text{ (Nominal Operation)} \end{cases}$$
* **Class Distribution**:
  * Positive (Failure Risk, 1): **3,100 samples (15.03%)**
  * Negative (Healthy Operation, 0): **17,531 samples (84.97%)**
  * Imbalance Ratio: **5.66 : 1**

---

## Feature Engineering
To strictly prevent target leakage and simulate realistic production inference, 56 time-aware features were engineered strictly using backward-looking/current observations grouped per engine (`unit_number`):
* **Lag Features (`lag_1`)**: Sensor readings from the previous operational cycle ($t - 1$).
* **Cycle-to-Cycle Deltas (`delta`)**: Rate of change between consecutive cycles ($x_t - x_{t-1}$).
* **Rolling Statistics (`roll_mean_5`, `roll_std_5`)**: 5-cycle backward rolling mean and rolling standard deviation capturing smoothed thermodynamic degradation trends and volatility.
* **Leakage Guardrails**: RUL and target labels were strictly excluded from model feature sets. Engine IDs were retained solely for grouped splitting and tracking.

---

## Time-Aware Evaluation
To prevent future temporal leakage and cross-engine contamination, an engine fleet-level chronological split was used:
* **Training Fleet**: Engines 1 to 80 (`16,138` observations, 78.22%)
* **Testing Fleet**: Engines 81 to 100 (`4,493` observations, 21.78%)
* Unseen test fleet lifecycles were evaluated completely out-of-sample with no past/future cycle mixing.

---

## Models
Three machine-learning algorithms were trained and calibrated with class-imbalance mitigation:
* **Logistic Regression**: Scaled with `StandardScaler` inside a Pipeline, trained with `class_weight='balanced'`.
* **Random Forest**: Ensemble of 100 decision trees (`max_depth=15`, `class_weight='balanced'`).
* **XGBoost**: Gradient boosted decision trees (`scale_pos_weight=5.51`, `max_depth=6`).

---

## Final Model
* **Champion Algorithm**: **Random Forest Classifier**
* **Hyperparameters**: `n_estimators=100`, `max_depth=15`, `min_samples_leaf=2`, `class_weight='balanced'`
* **Selection Rationale**:
  * Achieved the highest **PR-AUC (0.9456)** and highest **ROC-AUC (0.9890)** under heavy class imbalance.
  * Delivered the highest **F1-Score (0.8480)**.
  * Slashed false alarms by **40%** compared to Logistic Regression (153 vs. 254).

---

## Alert Threshold
* **Selected Project Operating Threshold**: **`0.40`**
* **Operating Rationale**:
  * Prioritizes safety by boosting failure recall from `91.77%` to **`94.52%`**, catching 17 additional impending failures (missed failures cut from 51 to 34).
  * Limits the False Alarm Rate to **`5.65%`** (over 94.3% of healthy operation remains unflagged).
  * Maintains strong operational precision of **`72.80%`**.
* *Note*: This is a project-defined operating threshold optimized for this predictive maintenance scenario, not a universal maintenance rule.

---

## Evaluation Metrics
Evaluated on the unseen chronological test fleet (Units 81–100, `4,493` cycles) at default ($\tau = 0.50$) and calibrated operating threshold ($\tau = 0.40$):

### Cross-Model Comparison ($\tau = 0.50$)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | False Alarms (FP) | Missed Failures (FN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9350 | 0.6962 | 0.9387 | 0.7995 | 0.9875 | 0.9376 | 254 | 38 |
| **Random Forest (Champion)** | **0.9546** | **0.7881** | **0.9177** | **0.8480** | **0.9890** | **0.9456** | **153** | **51** |
| **XGBoost** | 0.9475 | 0.7588 | 0.9081 | 0.8267 | 0.9880 | 0.9456 | 179 | 57 |

### Random Forest Threshold Optimization Table

| Threshold | Precision | Recall | F1 | False Alarms (FP) | Missed (FN) | False Alarm Rate | Total Alerts |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.20** | 0.6095 | 0.9742 | 0.7498 | 387 | 16 | 9.99% | 991 |
| **0.30** | 0.6637 | 0.9613 | 0.7852 | 302 | 24 | 7.80% | 898 |
| **0.40 (Selected)** | **0.7280** | **0.9452** | **0.8225** | **219** | **34** | **5.65%** | **805** |
| **0.50** | 0.7881 | 0.9177 | 0.8480 | 153 | 51 | 3.95% | 722 |
| **0.60** | 0.8240 | 0.8758 | 0.8491 | 116 | 77 | 3.00% | 659 |
| **0.70** | 0.8817 | 0.8532 | 0.8672 | 71 | 91 | 1.83% | 600 |
| **0.80** | 0.9223 | 0.7855 | 0.8484 | 41 | 133 | 1.06% | 528 |

---

## High-Risk Alert System
Risk is categorized into an actionable 3-tier operational framework:
* 🟢 **LOW RISK (`P < 0.30`)**: Normal operating conditions; routine scheduled maintenance.
* 🟡 **MEDIUM RISK (`0.30 <= P < 0.40`)**: Advisory warning; early thermodynamic divergence.
* 🔴 **HIGH RISK (`P >= 0.40`)**: High-priority alert; impending failure predicted within 30 cycles.

### Top Model-Important Statistical Indicators
1. `sensor_4_roll_mean_5` (Low-Pressure Turbine Outlet Temperature - T50)
2. `sensor_15_roll_mean_5` (Bypass Ratio - BPR)
3. `sensor_21_roll_mean_5` (Low-Pressure Turbine Coolant Bleed - W32)
4. `sensor_11_roll_mean_5` (High-Pressure Compressor Outlet Static Pressure - Ps30)
5. `sensor_3_roll_mean_5` (High-Pressure Compressor Outlet Temperature - T30)

*Disclosure: Feature importance reflects statistical association identified by the model; it does not claim root physical causality.*

---

## Streamlit Dashboard
The interactive operations dashboard (`app.py`) provides:
1. **Fleet Health Overview**: High-level KPIs tracking monitored units, high-risk counts, and operating threshold.
2. **Selected Equipment Status**: Current cycle telemetry, estimated risk percentage, and visual status badges.
3. **Failure Risk Trajectory**: Historical failure probability curve over cycle lifespan with alert threshold demarcation.
4. **Multi-Sensor Telemetry & Trend Analysis**: Interactive charts showing raw sensor signals alongside causal rolling trends.
5. **High-Risk Equipment Alerts**: Ranked operational alert table prioritizing critical equipment.
6. **Model Governance & Threshold Analysis**: Transparent comparison tables and interactive threshold optimization curves.
7. **Feature Importance Inspection**: Bar charts highlighting top model-important features.

---

## Project Structure

```text
predictive-maintenance/
│
├── app.py                                  # Streamlit operations dashboard
├── train.py                                # End-to-end model training & evaluation script
├── setup_model.py                          # Environment verification & setup utility
├── requirements.txt                        # Application dependencies
├── README.md                               # Project documentation
│
├── data/
│   ├── train_FD001.txt                     # NASA C-MAPSS training set (100 engines)
│   ├── test_FD001.txt                      # NASA C-MAPSS test set (100 engines)
│   ├── RUL_FD001.txt                       # Ground truth test RUL
│   ├── readme.txt                          # Official benchmark documentation
│   └── processed_fd001.csv                 # Preprocessed dataset with engineered features
│
├── models/
│   └── predictive_maintenance_model.joblib # Champion Random Forest model bundle (4.9 MB)
│
├── src/
│   ├── __init__.py                         # Package initialization
│   ├── preprocessing.py                    # Preprocessing & lag/rolling feature pipeline
│   ├── alert_system.py                     # Threshold tuning & alert generation
│   └── predict.py                          # Model loading & inference utilities
│
├── notebooks/
│   └── predictive_maintenance_analysis.ipynb # End-to-end analysis notebook
│
└── outputs/
    ├── feature_list.txt                    # List of 80 input features
    ├── model_metrics.csv                   # Model comparison table
    ├── threshold_analysis.csv              # 7-threshold sensitivity analysis
    ├── high_risk_alerts.csv                # Cycle-level alert records sorted by risk
    ├── high_risk_equipment.csv             # Equipment-level high-risk units
    ├── equipment_risk_summary.csv          # Fleet-wide risk summary
    ├── feature_importance.csv              # Top 15 features across models
    └── plots/                              # High-resolution diagnostic plots (01 to 16)
```

---

## Installation & Running

```bash
# Clone the repository
git clone <repository-url>
cd predictive-maintenance

# Install required dependencies
pip install -r requirements.txt

# Verify environment and artifacts
python setup_model.py

# Launch the interactive Streamlit dashboard
streamlit run app.py
```

The dashboard will open locally at `http://localhost:8501`.

---

## Deployment
This repository is pre-configured for one-click deployment on **Streamlit Community Cloud**:
* `app.py` and `requirements.txt` are situated in the repository root.
* Pre-trained model bundle (`models/predictive_maintenance_model.joblib`) is included.
* All paths are relative and self-contained; zero hardcoded local machine paths.
* Zero startup retraining: uses `@st.cache_resource` for immediate inference.

---

## Disclaimer
⚠️ **Operational Disclaimer**: *Predicted failure risk is a machine-learning estimate intended as an operational decision-support signal. It does not guarantee that equipment will fail. Maintenance decisions should incorporate certified engineering inspection, environmental context, and physical diagnostic verification.*
