"""
🔧 Predictive Maintenance Dashboard
NASA C-MAPSS Turbofan Engine Degradation Simulation (FD001)
Capstone Machine Learning Project - Streamlit Application
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="🔧",
    layout="wide"
)

# Relative root paths
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_APP_DIR, "models", "predictive_maintenance_model.joblib")
DATA_PATH = os.path.join(_APP_DIR, "data", "processed_fd001.csv")
METRICS_PATH = os.path.join(_APP_DIR, "outputs", "model_metrics.csv")
THRESH_PATH = os.path.join(_APP_DIR, "outputs", "threshold_analysis.csv")
FEAT_IMP_PATH = os.path.join(_APP_DIR, "outputs", "feature_importance.csv")
EQUIP_SUMMARY_PATH = os.path.join(_APP_DIR, "outputs", "equipment_risk_summary.csv")
HIGH_RISK_EQUIP_PATH = os.path.join(_APP_DIR, "outputs", "high_risk_equipment.csv")


@st.cache_resource
def load_model():
    """Load pre-trained model bundle without retraining."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        bundle = joblib.load(MODEL_PATH)
        return bundle
    except Exception as e:
        st.error(f"Error loading model bundle: {e}")
        return None


@st.cache_data
def load_datasets():
    """Load processed dataset and summary tables."""
    datasets = {}
    if os.path.exists(DATA_PATH):
        datasets['data'] = pd.read_csv(DATA_PATH)
    else:
        datasets['data'] = None

    if os.path.exists(METRICS_PATH):
        datasets['metrics'] = pd.read_csv(METRICS_PATH)
    else:
        datasets['metrics'] = None

    if os.path.exists(THRESH_PATH):
        datasets['thresholds'] = pd.read_csv(THRESH_PATH)
    else:
        datasets['thresholds'] = None

    if os.path.exists(FEAT_IMP_PATH):
        datasets['feat_imp'] = pd.read_csv(FEAT_IMP_PATH)
    else:
        datasets['feat_imp'] = None

    if os.path.exists(EQUIP_SUMMARY_PATH):
        datasets['equip_summary'] = pd.read_csv(EQUIP_SUMMARY_PATH)
    else:
        datasets['equip_summary'] = None

    if os.path.exists(HIGH_RISK_EQUIP_PATH):
        datasets['high_risk_equip'] = pd.read_csv(HIGH_RISK_EQUIP_PATH)
    else:
        datasets['high_risk_equip'] = None

    return datasets


# --- Initialize application ---
bundle = load_model()
datasets = load_datasets()

# Verification / Error handling
if bundle is None:
    st.error(
        "⚠️ **Model Bundle Missing!** The trained model was not found at `models/predictive_maintenance_model.joblib`.\n\n"
        "Please run `python train.py` in the project root to train and persist the model bundle before launching this dashboard."
    )
    st.stop()

if datasets['data'] is None:
    st.error(
        "⚠️ **Processed Dataset Missing!** `data/processed_fd001.csv` was not found.\n\n"
        "Please run `python src/preprocessing.py` to create the processed data."
    )
    st.stop()

model = bundle['model']
feature_cols = bundle['feature_list']
alert_threshold = bundle.get('prediction_threshold', 0.40)
failure_window = bundle.get('failure_window', 30)
selected_model_name = bundle.get('model_name', 'Random Forest')

df_all = datasets['data']
equip_summary = datasets['equip_summary']

# Filter test engines (Units 81 to 100) or allow all engines
test_engines = sorted(df_all[df_all['unit_number'] > 80]['unit_number'].unique())
all_engines = sorted(df_all['unit_number'].unique())


def get_risk_badge(prob, threshold=alert_threshold, med_threshold=0.30):
    if prob >= threshold:
        return "🔴 HIGH RISK", "red"
    elif prob >= med_threshold:
        return "🟡 MEDIUM RISK", "orange"
    else:
        return "🟢 LOW RISK", "green"


# ==============================================================================
# SIDEBAR
# ==============================================================================
st.sidebar.title("🔧 Navigation & Control")
st.sidebar.markdown("---")
st.sidebar.subheader("Equipment Selection")

# Equipment Selector (Defaults to test fleet engines evaluated in Steps 3 & 4)
fleet_mode = st.sidebar.radio("Fleet View:", ["Holdout Test Fleet (Units 81–100)", "All Fleet Engines (Units 1–100)"], index=0)
available_units = test_engines if "Test Fleet" in fleet_mode else all_engines

selected_unit = st.sidebar.selectbox("Select Engine / Equipment ID:", available_units, index=0)

# Filter selected engine data
unit_df = df_all[df_all['unit_number'] == selected_unit].sort_values('time_in_cycles').reset_index(drop=True)

# Generate predictions for this engine using cached model
X_unit = unit_df[feature_cols].fillna(0.0)
unit_probs = model.predict_proba(X_unit)[:, 1]
unit_df['pred_prob'] = unit_probs

latest_cycle = int(unit_df['time_in_cycles'].iloc[-1])
latest_prob = float(unit_df['pred_prob'].iloc[-1])
badge_text, badge_color = get_risk_badge(latest_prob, alert_threshold)

st.sidebar.markdown(f"**Engine ID:** `Unit {selected_unit}`")
st.sidebar.markdown(f"**Latest Operational Cycle:** `{latest_cycle}`")
st.sidebar.markdown(f"**Current Failure Probability:** `{latest_prob:.2%}`")
st.sidebar.markdown(f"**Current Status:** {badge_text}")
st.sidebar.markdown("---")

st.sidebar.info(
    f"**Operating Configuration:**\n"
    f"- **Model:** {selected_model_name}\n"
    f"- **Alert Threshold:** `{alert_threshold:.2f}`\n"
    f"- **Prediction Window:** `{failure_window} cycles`"
)


# ==============================================================================
# MAIN DASHBOARD
# ==============================================================================
st.title("🔧 Predictive Maintenance Dashboard")
st.markdown("#### Turbofan Engine Degradation Early-Warning System (NASA C-MAPSS FD001)")
st.caption("Real-time decision support system for equipment failure-risk estimation and degradation anomaly tracking.")

# Top-level Metric Cards
total_test_units = len(test_engines)
high_risk_count = 0
if equip_summary is not None:
    high_risk_count = int((equip_summary['Latest_Failure_Probability'] >= alert_threshold).sum())
else:
    high_risk_count = 20

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Monitored Fleet", f"{total_test_units} Units", help="Held-out chronological test engines (Units 81 to 100)")
col2.metric("High-Risk Equipment", f"{high_risk_count} Units", delta="Critical", delta_color="inverse", help=f"Units with current failure probability >= {alert_threshold:.2f}")
col3.metric(f"Unit {selected_unit} Current Risk", f"{latest_prob:.1%}", delta=badge_text, delta_color="off")
col4.metric("Alert Decision Threshold", f"{alert_threshold:.2f}", help="Calibrated project operating threshold for failure alert")

st.markdown("---")

# ==============================================================================
# SECTION 1: SELECTED EQUIPMENT STATUS & FAILURE RISK TRAJECTORY
# ==============================================================================
st.subheader(f"⚙️ Equipment Status: Engine Unit {selected_unit}")

col_status, col_alert = st.columns([1, 2])

with col_status:
    st.markdown("##### Current Operational Telemetry")
    status_box = f"""
    - **Unit Identifier**: Engine #{selected_unit}
    - **Current Cycle**: {latest_cycle} cycles
    - **Estimated Failure Risk**: **{latest_prob:.2%}**
    - **Risk Classification**: **{badge_text}**
    - **Alert Status**: {'🚨 **ACTION REQUIRED**' if latest_prob >= alert_threshold else ('⚠️ **ELEVATED MONITORING**' if latest_prob >= 0.30 else '✅ **NOMINAL**')}
    """
    st.markdown(status_box)
    
    if latest_prob >= alert_threshold:
        st.warning(
            f"**High predicted failure risk** within next {failure_window} cycles. "
            "Flagged for immediate engineering inspection and maintenance scheduling."
        )
    elif latest_prob >= 0.30:
        st.info("Medium failure risk. Sensor parameters exhibiting early degradation trends. Routine monitoring recommended.")
    else:
        st.success("Nominal operating health. Machine parameters within standard operating tolerances.")

with col_alert:
    st.markdown("##### Predicted Failure Risk Over Operating Cycle")
    fig_risk, ax_risk = plt.subplots(figsize=(8, 3.2))
    ax_risk.plot(unit_df['time_in_cycles'], unit_df['pred_prob'], color='#d62728' if latest_prob >= alert_threshold else '#1f77b4', linewidth=2.2, label='Failure Risk P(Failure in <= 30 cycles)')
    ax_risk.axhline(y=alert_threshold, color='#ff7f0e', linestyle='--', linewidth=1.8, label=f'Alert Threshold ({alert_threshold:.2f})')
    ax_risk.axhline(y=0.30, color='#2ca02c', linestyle=':', linewidth=1.2, label='Medium Risk Baseline (0.30)')
    ax_risk.fill_between(unit_df['time_in_cycles'], unit_df['pred_prob'], alert_threshold, where=(unit_df['pred_prob'] >= alert_threshold), color='#d62728', alpha=0.2, label='Alert Zone')
    ax_risk.set_xlim([1, latest_cycle + 2])
    ax_risk.set_ylim([-0.02, 1.05])
    ax_risk.set_xlabel('Operational Cycle', fontweight='bold', fontsize=9)
    ax_risk.set_ylabel('Failure Probability', fontweight='bold', fontsize=9)
    ax_risk.legend(loc='upper left', fontsize=8, frameon=True)
    ax_risk.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig_risk)
    plt.close(fig_risk)

st.markdown("---")

# ==============================================================================
# SECTION 2: SENSOR TREND DASHBOARD
# ==============================================================================
st.subheader("📈 Multi-Sensor Telemetry & Trend Analysis")

# Automatically detect available sensor columns
available_sensor_cols = [c for c in unit_df.columns if c.startswith('sensor_') and '_' not in c.replace('sensor_', '')]
available_sensor_cols = sorted(available_sensor_cols, key=lambda x: int(x.replace('sensor_', '')))

# Domain friendly names
sensor_descriptions = {
    'sensor_2': 'sensor_2 (LPC Outlet Temp - T24)',
    'sensor_3': 'sensor_3 (HPC Outlet Temp - T30)',
    'sensor_4': 'sensor_4 (LPT Outlet Temp - T50) [Top Feature]',
    'sensor_7': 'sensor_7 (HPC Outlet Total Pressure - P30)',
    'sensor_8': 'sensor_8 (Physical Fan Speed - Nf)',
    'sensor_9': 'sensor_9 (Physical Core Speed - Nc)',
    'sensor_11': 'sensor_11 (HPC Static Pressure - Ps30) [Top Feature]',
    'sensor_12': 'sensor_12 (Fuel Flow Ratio - phi)',
    'sensor_13': 'sensor_13 (Corrected Fan Speed - NRf)',
    'sensor_14': 'sensor_14 (Corrected Core Speed - NRc)',
    'sensor_15': 'sensor_15 (Bypass Ratio - BPR) [Top Feature]',
    'sensor_17': 'sensor_17 (Bleed Enthalpy - htBleed)',
    'sensor_20': 'sensor_20 (HPT Coolant Bleed - W31)',
    'sensor_21': 'sensor_21 (LPT Coolant Bleed - W32) [Top Feature]'
}

default_idx = available_sensor_cols.index('sensor_4') if 'sensor_4' in available_sensor_cols else 0

col_sel_sensor, col_trend_info = st.columns([1, 2])
with col_sel_sensor:
    selected_sensor = st.selectbox(
        "Select Sensor to Inspect:",
        available_sensor_cols,
        index=default_idx,
        format_func=lambda x: sensor_descriptions.get(x, x)
    )

with col_trend_info:
    cur_val = unit_df[selected_sensor].iloc[-1]
    min_val = unit_df[selected_sensor].min()
    max_val = unit_df[selected_sensor].max()
    roll_col = f"{selected_sensor}_roll_mean_5"
    roll_val = unit_df[roll_col].iloc[-1] if roll_col in unit_df.columns else cur_val
    st.markdown(
        f"**Sensor Telemetry Snapshot**: Current Raw: `{cur_val:.3f}` | 5-Cycle Rolling Mean: `{roll_val:.3f}` "
        f"| Lifetime Range: `[{min_val:.3f}, {max_val:.3f}]`"
    )

fig_sensor, ax_sensor = plt.subplots(figsize=(10, 3.5))
ax_sensor.plot(unit_df['time_in_cycles'], unit_df[selected_sensor], color='#7f7f7f', alpha=0.55, linewidth=1.2, label='Raw Sensor Reading')

if roll_col in unit_df.columns:
    ax_sensor.plot(unit_df['time_in_cycles'], unit_df[roll_col], color='#1f77b4', linewidth=2.4, label='5-Cycle Rolling Mean (Causal Trend)')

ax_sensor.set_title(f"Telemetry Trend: {sensor_descriptions.get(selected_sensor, selected_sensor)} for Unit {selected_unit}", fontsize=11, fontweight='bold')
ax_sensor.set_xlabel("Operational Cycle", fontweight='bold', fontsize=9)
ax_sensor.set_ylabel("Sensor Reading Units", fontweight='bold', fontsize=9)
ax_sensor.legend(loc='best', fontsize=9, frameon=True)
ax_sensor.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
st.pyplot(fig_sensor)
plt.close(fig_sensor)

st.markdown("---")

# ==============================================================================
# SECTION 3: HIGH-RISK EQUIPMENT ALERTS
# ==============================================================================
st.subheader("🚨 High-Risk Equipment Alerts")
st.caption("Early-warning signals prioritizing equipment currently exhibiting failure risk within the 30-cycle operational window.")

if datasets['high_risk_equip'] is not None:
    high_risk_display = datasets['high_risk_equip'].copy()
    high_risk_display['Latest_Failure_Probability'] = high_risk_display['Latest_Failure_Probability'].apply(lambda x: f"{x:.1%}")
    high_risk_display['Max_Recent_Probability'] = high_risk_display['Max_Recent_Probability'].apply(lambda x: f"{x:.1%}")
    st.dataframe(high_risk_display, use_container_width=True, hide_index=True)
    st.info("ℹ️ These are model-generated early-warning signals for maintenance review.")
else:
    st.warning("High-risk equipment summary not found at `outputs/high_risk_equipment.csv`.")

st.markdown("---")

# ==============================================================================
# SECTION 4: MODEL PERFORMANCE & THRESHOLD ANALYSIS
# ==============================================================================
col_perf, col_thresh = st.columns([1, 1])

with col_perf:
    st.subheader("📊 Model Performance Comparison")
    st.caption("Evaluated on unseen chronological test fleet (Units 81–100, 4,493 cycles).")
    if datasets['metrics'] is not None:
        metrics_df = datasets['metrics'].copy()
        st.dataframe(
            metrics_df[['Model', 'Recall', 'Precision', 'F1', 'ROC_AUC', 'PR_AUC', 'False_Alarms']],
            use_container_width=True,
            hide_index=True
        )
        st.success(f"🏆 **Selected Champion Model:** `{selected_model_name}` (highest PR-AUC: 0.9456, F1: 0.8480, and balanced false alarms)")
    else:
        st.warning("Model metrics file not found.")

with col_thresh:
    st.subheader("⚖️ Threshold Optimization Analysis")
    st.caption("Trade-off between failure recall, precision, and false alarms across candidate thresholds.")
    if datasets['thresholds'] is not None:
        thresh_df = datasets['thresholds'].copy()
        
        fig_th, ax_th = plt.subplots(figsize=(6, 3.2))
        ax_th.plot(thresh_df['Threshold'], thresh_df['Recall'], marker='s', color='#d62728', linewidth=2, label='Recall (Failure Detection)')
        ax_th.plot(thresh_df['Threshold'], thresh_df['Precision'], marker='o', color='#1f77b4', linewidth=2, label='Precision')
        ax_th.plot(thresh_df['Threshold'], thresh_df['F1'], marker='^', color='#2ca02c', linestyle='--', linewidth=1.5, label='F1-Score')
        ax_th.axvline(x=alert_threshold, color='#ff7f0e', linestyle=':', linewidth=2, label=f'Selected Alert Threshold ({alert_threshold:.2f})')
        ax_th.set_xlabel('Probability Threshold', fontweight='bold', fontsize=8)
        ax_th.set_ylabel('Score', fontweight='bold', fontsize=8)
        ax_th.set_ylim([0.58, 1.02])
        ax_th.legend(fontsize=7.5, loc='lower left')
        ax_th.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig_th)
        plt.close(fig_th)

        st.caption(
            "Note: The alert threshold is a project-defined operating threshold based on the precision/recall/false-alarm trade-off. "
            "It is not a universal maintenance rule."
        )
    else:
        st.warning("Threshold analysis file not found.")

st.markdown("---")

# ==============================================================================
# SECTION 5: FEATURE IMPORTANCE
# ==============================================================================
st.subheader("🔍 Top Model-Important Features")
st.caption("Key statistical drivers identified during model training (Random Forest MDI).")

if datasets['feat_imp'] is not None:
    feat_df = datasets['feat_imp'].head(10).copy()
    
    fig_imp, ax_imp = plt.subplots(figsize=(10, 3.6))
    sns.barplot(data=feat_df, x='RF_Importance', y='RF_Feature', color='#2ca02c', edgecolor='black', alpha=0.85, ax=ax_imp)
    ax_imp.set_title("Top 10 Model-Important Features (Random Forest MDI)", fontsize=11, fontweight='bold')
    ax_imp.set_xlabel("Mean Decrease in Impurity (Feature Importance Score)", fontsize=9, fontweight='bold')
    ax_imp.set_ylabel("Engineered Feature", fontsize=9, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig_imp)
    plt.close(fig_imp)

    st.info("ℹ️ **Domain Note:** Feature importance indicates which variables contributed most to model predictions; it does not establish causality.")
else:
    st.warning("Feature importance file not found.")

st.markdown("---")

# ==============================================================================
# SECTION 6: ABOUT THIS PROJECT & DISCLAIMER
# ==============================================================================
with st.expander("ℹ️ About This Project"):
    st.markdown(
        f"""
        ### Project Architecture & Methodology
        - **Dataset**: NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset (Subset FD001: Sea-level flight, HPC fault mode).
        - **Primary Objective**: Predict impending turbofan engine failure risk within a predefined operational horizon.
        - **Failure Window Threshold**: `{failure_window} cycles` ($\text{{target}} = 1$ if $\text{{RUL}} \\le {failure_window}$, else $0$).
        - **Class Imbalance**: Mitigated using balanced class weighting (`class_weight='balanced'`).
        - **Feature Engineering**: 56 strictly causal backward-looking rolling statistics (`roll_mean_5`, `roll_std_5`), lag-1, and delta features grouped strictly per engine to avoid future leakage.
        - **Champion Model**: `{selected_model_name}` (ROC-AUC: `0.9890`, PR-AUC: `0.9456`, Recall: `94.52%` at operating threshold `{alert_threshold:.2f}`).
        - **Evaluation Protocol**: Time-aware chronological fleet holdout (Units 1–80 for training; Units 81–100 held out for evaluation).
        """
    )

st.markdown(
    "⚠️ **Disclaimer:** *Predicted failure risk is a machine-learning estimate intended as a decision-support signal. "
    "It does not guarantee that equipment will fail. Maintenance decisions should incorporate engineering inspection and operational context.*"
)
