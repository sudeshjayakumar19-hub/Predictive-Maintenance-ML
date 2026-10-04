"""
NASA C-MAPSS FD001 Threshold Tuning and High-Risk Alert System
Predictive Maintenance ML Capstone Project - Step 4

This module:
1. Loads the saved champion model (Random Forest) from Step 3.
2. Generates predicted probabilities on the chronological test set (Units 81-100).
3. Conducts granular threshold analysis across [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80].
4. Generates threshold diagnostic plots (Precision-Recall vs Threshold, False Alarms vs Threshold).
5. Defines an operationally tuned project threshold (0.40) and 3-tier risk system (LOW, MEDIUM, HIGH).
6. Identifies model-important sensor indicators and synthesizes non-causal alert rationales.
7. Produces cycle-level high-risk alerts and equipment-level risk summaries.
8. Generates equipment risk distribution and high-risk equipment visualizations.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Global configuration
MODEL_PATH = 'models/predictive_maintenance_model.joblib'
DATA_PATH = 'data/processed_fd001.csv'
OUTPUTS_DIR = 'outputs'
PLOTS_DIR = 'outputs/plots'

# Tuned project operating threshold
SELECTED_THRESHOLD = 0.40
MEDIUM_THRESHOLD = 0.30

THRESHOLDS_TO_EVALUATE = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]


def load_model_and_test_data():
    """Load persisted champion model bundle and chronological test set."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model bundle not found at {MODEL_PATH}. Run train.py first.")

    bundle = joblib.load(MODEL_PATH)
    model = bundle['model']
    feature_cols = bundle['feature_list']

    df = pd.read_csv(DATA_PATH)
    test_df = df[df['unit_number'] > 80].copy().reset_index(drop=True)

    X_test = test_df[feature_cols]
    y_test = test_df['target']

    probs = model.predict_proba(X_test)[:, 1]
    test_df['failure_probability'] = probs

    return bundle, model, feature_cols, test_df, X_test, y_test


def evaluate_thresholds(y_test: pd.Series, probs: np.ndarray, thresholds: list = THRESHOLDS_TO_EVALUATE):
    """
    Evaluate multiple candidate thresholds and produce comparison dataframe.
    """
    records = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        alerts = tp + fp

        records.append({
            'Threshold': round(t, 2),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1': round(f1, 4),
            'False_Positives': int(fp),
            'False_Negatives': int(fn),
            'False_Alarm_Rate': round(far, 4),
            'Alerts': int(alerts)
        })

    thresh_df = pd.DataFrame(records)
    csv_path = os.path.join(OUTPUTS_DIR, 'threshold_analysis.csv')
    thresh_df.to_csv(csv_path, index=False)
    print(f"Saved threshold analysis to: {csv_path}")
    return thresh_df


def plot_threshold_diagnostics(thresh_df: pd.DataFrame, output_dir: str = PLOTS_DIR):
    """
    Plot:
    1. 13_threshold_precision_recall.png
    2. 14_threshold_false_alarms.png
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style='whitegrid')

    # Plot 13: Precision and Recall vs Threshold
    plt.figure(figsize=(8, 5))
    plt.plot(thresh_df['Threshold'], thresh_df['Precision'], marker='o', color='#1f77b4', linewidth=2.2, label='Precision')
    plt.plot(thresh_df['Threshold'], thresh_df['Recall'], marker='s', color='#d62728', linewidth=2.2, label='Recall')
    plt.plot(thresh_df['Threshold'], thresh_df['F1'], marker='^', color='#2ca02c', linestyle='--', linewidth=1.8, label='F1-Score')
    plt.axvline(x=SELECTED_THRESHOLD, color='#ff7f0e', linestyle=':', linewidth=2.2,
                label=f'Selected Operating Threshold ({SELECTED_THRESHOLD:.2f})')
    plt.title('Precision, Recall, and F1-Score vs Probability Threshold', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Probability Threshold', fontsize=11, fontweight='bold')
    plt.ylabel('Metric Score', fontsize=11, fontweight='bold')
    plt.ylim([0.55, 1.02])
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    p13_path = os.path.join(output_dir, '13_threshold_precision_recall.png')
    plt.savefig(p13_path, dpi=300)
    plt.close()
    print(f"Saved: {p13_path}")

    # Plot 14: Threshold vs False Alarms (False Positives)
    plt.figure(figsize=(8, 5))
    bars = plt.bar(thresh_df['Threshold'].astype(str), thresh_df['False_Positives'], color='#d9534f', edgecolor='black', alpha=0.85, width=0.45)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 8, f'{int(h)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.axvline(x=str(SELECTED_THRESHOLD), color='#1f77b4', linestyle='--', linewidth=2,
                label=f'Selected Operating Threshold ({SELECTED_THRESHOLD:.2f})')
    plt.title('Threshold vs False Alarms (False Positive Predictions)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Probability Threshold', fontsize=11, fontweight='bold')
    plt.ylabel('Number of False Alarms (FP)', fontsize=11, fontweight='bold')
    plt.ylim(0, thresh_df['False_Positives'].max() * 1.15)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    p14_path = os.path.join(output_dir, '14_threshold_false_alarms.png')
    plt.savefig(p14_path, dpi=300)
    plt.close()
    print(f"Saved: {p14_path}")


def assign_risk_level(prob: float, high_thresh: float = SELECTED_THRESHOLD, med_thresh: float = MEDIUM_THRESHOLD) -> str:
    """Classify risk probability into 3-tier operational system."""
    if prob >= high_thresh:
        return 'HIGH'
    elif prob >= med_thresh:
        return 'MEDIUM'
    else:
        return 'LOW'


def generate_alert_reason(row: pd.Series) -> str:
    """
    Generate non-causal, model-associated descriptive text highlighting key sensor patterns.
    """
    s4 = row.get('sensor_4_roll_mean_5', row.get('sensor_4', np.nan))
    s11 = row.get('sensor_11_roll_mean_5', row.get('sensor_11', np.nan))
    s15 = row.get('sensor_15_roll_mean_5', row.get('sensor_15', np.nan))
    s21 = row.get('sensor_21_roll_mean_5', row.get('sensor_21', np.nan))
    
    return (
        f"Elevated recent sensor trend in model-important features: "
        f"sensor_4 LPT outlet temp (roll mean={s4:.2f}R), "
        f"sensor_11 HPC static press (roll mean={s11:.2f}psia), "
        f"sensor_15 bypass ratio (roll mean={s15:.4f}), "
        f"sensor_21 LPT coolant bleed (roll mean={s21:.4f}lbm/s)."
    )


def create_alert_tables(test_df: pd.DataFrame):
    """
    Create:
    - outputs/high_risk_alerts.csv
    - outputs/high_risk_equipment.csv
    """
    test_df['predicted_label'] = (test_df['failure_probability'] >= SELECTED_THRESHOLD).astype(int)
    test_df['risk_level'] = test_df['failure_probability'].apply(assign_risk_level)
    test_df['actual_failure_label'] = test_df['target'].apply(lambda x: 'Failure-Risk (RUL <= 30)' if x == 1 else 'Healthy')

    # Add descriptive non-causal reason for all records
    test_df['alert_reason'] = test_df.apply(generate_alert_reason, axis=1)

    # Key columns to present
    selected_sensor_cols = [
        'sensor_4', 'sensor_4_roll_mean_5',
        'sensor_11', 'sensor_11_roll_mean_5',
        'sensor_15', 'sensor_15_roll_mean_5',
        'sensor_21', 'sensor_21_roll_mean_5'
    ]

    base_cols = [
        'unit_number', 'time_in_cycles', 'failure_probability',
        'predicted_label', 'risk_level', 'actual_failure_label',
        'alert_reason'
    ]
    
    output_cols = base_cols + selected_sensor_cols

    # High-Risk alerts table (all alerts where risk_level == 'HIGH' or prob >= 0.40)
    alerts_df = test_df[test_df['risk_level'] == 'HIGH'][output_cols].copy()
    alerts_df = alerts_df.sort_values(by='failure_probability', ascending=False).reset_index(drop=True)
    alerts_csv_path = os.path.join(OUTPUTS_DIR, 'high_risk_alerts.csv')
    alerts_df.to_csv(alerts_csv_path, index=False)
    print(f"Saved {len(alerts_df)} high-risk alert records to: {alerts_csv_path}")

    return alerts_df, test_df


def create_equipment_risk_summary(test_df: pd.DataFrame):
    """
    Create equipment-level summary:
    - outputs/equipment_risk_summary.csv
    - outputs/high_risk_equipment.csv (filtered strictly to HIGH risk)
    """
    summary_rows = []
    for unit_id, group in test_df.groupby('unit_number'):
        group_sorted = group.sort_values('time_in_cycles')
        latest_row = group_sorted.iloc[-1]

        latest_cycle = int(latest_row['time_in_cycles'])
        latest_prob = float(latest_row['failure_probability'])
        max_recent_prob = float(group_sorted['failure_probability'].tail(10).max())
        high_risk_alerts = int((group_sorted['failure_probability'] >= SELECTED_THRESHOLD).sum())
        current_risk_level = assign_risk_level(latest_prob)
        actual_failure_status = 'Failure-Risk (RUL <= 30)' if latest_row['target'] == 1 else 'Healthy'

        summary_rows.append({
            'Equipment_ID': int(unit_id),
            'Latest_Cycle': latest_cycle,
            'Latest_Failure_Probability': round(latest_prob, 4),
            'Max_Recent_Probability': round(max_recent_prob, 4),
            'Current_Risk_Level': current_risk_level,
            'High_Risk_Alert_Count': high_risk_alerts,
            'Actual_Failure_Status': actual_failure_status
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(by='Latest_Failure_Probability', ascending=False).reset_index(drop=True)
    summary_csv = os.path.join(OUTPUTS_DIR, 'equipment_risk_summary.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved equipment risk summary to: {summary_csv}")

    # High-Risk Equipment table (filter where Current_Risk_Level == 'HIGH')
    high_risk_equipment_df = summary_df[summary_df['Current_Risk_Level'] == 'HIGH'].reset_index(drop=True)
    high_risk_equip_csv = os.path.join(OUTPUTS_DIR, 'high_risk_equipment.csv')
    high_risk_equipment_df.to_csv(high_risk_equip_csv, index=False)
    print(f"Saved {len(high_risk_equipment_df)} high-risk equipment records to: {high_risk_equip_csv}")

    return summary_df, high_risk_equipment_df


def plot_alert_visualizations(test_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: str = PLOTS_DIR):
    """
    Plot:
    1. 15_equipment_risk_distribution.png
    2. 16_high_risk_equipment.png
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style='whitegrid')

    # Plot 15: Probability distribution with threshold and zones
    plt.figure(figsize=(9, 5))
    probs = test_df['failure_probability']
    sns.histplot(probs, bins=40, kde=True, color='#2b5c8f', edgecolor='black', alpha=0.6)
    
    # Shade risk zones
    plt.axvspan(0.0, MEDIUM_THRESHOLD, color='#28a745', alpha=0.10, label='LOW Risk (< 0.30)')
    plt.axvspan(MEDIUM_THRESHOLD, SELECTED_THRESHOLD, color='#ffc107', alpha=0.15, label='MEDIUM Risk (0.30 - 0.40)')
    plt.axvspan(SELECTED_THRESHOLD, 1.0, color='#dc3545', alpha=0.15, label='HIGH Risk (>= 0.40)')
    
    plt.axvline(x=SELECTED_THRESHOLD, color='#d62728', linestyle='--', linewidth=2.5,
                label=f'Selected Operating Threshold ({SELECTED_THRESHOLD:.2f})')
    
    plt.title('Distribution of Predicted Failure Probabilities on Test Fleet', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Predicted Failure Risk Probability P(Failure within 30 cycles)', fontsize=11, fontweight='bold')
    plt.ylabel('Cycle Count', fontsize=11, fontweight='bold')
    plt.xlim(0.0, 1.0)
    plt.legend(loc='upper center', frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5)
    plt.tight_layout()
    p15_path = os.path.join(output_dir, '15_equipment_risk_distribution.png')
    plt.savefig(p15_path, dpi=300)
    plt.close()
    print(f"Saved: {p15_path}")

    # Plot 16: High-risk equipment ranking
    plt.figure(figsize=(10, 6))
    sorted_summary = summary_df.sort_values('High_Risk_Alert_Count', ascending=True)
    
    colors = ['#dc3545' if r == 'HIGH' else '#ffc107' for r in sorted_summary['Current_Risk_Level']]
    bars = plt.barh(sorted_summary['Equipment_ID'].astype(str), sorted_summary['High_Risk_Alert_Count'], color=colors, edgecolor='black', alpha=0.85)
    
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 1, bar.get_y() + bar.get_height()/2., f'{int(w)} alerts', va='center', fontsize=9, fontweight='bold')
        
    plt.title('Fleet Units Ranked by Total High-Risk Alerts (Threshold >= 0.40)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Number of High-Risk Operational Cycles Alerted', fontsize=11, fontweight='bold')
    plt.ylabel('Turbofan Engine Unit ID', fontsize=11, fontweight='bold')
    plt.xlim(0, sorted_summary['High_Risk_Alert_Count'].max() * 1.15)
    plt.tight_layout()
    p16_path = os.path.join(output_dir, '16_high_risk_equipment.png')
    plt.savefig(p16_path, dpi=300)
    plt.close()
    print(f"Saved: {p16_path}")


def run_alert_system():
    """Execute complete Step 4 threshold tuning and alert system pipeline."""
    print("=" * 70)
    print("STARTING THRESHOLD TUNING & ALERT SYSTEM (STEP 4)")
    print("=" * 70)

    # 1. Load Model and Data
    bundle, model, feature_cols, test_df, X_test, y_test = load_model_and_test_data()
    print(f"Loaded test set with {len(test_df)} observations across {test_df['unit_number'].nunique()} engines.")

    # 2. Threshold Analysis
    thresh_df = evaluate_thresholds(y_test, test_df['failure_probability'].values)
    print("\n--- THRESHOLD ANALYSIS RESULTS ---")
    print(thresh_df.to_string(index=False))

    # 3. Plots: Precision-Recall vs Threshold, False Alarms vs Threshold
    plot_threshold_diagnostics(thresh_df, output_dir=PLOTS_DIR)

    # 4. Generate Alert Tables
    alerts_df, test_df = create_alert_tables(test_df)

    # 5. Equipment Risk Summary
    summary_df, high_risk_equipment_df = create_equipment_risk_summary(test_df)

    # 6. Plots: Risk Distribution and Equipment Ranking
    plot_alert_visualizations(test_df, summary_df, output_dir=PLOTS_DIR)

    print("\n" + "=" * 70)
    print("STEP 4 PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)

    return thresh_df, alerts_df, summary_df, high_risk_equipment_df


if __name__ == '__main__':
    run_alert_system()
